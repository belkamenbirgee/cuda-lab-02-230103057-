
import numpy as np
from numba import cuda


# ============================================================
# TASK 4: 2D SPATIAL CONVOLUTION - SOBEL HORIZONTAL FILTER
# ============================================================

THREADS_X = 16
THREADS_Y = 16


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):

    # 2D global coordinates
    col, row = cuda.grid(2)

    # Boundary guard
    if row < rows and col < cols:

        # Outer border pixels are zero
        if (
            row == 0
            or row == rows - 1
            or col == 0
            or col == cols - 1
        ):
            d_out[row, col] = 0.0

        else:

            # Sobel-X:
            #
            # [-1  0  1]
            # [-2  0  2]
            # [-1  0  1]
            #
            value = (
                -1.0 * d_in[row - 1, col - 1]
                + 1.0 * d_in[row - 1, col + 1]
                - 2.0 * d_in[row, col - 1]
                + 2.0 * d_in[row, col + 1]
                - 1.0 * d_in[row + 1, col - 1]
                + 1.0 * d_in[row + 1, col + 1]
            )

            d_out[row, col] = value


def run_sobel(h_img):

    h_img = np.asarray(
        h_img,
        dtype=np.float32
    )

    rows, cols = h_img.shape

    # Allocate device memory
    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array_like(h_img)

    # 2D block dimensions
    threads_2d = (
        THREADS_X,
        THREADS_Y
    )

    # Dynamic grid geometry
    blocks_x = (
        cols + THREADS_X - 1
    ) // THREADS_X

    blocks_y = (
        rows + THREADS_Y - 1
    ) // THREADS_Y

    blocks_2d = (
        blocks_x,
        blocks_y
    )

    # Launch kernel
    sobel_x_kernel[
        blocks_2d,
        threads_2d
    ](
        d_in,
        d_out,
        rows,
        cols
    )

    cuda.synchronize()

    # Copy result back
    h_out = d_out.copy_to_host()

    return h_out


def main():

    print("=" * 70)
    print("TASK 4: 2D SPATIAL CONVOLUTION - SOBEL-X")
    print("=" * 70)

    # Required 2D input matrix
    rows = 512
    cols = 512

    h_img = np.ones(
        (rows, cols),
        dtype=np.float32
    )

    print(f"Image Shape: {rows} x {cols}")
    print(
        f"Threads Per Block: "
        f"{THREADS_X} x {THREADS_Y}"
    )

    blocks_x = (
        cols + THREADS_X - 1
    ) // THREADS_X

    blocks_y = (
        rows + THREADS_Y - 1
    ) // THREADS_Y

    print(
        f"Blocks in X: {blocks_x}"
    )

    print(
        f"Blocks in Y: {blocks_y}"
    )

    print(
        f"Total Blocks: "
        f"{blocks_x * blocks_y}"
    )

    # Run GPU Sobel-X
    result = run_sobel(h_img)

    print()
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Interior of a uniform image must have zero gradient
    interior_max = np.max(
        np.abs(result[1:-1, 1:-1])
    )

    # Borders must be zero
    top_border_ok = np.all(
        result[0, :] == 0.0
    )

    bottom_border_ok = np.all(
        result[-1, :] == 0.0
    )

    left_border_ok = np.all(
        result[:, 0] == 0.0
    )

    right_border_ok = np.all(
        result[:, -1] == 0.0
    )

    print(
        f"Maximum Interior Gradient: "
        f"{interior_max:.10f}"
    )

    print(
        f"Top Border Zero: {top_border_ok}"
    )

    print(
        f"Bottom Border Zero: {bottom_border_ok}"
    )

    print(
        f"Left Border Zero: {left_border_ok}"
    )

    print(
        f"Right Border Zero: {right_border_ok}"
    )

    # Exact verification
    assert interior_max < 1e-5
    assert top_border_ok
    assert bottom_border_ok
    assert left_border_ok
    assert right_border_ok

    print()
    print("TASK 4 PASSED")

    print()
    print("Center Pixel:", result[rows // 2, cols // 2])
    print("Top-Left Pixel:", result[0, 0])

    print("=" * 70)


if __name__ == "__main__":
    main()
