
import numpy as np
from numba import cuda


# ============================================================
# TASK 3: ARBITRARY-SIZE VECTOR SCALING
#            VIA GRID-STRIDE LOOP
# ============================================================

THREADS_PER_BLOCK = 256
BLOCKS_PER_GRID = 64


@cuda.jit
def grid_stride_scale_kernel(d_arr, factor, N):

    # First global thread index
    start = cuda.grid(1)

    # Total number of threads in the entire grid
    stride = cuda.gridsize(1)

    # Grid-stride loop
    for i in range(start, N, stride):
        d_arr[i] = d_arr[i] * factor


def run_grid_stride(h_arr, factor):

    h_arr = np.asarray(h_arr, dtype=np.float32)

    N = h_arr.size

    d_arr = cuda.to_device(h_arr)

    # Exactly 64 blocks × 256 threads
    grid_stride_scale_kernel[
        BLOCKS_PER_GRID,
        THREADS_PER_BLOCK
    ](
        d_arr,
        np.float32(factor),
        N
    )

    cuda.synchronize()

    result = d_arr.copy_to_host()

    return result


def main():

    print("=" * 70)
    print("TASK 3: GRID-STRIDE VECTOR SCALING")
    print("=" * 70)

    # Arbitrary vector size
    N = 100000

    factor = 4.25

    total_threads = (
        BLOCKS_PER_GRID * THREADS_PER_BLOCK
    )

    print(f"Vector Size: {N}")
    print(f"Scaling Factor: {factor}")
    print(f"Threads Per Block: {THREADS_PER_BLOCK}")
    print(f"Blocks Per Grid: {BLOCKS_PER_GRID}")
    print(f"Total Hardware Threads Launched: {total_threads}")

    # Create input
    h_arr = np.ones(
        N,
        dtype=np.float32
    )

    # Run GPU grid-stride kernel
    result = run_grid_stride(
        h_arr,
        factor
    )

    # Verify all elements
    expected = np.float32(factor)

    correct = np.allclose(
        result,
        expected
    )

    max_delta = np.max(
        np.abs(result - expected)
    )

    print()
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    print(f"Expected Value: {expected}")
    print(f"Actual First 10 Values: {result[:10]}")
    print(f"Maximum Delta: {max_delta:.10f}")
    print(f"All Elements Correct: {correct}")

    assert correct, (
        "Task 3 elements not uniformly scaled"
    )

    print()
    print("TASK 3 PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
