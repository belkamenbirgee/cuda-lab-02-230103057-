
import numpy as np
from numba import cuda


# ============================================================
# TASK 2: 1D BOUNDARY STENCIL & HALO PROTECTION
# ============================================================

THREADS_PER_BLOCK = 256


@cuda.jit
def stencil_1d(d_in, d_out, N):

    idx = cuda.grid(1)

    if idx < N:

        # Boundary clamping
        if idx == 0:
            left = d_in[0]
        else:
            left = d_in[idx - 1]

        if idx == N - 1:
            right = d_in[N - 1]
        else:
            right = d_in[idx + 1]

        center = d_in[idx]

        # 3-point smoothing filter
        d_out[idx] = (
            0.25 * left
            + 0.50 * center
            + 0.25 * right
        )


def cpu_stencil(arr):

    padded = np.pad(arr, (1, 1), mode='edge')

    return (
        0.25 * padded[:-2]
        + 0.50 * padded[1:-1]
        + 0.25 * padded[2:]
    )


def run_stencil(h_in):

    h_in = np.asarray(h_in, dtype=np.float32)

    N = h_in.size

    h_out = np.empty_like(h_in)

    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(h_in)

    blocks_per_grid = (
        N + THREADS_PER_BLOCK - 1
    ) // THREADS_PER_BLOCK

    stencil_1d[
        blocks_per_grid,
        THREADS_PER_BLOCK
    ](
        d_in,
        d_out,
        N
    )

    cuda.synchronize()

    d_out.copy_to_host(h_out)

    return h_out


def main():

    print("=" * 70)
    print("TASK 2: 1D BOUNDARY STENCIL & HALO PROTECTION")
    print("=" * 70)

    # Required odd, non-power-of-two size
    N = 10007

    print(f"Array Size: {N}")
    print(f"Threads Per Block: {THREADS_PER_BLOCK}")

    blocks_per_grid = (
        N + THREADS_PER_BLOCK - 1
    ) // THREADS_PER_BLOCK

    print(f"Blocks Per Grid: {blocks_per_grid}")
    print(
        f"Total Threads Launched: "
        f"{blocks_per_grid * THREADS_PER_BLOCK}"
    )

    # Test input
    h_in = np.sin(
        np.linspace(0, 10, N)
    ).astype(np.float32)

    # GPU result
    h_out_gpu = run_stencil(h_in)

    # CPU reference
    cpu_ref = cpu_stencil(h_in)

    # Numerical validation
    max_delta = np.max(
        np.abs(h_out_gpu - cpu_ref)
    )

    print()
    print(f"Maximum Delta: {max_delta:.10f}")

    assert np.allclose(
        h_out_gpu,
        cpu_ref,
        atol=1e-4
    )

    print()
    print(
        f"TASK 2 PASSED: MAX DELTA = "
        f"{max_delta:.10f}"
    )

    print()
    print("Input Sample:")
    print(h_in[:10])

    print()
    print("GPU Output Sample:")
    print(h_out_gpu[:10])

    print()
    print("CPU Reference Sample:")
    print(cpu_ref[:10])

    print("=" * 70)


if __name__ == "__main__":
    main()
