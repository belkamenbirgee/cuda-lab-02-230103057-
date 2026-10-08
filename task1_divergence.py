
import time
import numpy as np
from numba import cuda


# ============================================================
# TASK 1: WARP DIVERGENCE MICROBENCHMARK
# ============================================================

ITERATIONS = 1000
THREADS_PER_BLOCK = 256


# ------------------------------------------------------------
# Kernel A: Uniform Path
# Every thread executes the same arithmetic.
# ------------------------------------------------------------
@cuda.jit
def kernel_uniform(d_in, d_out):
    idx = cuda.grid(1)

    if idx < d_in.size:
        x = d_in[idx]

        for _ in range(ITERATIONS):
            x = x * 1.000001 + 0.000001

        d_out[idx] = x


# ------------------------------------------------------------
# Kernel B: Full Divergence - Interleaved
# Even threads use Path 1.
# Odd threads use Path 2.
#
# Adjacent threads in every warp take different paths.
# ------------------------------------------------------------
@cuda.jit
def kernel_divergent(d_in, d_out):
    idx = cuda.grid(1)

    if idx < d_in.size:
        x = d_in[idx]

        if idx % 2 == 0:

            for _ in range(ITERATIONS):
                x = x * 1.000001 + 0.000001

        else:

            for _ in range(ITERATIONS):
                x = x - 0.000001
                x = x / 1.000001

        d_out[idx] = x


# ------------------------------------------------------------
# Kernel C: Warp-Aligned Branching
#
# All threads within a warp take the same path.
# ------------------------------------------------------------
@cuda.jit
def kernel_warp_aligned(d_in, d_out):
    idx = cuda.grid(1)

    if idx < d_in.size:
        x = d_in[idx]

        warp_id = idx // 32

        if warp_id % 2 == 0:

            for _ in range(ITERATIONS):
                x = x * 1.000001 + 0.000001

        else:

            for _ in range(ITERATIONS):
                x = x - 0.000001
                x = x / 1.000001

        d_out[idx] = x


# ------------------------------------------------------------
# Benchmark helper
# Kernel-only timing.
# Host-to-device and device-to-host transfers are excluded.
# ------------------------------------------------------------
def benchmark_kernel(kernel, d_in, d_out, blocks_per_grid):
    # Warm-up launch
    kernel[blocks_per_grid, THREADS_PER_BLOCK](d_in, d_out)
    cuda.synchronize()

    times = []

    for _ in range(10):
        start = time.perf_counter()

        kernel[blocks_per_grid, THREADS_PER_BLOCK](d_in, d_out)

        cuda.synchronize()

        end = time.perf_counter()

        times.append((end - start) * 1000.0)

    return np.mean(times)


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------
def main():

    print("=" * 70)
    print("TASK 1: WARP DIVERGENCE MICROBENCHMARK")
    print("=" * 70)

    # Large enough workload for meaningful timing
    N = 1_048_576

    print(f"Array Size: {N}")
    print(f"Iterations Per Element: {ITERATIONS}")
    print(f"Threads Per Block: {THREADS_PER_BLOCK}")

    blocks_per_grid = (N + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    print(f"Blocks Per Grid: {blocks_per_grid}")
    print(f"Total Threads Launched: "
          f"{blocks_per_grid * THREADS_PER_BLOCK}")

    # Host arrays
    h_input = np.ones(N, dtype=np.float32)
    h_output = np.zeros(N, dtype=np.float32)

    # Device arrays
    d_input = cuda.to_device(h_input)
    d_output = cuda.to_device(h_output)

    # --------------------------------------------------------
    # Kernel A
    # --------------------------------------------------------
    uniform_time = benchmark_kernel(
        kernel_uniform,
        d_input,
        d_output,
        blocks_per_grid
    )

    # --------------------------------------------------------
    # Kernel B
    # --------------------------------------------------------
    divergent_time = benchmark_kernel(
        kernel_divergent,
        d_input,
        d_output,
        blocks_per_grid
    )

    # --------------------------------------------------------
    # Kernel C
    # --------------------------------------------------------
    warp_aligned_time = benchmark_kernel(
        kernel_warp_aligned,
        d_input,
        d_output,
        blocks_per_grid
    )

    print()
    print("=" * 70)
    print("EMPIRICAL KERNEL-ONLY RESULTS")
    print("=" * 70)

    print(f"{'Kernel':<25} {'Average Time (ms)':>20}")
    print("-" * 70)

    print(f"{'A - Uniform Path':<25} "
          f"{uniform_time:>20.4f}")

    print(f"{'B - Full Divergence':<25} "
          f"{divergent_time:>20.4f}")

    print(f"{'C - Warp-Aligned':<25} "
          f"{warp_aligned_time:>20.4f}")

    print("=" * 70)

    # Relative comparison
    print()
    print("RELATIVE PERFORMANCE")
    print("-" * 70)

    print(
        f"Divergent / Uniform: "
        f"{divergent_time / uniform_time:.2f}x"
    )

    print(
        f"Warp-Aligned / Uniform: "
        f"{warp_aligned_time / uniform_time:.2f}x"
    )

    print(
        f"Divergent / Warp-Aligned: "
        f"{divergent_time / warp_aligned_time:.2f}x"
    )

    print("=" * 70)

    # Copy a small sample back for sanity checking
    result = d_output.copy_to_host()

    print()
    print("Output Sample:")
    print(result[:10])

    print()
    print("TASK 1 COMPLETED")


if __name__ == "__main__":
    main()
