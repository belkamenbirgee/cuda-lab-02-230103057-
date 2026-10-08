
# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230103057

**Allocated GPU Node:** Tesla T4

**CUDA Compute Capability:** 7.5

**Official Verification Token:** 732FE7C6B9033219CD92


## 1. Environment

| Property | Value |
|---|---|
| GPU | Tesla T4 |
| Compute Capability | 7.5 |
| Streaming Multiprocessors | 40 |
| Warp Size | 32 |
| Maximum Threads Per Block | 1024 |
| Threads Per Block Used in Task 1 | 256 |
| Threads Per Block Used in Task 2 | 256 |
| Threads Per Block Used in Task 3 | 256 |
| 2D Threads Per Block in Task 4 | 16 x 16 |


## 2. Task 1 — Warp Divergence Microbenchmark

The experiment compares three CUDA kernels:

- Kernel A: Uniform Path
- Kernel B: Full Divergence with interleaved branches
- Kernel C: Warp-Aligned Branching

The benchmark uses 1,048,576 elements and 1,000 iterations per element.

Kernel execution time was measured using `time.perf_counter()` and `cuda.synchronize()`. Host-to-device and device-to-host transfer times were excluded.

### Empirical Results

| Kernel | Average Kernel Time |
|---|---:|
| A — Uniform Path | 25.9098 ms |
| B — Full Divergence | 93.7694 ms |
| C — Warp-Aligned | 46.7873 ms |

### Relative Performance

| Comparison | Result |
|---|---:|
| Divergent / Uniform | 3.62x |
| Warp-Aligned / Uniform | 1.81x |
| Divergent / Warp-Aligned | 2.00x |

### Observation

The full-divergence kernel was approximately 2 times slower than the warp-aligned version. In Kernel B, adjacent threads within each 32-thread warp take different branches, causing the GPU to execute the divergent paths serially within the warp. Kernel C assigns the same branch to an entire warp, avoiding intra-warp branch divergence.


## 3. Task 2 — 1D Boundary Stencil & Halo Protection

The CUDA kernel implements a three-point smoothing filter:

- 25% contribution from the left neighbor
- 50% contribution from the center element
- 25% contribution from the right neighbor

Boundary elements use clamping, meaning the missing neighbor is replaced by the nearest valid element.

The required odd, non-power-of-two test size was:

- N = 10,007
- Threads per block = 256
- Blocks per grid = 40
- Total threads launched = 10,240

The CUDA result was compared against the NumPy CPU reference implementation.

### Verification Result

```text
Maximum Delta: 0.0000000596
TASK 2 PASSED
