# MPSGraph fp16 / bf16 normalization probes (2026-09-30)

Standalone Swift programs (no project code) that run MPSGraph's `mean`, `variance(of:mean:)` and `normalize` — the ops `ChessNetwork`'s BatchNorm and LayerNorm use — in fp32, fp16 and bf16, on macOS 27.2 (Apple Silicon GPU). Build and run each with `swiftc -O <file>.swift -o probe && ./probe` (the "Incompatible element type for ANE" lines on stderr are MPSGraph declining the Neural Engine for bf16; harmless).

- `mpsgraph_norm_formats.swift` — variance boundary, isolated large values, mean ≫ std, tiny variance vs ε, all three formats (graph also returns mean and variance).
- `mpsgraph_fp16_fused_normalize.swift` — the same fp16 normalize with and without the mean/variance also requested as outputs; threshold scan.
- `mpsgraph_norm_error_by_size.swift` — worst-element error vs fp32 for several sizes and distributions (normalize output only).
- `mpsgraph_fp16_subnormals.swift` — whether fp16 subnormals survive cast, multiply, add, matmul and reductions.

Results are recorded in GitHub issue #11.
