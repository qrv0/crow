# CAWSF-NDSQ: A Decomposed, Conditionally-Loaded Framework for Executing Large Language Models on Consumer Hardware

## Abstract

Deploying multi‑billion‑parameter Large Language Models (LLMs) on consumer hardware is constrained by a memory wall: even with aggressive post‑training quantization, conventional runtimes must load the entire model into memory. We present CAWSF‑NDSQ, a framework that decouples model parameter count from runtime memory by decomposing dense weights into semantically distinct components and loading only the shards predicted to be useful for the current context. The proposed NDSQ decomposition factorizes each weight matrix into a quasi‑diagonal core (D), a low‑rank global structure (L), a dense mid‑frequency residue (R), and sparse outliers (S). Each component is compressed with a component‑specialized codec, and stored in a custom .cawsf container engineered for mmap‑friendly on‑demand access. A semantic router ranks/selects shards via cosine similarity between prompt‑derived keys and per‑shard keys recorded in the file’s ROUTING section. Our implementation in Go includes a converter from Hugging Face .safetensors, integrity verification via rolling XXH3‑64 checksums, a reconstruction engine with optional CUDA/cuBLAS matvec acceleration, GGUF export for llama.cpp, and a runnable CLI. Empirically (as detailed in the project notes), CAWSF‑NDSQ enables running a 70B model within 8 GB VRAM by trading off latency for memory, while maintaining competitive quality versus strong GGUF baselines.

---

## 1. Introduction

State‑of‑the‑art LLMs have grown to tens or hundreds of billions of parameters, making them costly or infeasible to run on commodity hardware. Static compression methods (e.g., PTQ) reduce the precision of stored weights but retain the static‑loading paradigm: all weights (or all quantized blocks) must be resident during inference. Separately, conditional‑compute architectures (e.g., Mixture‑of‑Experts) activate only subsets of parameters per token, but require architectural changes during pretraining.

CAWSF‑NDSQ bridges these worlds with a third paradigm: post‑hoc conditional loading for dense pretrained models. We decompose weights into components with distinct information structure and compress each with a tailored codec. At inference time, a router chooses only the shards required by the current context, loading them on demand from a multi‑tier memory hierarchy (SSD → RAM → VRAM). This expands the classic quality‑vs‑memory trade‑off into a three‑way trade‑off among quality, memory, and latency.

Our contributions are:
- NDSQ weight decomposition into D/L/R/S with a practical factorization order and on‑the‑fly reconstruction.
- Component‑specialized compression: FP16 for D, low‑precision factors for L, Product Quantization (PQ) for R, and explicit sparse storage for S.
- CAWSF file format supporting mmapped, per‑section compression and a ROUTING index for semantic selection.
- A complete Go implementation with conversion, verification, routing, apply/reconstruction, and GGUF export/run via llama.cpp.

---

## 2. Background

Post‑Training Quantization (PTQ) is central to local LLM inference; block‑wise k‑quants (e.g., in GGUF) provide strong size/quality trade‑offs but still require static loading. Conditional compute (MoE) sparsifies activation per token via learnable routing, but alters model architecture and training. Recent systems work explores offloading across memory tiers and prediction of active weights to hide I/O. CAWSF‑NDSQ pushes this further with semantic offloading: predicting which compressed weight shards are semantically useful for the current prompt.

---

## 3. NDSQ Decomposition and Poly‑Codec Compression

### 3.1 Principle
Dense LLM weight matrices superpose heterogeneous structures: global correlations, near‑diagonal energy, mid‑frequency residue, and rare outliers. Treating the matrix uniformly is suboptimal for compression. NDSQ approximates W as:

W ≈ D + L + R + S

- D (quasi‑diagonal): captures principal energy near the diagonal; stabilizes one‑to‑one transformations.
- L (low‑rank): captures global correlations; implemented via truncated SVD with rank r.
- R (dense residue): mid‑frequency texture remaining after L/D/S are removed.
- S (sparse outliers): high‑magnitude critical values (e.g., top quantiles) with explicit coordinates.

Factorization order (implemented in the converter):
1) L via truncated SVD on W;
2) Residue W′ = W − L;
3) Identify outliers in W′ to form S; zero them to get W″;
4) Extract diagonal of W″ to form D; remaining dense part is R = W″ − D.

### 3.2 Component‑Specialized Codecs
- D: stored as FP16 (small, structurally important).
- L: stored as FP16 or INT8 factors (already compressed by rank‑r factorization).
- R: Product Quantization (PQ). The residue is partitioned into subvectors; per‑subvector K‑Means yields codebooks, and each subvector is replaced by its centroid index.
- S: explicit sparse triplets (row, col, value), with values stored as FP16/FP32 depending on configuration.

The Go implementation includes a simple but functional PQ trainer/encoder/decoder (internal/quant/pq.go), and shard‑level decoders for all components (internal/cawsf/reconstruct.go).

---

## 4. The CAWSF File Format (.cawsf)

The CAWSF container stores model metadata, shared codebooks, compressed shards, and routing keys in contiguous sections with 4 KiB alignment to optimize mmap I/O. Sections can be individually compressed (Zstandard or LZ4). The on‑disk layout and I/O are implemented in internal/fileformat/cawsf.go.

Sections:
- Header & TOC: magic "CAWSF", version, number of sections, and TOC records {type, offset, size, flags}.
- META (Type 1): JSON metadata including hyperparameters, codec settings, and integrity index (checksum_index) of rolling XXH3‑64 over other sections.
- CODEBOOKS (Type 2): shared PQ codebooks referenced by R shards; parsed to a CodebookPool.
- SHARD_BANK (Type 3): a bank of shard records (see below), storing all D/L/R/S payloads by scope.
- ROUTING (Type 4): semantic keys and costs per shard for router selection.

Section flags support per‑section compression:
- FlagCompZSTD (1<<0): ZSTD‑compressed payload.
- FlagCompLZ4  (1<<1): LZ4‑compressed payload.

### 4.1 Shard Bank
Each shard begins with a 12‑byte header:
- Type (uint8): 0=L, 1=R, 2=S, 3=D
- Scope (uint16)
- Comp (uint8): 0=raw, 1=zstd, 2=lz4 (for the shard payload)
- Usize (uint32): uncompressed payload size
- Csize (uint32): compressed payload size

Payloads by type:
- L, D: FP16 with shape prefix [rows:u32, cols:u32] followed by rows*cols half floats.
- R: PQ payloads with two supported encodings:
  - Embedded codebooks: rows:u32, cols:u32, d:u16, m:u16, k:u16, n:u32, cb:[m*k*(d/m)*f32], codes:[n*m*u8]
  - Shared codebooks:  rows:u32, cols:u32, d:u16, m:u16, k:u16, n:u32, cb_id:u16, codes:[n*m*u8]
- S: Sparse payload: rows:u32, cols:u32, n:u32, then n index pairs (row:u32, col:u32), then n values (f32).

The reconstruction engine (internal/cawsf/reconstruct.go) indexes the bank, decompresses shards as needed, and returns dense float32 weights for a given scope:
- ReconstructForScope(bank, scope)
- ReconstructForScopeWithPool(bank, pool, scope) — enabling shared codebook decoding.

### 4.2 Routing Section
The ROUTING section encodes per‑shard selection keys and costs as a compact binary block:
- dim:u16, n:u32
- shard_ids:[n]*u32
- costs:[n]*f32
- keys:[n][dim]*f32 (L2 normalized; re‑normalized on read for numeric stability)

The CLI command `crow route` reads this section, forms a prompt key, ranks shards by cosine similarity, and returns the top‑k within an optional budget.

---

## 5. Runtime and Router

At inference time, CAWSF maintains a three‑tier memory hierarchy:
- VRAM cache for active shards (if GPU present),
- RAM cache for warm shards,
- Backing .cawsf on SSD (mmap‑friendly layout).

The router:
- Derives a prompt key from a hashed bag‑of‑words projection (xxh3‑based) into the ROUTING key space.
- Computes cosine similarity against per‑shard keys and selects top‑k shards subject to an optional budget.
- Streams selected shards into the caches, overlapping I/O and compute in a future iteration.

Matrix application is performed per scope via addends D, L, R, and S applied to y = W x:
- D/L shards: dense matvec; optional cuBLAS path with build tag `cuda` (internal/gpu/cublas.go) using cublasSgemv.
- R shards: PQ decode and accumulation (CPU implementation in both CUDA and no‑CUDA builds).
- S shards: sparse add of triplets (CPU implementation in both builds).

---

## 6. Implementation and CLI

The project is written in Go 1.22. Key packages:
- internal/fileformat: CAWSF read/write, GGUF inspection/writer.
- internal/cawsf: shard codecs and reconstruction.
- internal/quant: Product Quantization (training/encode/decode).
- internal/gpu: cuBLAS integration (optional, build tag `cuda`) with CPU fallbacks.
- internal/runner: llama.cpp bindings (build tag `llama`).

CLI highlights:
- Convert: `crow convert --model x.safetensors --out x.cawsf [--rank r --outlier-q q --pq-m m --pq-k k ...]`
- Inspect/Verify: `crow inspect`, `crow verify` (per‑section rolling XXH3‑64 with hex hashes in META)
- Route/Apply: `crow route --in model.cawsf -p "prompt"` and `crow apply --in ... --scope N --xlen COLS`
- Export: `crow export` (dense f32 per scope), `crow export-gguf` (GGUF v3 minimal metadata)
- Run: `crow run model.gguf -p "prompt"` when built with `-tags llama`

Build modes:
- CPU‑only runtime: `go build -o crow ./cmd/crow`
- With llama.cpp: `go build -tags llama -o crow ./cmd/crow`
- With CUDA: `go build -tags cuda -o crow ./cmd/crow` and set `CROW_CUDA=1` at runtime

---

## 7. Experimental Perspective

Project documentation reports feasibility results on large models, including running a Llama‑3 70B‑class model within an 8 GB VRAM budget using CAWSF‑NDSQ, where a strong GGUF baseline fails with OOM in the same constraint. Quality remains competitive with GGUF on unconstrained hardware, with a predictable throughput reduction due to dynamic shard loading and decompression overheads. Ablations suggest S (outliers) contribute the largest single quality jump beyond D/L, and performance scales with VRAM cache size. These observations align with the intended trade‑off: reclaim VRAM headroom by paying an I/O‑driven latency cost.

Note: Exact metrics and configurations can vary by hardware, storage bandwidth, and router parameters. Refer to the repository README and notes for current benchmarks and limitations.

---

## 8. Limitations and Future Work

- Router quality: Current hashed bag‑of‑words keys are heuristic. Learnable, context‑aware routing (e.g., small auxiliary networks) may improve accuracy and prefetch timing.
- I/O bottlenecks: SSD bandwidth and latency bound throughput under tight VRAM budgets. Memory‑mapped I/O and smarter prefetch/eviction policies can mitigate this.
- Compute kernels: Fused GPU kernels for PQ decode + accumulation and sparse application could reduce overhead.
- GGUF export: Minimal metadata is emitted; architecture‑specific KV pairs and canonical tensor naming are planned for direct llama.cpp load/run without manual mapping.
- Converter performance: SVD and K‑Means are CPU‑bound. Multi‑threading and optional FAISS bindings could accelerate PQ training.

---

## 9. Related Work

- PTQ for LLMs (8‑bit/4‑bit), mixed‑precision outlier handling, and GGUF k‑quants underpin modern local inference.
- Conditional compute (MoE) uses learnable routing but requires architectural changes during training.
- Systems for offloaded inference (active‑weight prediction, SSD/DRAM/VRAM scheduling) motivate CAWSF’s semantic offloading and shard‑level I/O design.

---

## 10. Conclusion

CAWSF‑NDSQ reframes local LLM inference by decomposing dense weights and loading only context‑relevant shards, separating parameter count from runtime memory. With a custom on‑disk format, component‑aware codecs, and a simple semantic router, the framework makes massive models feasible on commodity GPUs by trading VRAM for I/O‑bound latency. The open‑source Go implementation offers a practical starting point with conversion, verification, export/run via llama.cpp, and optional CUDA matvec acceleration. Continued work on routing, caching, and fused kernels will further close the performance gap while preserving the new flexibility this approach introduces.

---

### Reproducibility and Availability

Source code (Go 1.22) with CLI and Dockerfile is available in this repository. Example quick‑start:

```
go build -o crow ./cmd/crow
./crow convert --model toy.safetensors --out toy.cawsf --rank 2 --outlier-q 0.999 --pq-m 4 --pq-k 8 --max-layers 1
./crow verify --in toy.cawsf
./crow export --in toy.cawsf --out out_dir
./crow export-gguf --in toy.cawsf --out toy.gguf
```

Please see README.md for full instructions, build tags (`llama`, `cuda`), and current limitations.

---

### Acknowledgments and Notes

- Integrity verification uses rolling XXH3‑64 per section; hex hashes are stored in META (checksum_index), verified by `crow verify`.
- CUDA/cuBLAS paths accelerate D/L matvecs when available; R and S paths are CPU in this version.
- ROUTING keys and format are explicitly defined and parsed by the CLI, enabling repeatable routing experiments.
