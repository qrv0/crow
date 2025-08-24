# Crow

> CAWSF-NDSQ runtime and utilities for LLM weights in Go — with GGUF export/run via llama.cpp

<p align="center">
  <img src="https://img.shields.io/badge/status-alpha-orange" alt="status: alpha" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License" /></a>
</p>

Crow is a plug-and-play (Ollama-like) system to convert, inspect, and run LLM weights in the **CAWSF-NDSQ** format. It can also export **GGUF** and run GGUF models via **llama.cpp** (optional build tag).

* **Sharded, on-disk runtime**: weights are factorized into **D/L/R/S**, compressed with tailored codecs, and stored as shards on disk.
* **Router**: loads only the shards needed for the current prompt (contrast: classic quantization keeps the entire model resident).

> **Status**: **ALPHA** — APIs, file layout, ergonomics and performance are stabilizing. Expect breaking changes.

## Table of Contents

* [What is Crow?](#what-is-crow)
* [Key Features](#key-features)
* [Install](#install)
* [Quickstart (toy)](#quickstart-toy)
* [CLI — Commands](#cli--commands)
* [Build/Run](#buildrun)
* [Limitations (alpha)](#limitations-alpha)
* [Notes & tips](#notes--tips)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)

## What is Crow?

Crow is a runtime for large model weights using a storage/IO-efficient format (**CAWSF-NDSQ**). Instead of loading all weights into memory, it splits weights into shards and loads only what’s needed to answer a prompt.

It can also:

* Convert Hugging Face `.safetensors` → `.cawsf` (CAWSF-NDSQ);
* Reconstruct dense `float32` weights for export;
* Export **GGUF** (for the llama.cpp ecosystem) and run GGUF via llama.cpp when built with the `llama` tag.

## Key Features

* **CAWSF-NDSQ container**

  * Header + TOC with 4KiB alignment
  * Sections: **META** (JSON), **CODEBOOKS** (shared PQ), **SHARD\_BANK** (D/L/R/S shards), **ROUTING** (keys/costs per shard)
  * Optional per-section compression: zstd/lz4
* **Converter**

  * NDSQ decomposition: **L** via truncated SVD, **D** from residual diagonal, **S** from outliers (quantile), **R** by Product Quantization (k-means)
  * **R** shards reference shared codebooks in **CODEBOOKS**
* **Integrity**

  * Rolling **XXH3-64** checksums per section (1 MiB chunks), recorded in META; `crow verify` recomputes and checks
* **Routing**

  * Cosine-similarity selection using a deterministic hashed bag-of-words derived from the prompt; optional budget constraint
* **Apply/Export**

  * On-the-fly `y = W*x` per scope (no full materialization), or reconstruct and export dense f32 tensors
* **GGUF export**

  * Writes valid GGUF v3 with minimal metadata and reconstructed f32 tensors

## Install

> Requires **Go 1.22+** and a working CGO setup if using llama.cpp.

**From source**

```bash
git clone https://github.com/qrv0/crow.git
cd crow
go build -o bin/crow ./cmd/crow
go build -o bin/make_safetensors ./cmd/make_safetensors
# then
./bin/crow -h
```

## Quickstart (toy)

Generate a small “toy” weight set and run the end-to-end flow.

1. Build binaries

```bash
go build -o crow ./cmd/crow
go build -o make_safetensors ./cmd/make_safetensors
```

2. Create toy weights

```bash
./make_safetensors -out toy.safetensors -rows 8 -cols 16
```

3. Convert to CAWSF

```bash
./crow convert --model toy.safetensors --out toy.cawsf \
  --rank 2 --outlier-q 0.999 --pq-m 4 --pq-k 8 --max-layers 1
```

4. Verify checksums

```bash
./crow verify --in toy.cawsf
```

5. Export dense f32 tensors

```bash
./crow export --in toy.cawsf --out out_dir
```

6. Export GGUF

```bash
./crow export-gguf --in toy.cawsf --out toy.gguf
```

## CLI — Commands

```text
crow init                                   # initialize ~/.crow
crow pull <url>                             # download .gguf or .cawsf to ~/.crow/models
crow list                                   # list installed models
crow inspect <file.cawsf|.gguf>             # inspect a CAWSF/GGUF file
crow verify --in <file.cawsf>               # verify per-section checksums
crow route --in <file.cawsf> -p "prompt" [--k 8] [--budget X]
                                            # rank/select shards by cosine similarity
crow apply --in <file.cawsf> --scope N --xlen COLS
                                            # compute y = W*x for a given scope
crow export --in <file.cawsf> --out <dir>
                                            # reconstruct and export f32 blobs per scope
crow export-gguf --in <file.cawsf> --out <file.gguf>
                                            # export GGUF with f32 tensors
crow convert --model <file.safetensors> --out <file.cawsf>
  [--rank 64] [--outlier-q 0.999] [--pq-m 8] [--pq-k 256]
  [--max-layers 0] [--max-elems 0]          # convert Hugging Face to CAWSF-NDSQ
crow run <file.gguf> -p "prompt" [--ctx 4096] [--gpu-layers N]
  [--temperature 0.8] [--top-k 50] [--top-p 0.95] [--repeat-penalty 1.1]
                                            # generate text with llama.cpp (build tag `llama`)
```

## Build/Run

* **CPU-only (no llama.cpp)**

  ```bash
  go build -o crow ./cmd/crow
  # `run` is disabled unless built with the `llama` tag
  ```
* **With llama.cpp**

  ```bash
  go build -tags llama -o crow ./cmd/crow
  # requires CGO and github.com/go-skynet/go-llama.cpp
  ```
* **With CUDA (CAWSF L/D matvec)**

  ```bash
  go build -tags cuda -o crow ./cmd/crow
  # requires CUDA toolkit + cuBLAS
  # at runtime:
  export CROW_CUDA=1
  ```

  Tags can be combined, e.g. `-tags "llama,cuda"`.

## Limitations (alpha)

* **GGUF export**: minimal metadata & generic tensor names. For llama.cpp to load directly, architecture-specific KV fields and canonical names are needed (planned).
* **Converter**: CPU-bound (SVD + k-means). For large models, use `--max-layers`/`--max-elems` to slice, or run on a bigger machine.
* **Routing**: simple hashed BoW embedding; no trained semantic router in this version.
* **GPU**: CUDA/cuBLAS path currently covers **L/D** shard matvecs; **R/S** remain on CPU.

## Notes & tips

* GGUF export writes minimal metadata from META. Mapping to specific architectures (e.g., LLaMA/Mistral) may require additional KV and canonical tensor naming (see roadmap).
* Large model files are intentionally excluded from the repo. Use `crow pull` for model acquisition.

## Roadmap

* Architecture-specific GGUF metadata & canonical tensor naming for direct llama.cpp compatibility.
* Faster/better PQ (multi-threading; optional FAISS bindings).
* Optional GPU decode paths and on-demand shard caching.
* Pluggable semantic router (trainable embeddings) and budget-aware scheduling.
* More unit tests/benchmarks; memory-mapped I/O.

## Contributing

PRs and issues are very welcome in this ALPHA phase. See also `SECURITY.md` and `CODE_OF_CONDUCT.md`.

## License

Apache License 2.0 — see `LICENSE`.
