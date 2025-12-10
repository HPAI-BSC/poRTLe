# Adding a New Benchmark to PoRTLe

The recommended runner for new benchmarks is CVDP. This guide explains how to package any RTL benchmark so PoRTLe can serve it via CVDP.

## 1) Create a benchmark builder folder
- Add `src/datasets/builders/{benchmark}_cvdp/`.
- Use `src/datasets/builders/turtle_cvdp/` as a concrete reference.
- Inside this folder you will add the dataset builder script and harness assets.
- Clone any source benchmark repos into the shared pool at `src/datasets/builders/dataset_repos/` (e.g., `TuRTLe` lives here).

## 2) Author `build_dataset.py`
- Purpose: turn each source benchmark task into a CVDP task JSONL entry.
- Responsibilities:
  - Load the benchmark’s data (RTL, specs/docs, verification testbenches).
  - Build the CVDP task object: `prompt`, `context`, `patch`, `harness`, `categories`, `id`.
  - Write a JSONL file to `benchmark_datasets/{Benchmark}/{name}.jsonl`.
- You are free to read source data however you like: local clone (e.g., TuRTLe), Hugging Face dataset, etc.

## 3) Provide harness assets (required)
Each benchmark needs its own harness and container config. Create an `adapters/{benchmark}/` subfolder (see `turtle_cvdp/adapters/rtllm` and `turtle_cvdp/adapters/verilogeval` for examples) containing:
- `*_cvdp_harness.py`: runs the benchmark’s evaluation inside the container.
- `docker-compose.yml`: defines the image and command (usually `python3 /src/<harness>.py`).
- `.env`: passed into the container (can be empty but must exist).

## 4) Match the CVDP task layout
Agents see tasks in the container with this shape:
```
task_name/
├── rtl/                    # RTL code (SystemVerilog/Verilog)
│   └── module.sv
├── docs/                   # Documentation and specifications
│   └── specification.md
├── verif/                  # Verification files (testbenches)
│   └── module_tb.sv
├── src/                    # Test harness (Python scripts, etc.)
│   ├── test_runner.py
│   └── .env
└── docker-compose.yml      # Docker configuration for testing
```
Your `build_dataset.py` should populate `harness` entries (docker compose, harness script, `.env`) and any `context` files (RTL, tests, specs) so the agent sees this structure.

## 5) Source data guidance
- TuRTLe example: the repo is cloned into `src/datasets/builders/dataset_repos/TuRTLe`. Builders read directly from that clone to emit CVDP tasks.
- Hugging Face or other hosting: clone/download the dataset into `dataset_repos/` (or another predictable local path) and read from there.
- Reuse or reconstruct the original benchmark’s evaluation harness; keep behavior identical where possible.

## 6) Minimal checklist
- [ ] Folder `src/datasets/builders/{benchmark}_cvdp/` created.
- [ ] Source repo cloned into `src/datasets/builders/dataset_repos/{RepoName}` (if applicable).
- [ ] `build_dataset.py` generates a JSONL in `benchmark_datasets/{Benchmark}/`.
- [ ] Harness under `src/datasets/builders/{benchmark}_cvdp/adapters/{benchmark}/` with `*_cvdp_harness.py`, `docker-compose.yml`, `.env`.
- [ ] Tasks include RTL, specs/docs, tests, and harness files in the expected CVDP layout.
- [ ] Running `python3 src/datasets/builders/{benchmark}_cvdp/build_dataset.py --test` succeeds.

For concrete examples, browse `src/datasets/builders/turtle_cvdp/` and its `adapters/` subfolders.
