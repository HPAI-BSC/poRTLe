# TuRTLe → CVDP in PoRTLe

This package turns TuRTLe benchmarks into CVDP-compatible datasets for PoRTLe. TuRTLe already wraps popular RTL benchmarks for LLMs, so we reuse its task definitions and evaluation logic instead of reinventing them.

## Prerequisite: clone TuRTLe
- Clone the upstream repo into the shared pool: `src/datasets/builders/dataset_repos/TuRTLe`
  - `git clone https://github.com/HPAI-BSC/TuRTLe src/datasets/builders/dataset_repos/TuRTLe`
- All benchmark assets (prompts, references, tests, metrics helpers) are read directly from that local copy.

## What lives where
- TuRTLe tasks and metrics: `src/datasets/builders/dataset_repos/TuRTLe/turtle/tasks/*` and `.../metrics/*`.
- CVDP adapters (per benchmark): `src/datasets/builders/turtle_cvdp/adapters/{benchmark}/` containing the harness script, `docker-compose.yml`, and `.env`.
- Dataset builder: `src/datasets/builders/turtle_cvdp/build_dataset.py` emits JSONL files into `benchmark_datasets/TuRTLe/`.

## Supported benchmarks today
- RTLLM (categories: `cid003`, all tasks)
- VerilogEval code-completion subset only (categories: `cid002`)
(Adding another TuRTLe benchmark follows the pattern below.)

## How the builder works
- `build_dataset.py` imports TuRTLe task builders (e.g., `rtllm.py`, `verilog_eval.py`) and walks their task data.
- For each task it constructs a CVDP entry with prompt/context, RTL/test files, and harness assets from `adapters/{benchmark}/`.
- RTLLM tasks are labeled `cid003`; VerilogEval uses only the code-complete ICCAD2023 tasks and labels them `cid002`.
- Run:
  - `python3 src/datasets/builders/turtle_cvdp/build_dataset.py --rtllm`
  - `python3 src/datasets/builders/turtle_cvdp/build_dataset.py --verilogeval`
  - Add `--test` to emit only the first task for quick checks.

## Adding another TuRTLe benchmark
1) Ensure TuRTLe contains the benchmark data under `.../TuRTLe/turtle/tasks/{BenchmarkName}/` and its Python builder (e.g., `.../tasks/{benchmark}.py`).
2) Create an adapter folder `adapters/{benchmark}/` with:
   - `{benchmark}_cvdp_harness.py` (evaluation entry point inside the container)
   - `docker-compose.yml` (selects image and runs the harness)
   - `.env` (even if empty)
   Use the existing RTLLM and VerilogEval adapters as templates.
3) Extend `build_dataset.py`:
   - Load the new TuRTLe task builder.
   - Collect prompt/spec/RTL/test assets from the TuRTLe task folders.
   - Attach adapter files into the `harness` section of each CVDP task.
   - Set IDs/categories for the new benchmark.
4) Generate and spot-check:
   - `python3 src/datasets/builders/turtle_cvdp/build_dataset.py --{benchmark} --test`
   - Inspect the emitted JSONL in `benchmark_datasets/TuRTLe/`.

## CVDP task shape (what agents see in the container)
```
task_name/
├── rtl/                    # RTL code
├── docs/                   # Specs/docs
├── verif/                  # Testbenches
├── src/                    # Harness scripts + .env
└── docker-compose.yml      # Container config to run the harness
```
Your builder should populate `context` (RTL/docs/tests) and `harness` (compose, harness script, .env) accordingly.

## Why route through TuRTLe?
- Consistent interface across benchmarks already packaged for RTL LLM evaluation.
- Reuses validated prompts, references, and metric code.
- As TuRTLe gains benchmarks, you can pull the updated repo and add a matching adapter/build step to expose them in PoRTLe via CVDP.
