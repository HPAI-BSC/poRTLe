# Benchmark Runners

> **Note:** The contents of this directory are **not tracked by git** (see `.gitignore`). Clone external benchmark runner projects into this directory to use them with poRTLe.

## Overview

### These instructions highlight how to add CVDP as your benchmark runner and create a new benchmark runner if desired.

We separate benchmarks from benchmark runners. Currently poRTLe supports one benchmark runner (CVDP) as it is gives us the flexibility to run any agent or LLM on a given task within a Docker container. When integrating new benchmarks (VerilogEval, RTLLM, etc.) into poRTLe we use CVDP as our benchmark runner, meaning we use an adapter (src/datasets/builders) to construct a dataset of all tasks and eval harnesses from the benchmark in CVDP's runner expected format. With this distinciton between benchmarks and benchmark runners, we can easily incorporate new features added to CVDP to ANY benchmark that uses CVDP as its runner.

If a new benchmark is released that has a more capable or well-fit runner than CVDP that you wish to use (ex. Running agents in Singularity containers instead of Docker), then you must create a new benchmark runner and follow these instructions.

## Directory Structure

```text
benchmark_runners/
├── README.md (this file)
├── cvdp_benchmark/ (not tracked - clone externally)
└── [your-benchmark]_benchmark/ (not tracked - clone externally)
```

## Available Runners

Clone the following benchmark runners into this directory for use with poRTLe:

### Runner Template

```text
**Benchmark:** [Benchmark Name]
**Repository:** [GitHub/GitLab URL]
**Commit:** [commit-hash] (for compatibility)
**Description:** [Brief description]
**Setup:**
  cd benchmark_runners/
  git clone [repo-url] [benchmark-name]_benchmark
  cd [benchmark-name]_benchmark
  git checkout [commit-hash]
  [additional setup like building Docker images, installing deps, etc.]
**Adapter:** [adapter-name] (in src/benchmarks/)
```

---

### CVDP Benchmark Runner (Example)

**Repository:** <https://github.com/dakotazoid56/cvdp_benchmark.git>

**Commit:** `84e03dd33`

**Description:** Runner for CVDP benchmark tasks with Docker-based execution forked from NVIDIA CVDP

**Setup:**

```bash
cd benchmark_runners/
git clone https://github.com/dakotazoid56/cvdp_benchmark.git cvdp_benchmark
cd cvdp_benchmark
git checkout 84e03dd33
```

**Adapter:** `cvdp` (built-in at `src/benchmarks/cvdp_adapter.py`)

---

## Adding Your Own Runner

1. Clone your benchmark runner repository into this directory
2. Name it `[benchmark-name]_benchmark` to match the adapter's `get_benchmark_runner_dir()` method
3. Create a corresponding adapter in `src/benchmarks/` (see `cvdp_adapter.py` for reference)
4. Register the adapter in `src/benchmarks/__init__.py`
5. Document it in this README with repository URL and compatible commit hash

## Runner Requirements

Each benchmark runner should:

- Accept task IDs and dataset paths as inputs
- Execute tasks (e.g., via Docker, local execution, etc.)
- Generate results in a format parseable by the corresponding adapter
- Output logs and metrics (execution time, token count, errors, etc.)
