# Benchmark Datasets

> **Note:** The contents of this directory (except this README and `cvdp_example/`) are **not tracked by git** (see `.gitignore`). Clone external benchmark datasets into this directory to use them with poRTLe.

## Overview

This directory contains benchmark datasets in JSONL format. Each benchmark should have its own subdirectory containing `.jsonl` files with task definitions.

## Directory Structure

```text
benchmark_datasets/
├── README.md (this file)
├── cvdp_example/ (tracked - example dataset)
├── CVDP/ (not tracked - clone externally)
├── TuRTLe/ (not tracked - clone externally)
└── [your-benchmark]/ (not tracked - clone externally)
```

## Available Datasets

Clone the following benchmark datasets into this directory for use with poRTLe:

### Dataset Template

```text
**Benchmark:** [Benchmark Name]
**Repository:** [GitHub/GitLab URL]
**Commit:** [commit-hash] (for compatibility)
**Description:** [Brief description]
**Setup:**
  cd benchmark_datasets/
  git clone [repo-url] [benchmark-name]
  cd [benchmark-name]
  git checkout [commit-hash]
  [additional setup if needed]
**Build Command:**
  python src/build_benchmark_json.py [benchmark-name]
```

---

### CVDP Benchmark

**Source:** [🤗 nvidia/cvdp-benchmark-dataset](https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset)

**Commit (local clone):** `25aa15d6375da07ba538aaef7e732c133539d253`

**Description:** Comprehensive Verilog Design Problems dataset that powers the CVDP benchmark. Includes agentic and non-agentic JSONL task suites with metadata used by the poRTLe CVDP adapter.

**Setup:**

```bash
cd benchmark_datasets/
git clone https://huggingface.co/datasets/nvidia/cvdp-benchmark-dataset cvdp
cd cvdp
git checkout 25aa15d6375da07ba538aaef7e732c133539d253
```

**Build Command:**

```bash
python src/build_benchmark_json.py cvdp
```

---

## Adding Your Own Dataset

1. Clone your dataset repository into this directory
2. Ensure datasets are in JSONL format (one task per line)
3. Build the benchmark JSON: `python src/build_benchmark_json.py <benchmark_name>`
4. Document it in this README with repository URL and compatible commit hash

## Dataset Format

Datasets should be JSONL files where each line is a JSON object representing a task. The exact structure depends on your benchmark adapter implementation.
