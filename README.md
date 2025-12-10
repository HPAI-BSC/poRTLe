# poRTLe

**poRTLe** is a platform for running, tracking, and analyzing AI agent performance across RTL benchmarks, datasets, and tasks.

## Overview

poRTLe provides a unified interface for:

- Running AI agents against multiple benchmarks (CVDP, TuRTLe, DIY etc.)
- Creating custom DIY tasks from your own RTL code
- Tracking results in a structured database with rich metadata
- Visualizing performance with an interactive UI
- Managing benchmarks with persistent background execution
- Filtering and analyzing tasks with advanced query capabilities

## Quick Start

### 1. Clone External Dependencies

Clone the external projects that live in gitignored directories (only the runner is required for the quick start):

- **`benchmark_runners/`** – *Required for Quick Start.* Follow [benchmark_runners/README.md](benchmark_runners/README.md) to clone the CVDP runner (`cvdp_benchmark`) **and** set up its `cvdp_env` virtual environment.
- **`benchmark_datasets/`** – Benchmark task datasets (optional extras for Quick Start; see [README](benchmark_datasets/README.md)).
- **`agents/`** – Agent implementations (optional; see [README](agents/README.md)).
- **`results/`** – Shared results repository (optional for Quick Start; see [README](results/README.md)).

**Note:** `results/` is its **own git repository** so teams can share benchmark data independently of the main codebase.

### 2. Create Python Environment

Use a dedicated virtual environment so poRTLe’s dependencies stay isolated:

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .\.venv\Scripts\activate        # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

Reactivate `.venv` in every new shell and run `deactivate` when you are finished.

### 3. Set Up Results Directory

Copy the example results to initialize the results directory:

```bash
cp -r results_example/. results/
```

### 4. Launch the UI Shell

Start Streamlit to drive most workflows through the graphical interface:

```bash
streamlit run src/ui/portle_ui.py
```

Open `http://localhost:8501` (the app may take a few seconds to load on first launch).

> **Tip:** Most operations can be performed directly inside the UI instead of running Python scripts manually.


### 5. Run Benchmarks and View Results

**Note:** You must have your Docker daemon running in the background for this to work.

#### Run the Example Agent

Start with the example agent to verify your setup:

- **UI path:** In **Commands ▸ Run Benchmark**, select the `cvdp_example` benchmark and the example agent, configure options (background execution, task filters, execution mode), then click **Run Benchmark**. Monitor progress in the **Process Monitor** tab. Afterwards, explore results in the **Plots** tab (double-click cells for drill-down) or the **Search** tab for database queries.
- **CLI alternative:** Configure and execute the dataset runner:

  ```bash
  cp src/run_example.yaml src/run.yaml        # configure once
  python src/run_dataset.py
  ```

#### Run the OpenCode Agent

Once you've verified the example agent works, try the OpenCode agent:

1. First, follow the setup instructions in [agents/opencode-agent/README.md](agents/opencode-agent/README.md) to configure your API key.
2. Then run benchmarks using the `opencode` agent in the UI or CLI.

#### Run LLM with Force-Copilot

To run benchmarks using direct LLM calls (no Docker agent):

1. Set your OpenAI API key in `benchmark_runners/cvdp_benchmark/.env`:

   ```bash
   OPENAI_USER_KEY=sk-your-openai-api-key-here
   ```

2. In **Commands ▸ Run Benchmark**, select the `llm_gpt-5-mini` agent (pre-configured).
3. Set **Execution Mode** to **LLM Mode (force-copilot)**.
4. Run the benchmark.

To use other LLMs, add models to the model factory in the benchmark runner. Ensure you set the `cvdp_llm_name` when you add a new agent that matches the cvdp model factory name.

#### View Results

After a successful run:

1. Check `results/tmp/` for raw run output and logs.
2. Go to **Commands ▸ Build Database** to load your results into the SQLite database.
3. View results in the **Plots** tab (heatmaps, drill-down) or **Search** tab (database queries).

## Project Structure

```text
poRTLe/
├── agents/                    # AI agent implementations (not tracked)
├── benchmark_datasets/        # Benchmark task datasets (not tracked)
├── benchmark_runners/         # Benchmark execution scripts (not tracked)
├── results/                   # Generated results and database (not tracked)
│   ├── json/                  # JSON results by benchmark/dataset/run
│   ├── tmp/                   # Temporary run data
│   └── poRTLe.db             # SQLite database
├── src/
│   ├── benchmarks/           # Benchmark adapter system (NEW!)
│   │   ├── base.py          # Abstract adapter interface
│   │   ├── registry.py      # Adapter registry
│   │   └── cvdp_adapter.py  # CVDP implementation
│   ├── ui/                   # Streamlit UI components
│   ├── build_benchmark_json.py    # Build benchmark JSON from datasets
│   ├── build_datatable.py         # Build SQLite database from results
│   ├── run_dataset.py             # Run agents against benchmarks
│   ├── convert_run_results.py     # Convert raw results to JSON
│   └── run_example.yaml           # Example run configuration
└── README.md
```

## Architecture: Benchmark Adapters

poRTLe uses a **pluggable adapter system** to support multiple benchmarks. Each benchmark has its own adapter that handles:

- Dataset parsing (JSONL → JSON)
- Task execution (Docker, local, etc.)
- Result parsing (logs, metrics, token counts)
- Metadata extraction (benchmark-specific fields)

**Current Adapters:**

- `cvdp` - CVDP benchmark ([src/benchmarks/cvdp_adapter.py](src/benchmarks/cvdp_adapter.py))

**Adding a new benchmark:** See [src/benchmarks/base.py](src/benchmarks/base.py) for the adapter interface.

## Data

```text
Benchmarks (CVDP, TuRTLe, etc.)
  ↓ contains
Datasets (Commercial/Non-Commercial, Agentic/Non-Agentic)
  ↓ contains
Tasks (Individual problems)
  ↓ tested by
Agents (AI models/systems)
  ↓ produces
Runs (Benchmark execution instances)
  ↓ generates
Datapoints (Results: score, time, tokens, errors)
```

## Key Commands

All commands can be executed via the **UI (Commands tab)** or **CLI**. The UI provides additional features like metadata editing, task filtering, and background execution.

### Build Benchmark JSON

```bash
python src/build_benchmark_json.py <benchmark_name>
```

**UI:** **Commands ▸ Build Benchmark** tab. Add optional metadata (keys, notes, custom fields).

Converts JSONL datasets in `benchmark_datasets/<benchmark_name>/` to structured JSON.

### Build Database

```bash
python src/build_datatable.py
```

**UI:** **Commands ▸ Build Database** tab.

Builds SQLite database from all JSON results in `results/json/`.

### Add Agent

```bash
python src/add_agent.py --agent-id <id> --about <description> --backend-model <model> --agent-folder <path>
```

**UI:** **Commands ▸ Add Agent** tab. Add optional metadata and custom configuration.

Registers a new agent in the agents registry.

### Run Benchmark

```bash
python src/run_dataset.py
```

**UI:** **Commands ▸ Run Benchmark** tab. Configure:
- Task selection (manual or advanced filters)
- Execution mode (Agent or LLM/force-copilot)
- Background execution (persistent processes)
- Metadata (keys, notes, custom fields)

Runs agents against benchmarks using configuration in `src/run.yaml`.

### Create DIY Task

```bash
python src/create_diy_task.py --task-dir <path> --prompt <description> --benchmark-id <benchmark> --dataset-id <dataset>
```

**UI:** **Commands ▸ Create DIY Task** tab. Load from `input.jsonl` or configure manually.

Creates a custom task from your own RTL code and test infrastructure.

### Add Existing Run

```bash
python src/convert_run_results.py --run-dir <path> --benchmark-id <benchmark> --dataset-id <dataset> --agent-id <agent> --hardware-info <hardware>
```

**UI:** **Commands ▸ Add Existing Run** tab.

Converts an existing run directory into a finalized JSON entry.

### Monitor Processes

**UI:** **Process Monitor** tab.

View status, logs, and control background benchmark executions.

### Launch UI

```bash
streamlit run src/ui/portle_ui.py
```

Interactive web interface for browsing, analyzing, and managing results.

## UI Features

The Streamlit UI provides several powerful features for managing benchmarks:

### Pages

- **🔍 Search** - Find and view any entry (agents, benchmarks, datasets, tasks, runs, datapoints)
- **📊 Plots** - Interactive heatmaps and visualizations with drill-down capabilities
- **⚙️ Commands** - Execute all Python scripts from the UI with rich configuration options
- **🔄 Process Monitor** - View status and logs for background benchmark executions
- **ℹ️ About** - System overview and database statistics

### Metadata System

Add rich metadata to any entity (agents, benchmarks, runs, tasks):

- **Keys** - Categorize entries with tags (e.g., `production`, `experimental`, `verified`)
- **Notes** - Add timestamped notes with author attribution
- **Custom Fields** - Store arbitrary key-value pairs for additional metadata

### Background Execution

Run benchmarks as persistent background processes:

- Benchmarks continue running even if you close the UI
- Monitor multiple benchmarks simultaneously
- View live logs and status updates
- Automatic cleanup of orphaned processes

### Advanced Task Filtering

Filter tasks by metadata for targeted benchmark runs:

- Filter by difficulty, categories, or custom fields
- Combine multiple filters with AND/OR logic
- Save filter configurations for reuse
- Quick selection for manual task picking

### LLM Mode (force-copilot)

For CVDP benchmarks, run in LLM-only mode without Docker agents:

- Configure `cvdp_llm_name` in agent custom config
- Run benchmarks with direct LLM integration
- No Docker agent required
- Useful for testing and rapid iteration

### DIY Task Creation

Create custom tasks from your own RTL code:

- Organize code in `rtl/`, `docs/`, `verif/`, `src/` directories
- Load configuration from `input.jsonl` templates
- Automatically converts to JSONL format
- Integrates seamlessly with existing benchmarks

## Configuration

Edit `src/run.yaml` (copy from `src/run_example.yaml`):

```yaml
benchmark_id: CVDP
dataset_id: CVDP__dataset_name
agent_id: my-agent
hardware_info: local-machine
n: 3                    # Number of runs per task
threads: 1              # Parallel execution threads
adapter: cvdp           # Optional: specify adapter (defaults to benchmark_id.lower())
task_ids:               # Optional: specific tasks (omit for all)
  - CVDP__dataset__task_id_001
metadata:               # Optional: add metadata to the run
  keys:
    - experiment-1
    - production
  notes:
    - date_added: "01-15-25"
      author: "User"
      text: "Testing new configuration"
  custom:
    force_copilot: true  # For LLM mode
    difficulty: hard
```

## Git Ignore

The following are **not tracked** by git (see `.gitignore`):

- `agents/*` - Clone agents from external repositories
- `benchmark_datasets/*` (except `cvdp_example/` and README)
- `benchmark_runners/*` - Clone runners from external repositories
- `results/*` - Generated locally from runs
- `src/run.yaml` - Per-user configuration file
- `*.db` - SQLite databases

See each directory's README for compatible repositories and commit hashes.

## Contributing

To add support for a new benchmark:

1. Implement a `BenchmarkAdapter` in `src/benchmarks/<benchmark>_adapter.py`
2. Register it in `src/benchmarks/__init__.py`
3. Clone the benchmark's dataset into `benchmark_datasets/`
4. Clone the benchmark's runner into `benchmark_runners/`
5. Document repositories and commits in respective READMEs

## License

[Your License Here]
