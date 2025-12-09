# DIY Tasks Examples

This directory is a staging ground for DIY (Do-It-Yourself) tasks that can be turned into poRTLe benchmarks. Use it to prototype RTL challenges, keep their specs/testbenches organized, and publish them to the dataset JSONL files in `benchmark_datasets/`.

## Quickstart Workflow

From the repository root:

1. **Copy the template**

   ```bash
   cp -r diy_tasks/template diy_tasks/my_custom_task
   ```

2. **Customize the copy**
   - Replace `rtl/` with your RTL code
   - (Optional) Update `docs/` with documentation and `verif/` with testbenches to give the agent
   - (Optional) Edit `src/` and `src/test_runner.py` to encode your CocoTB tests (not given to the agent)
   - Adjust `input.jsonl` (id, prompt, categories) for the agent and metadata

3. **Register the task**
   - **UI flow**:
   `streamlit run src/ui/portle_ui.py` → **Commands → Create DIY Task** → point at `diy_tasks/my_custom_task` → click **📂 Load from input.jsonl** → **Create Task**
   - **CLI flow**:

     ```bash
     python src/create_diy_task.py \
         --task-dir diy_tasks/my_custom_task \
         --benchmark-id cvdp_custom \
         --dataset-id diy_tasks
     ```

4. **Build and index the benchmark**
   - **UI flow**:
   Handled in the UI when **Automatically build benchmark after creating tasks** is checked
   - **CLI flow**:
   ```bash
   python src/build_benchmark_json.py cvdp_custom
   python src/build_datatable.py
   ```
   `build_datatable.py` refreshes the SQLite cache that powers analytics and the UI filters.

5. **Run agents**
   - **UI flow**: Navigate to **Commands → Run Benchmark** → select the benchmark `my_custom_task`, select the agent, and → click **Run Benchamrk
   - **CLI flow**:
   - Edit `src/run.yaml`, then execute `python src/run_dataset.py` to benchmark the new task

## Template Layout & File Classification

Each task directory should resemble the template:

```
task_name/
├── rtl/                    # RTL code (SystemVerilog/Verilog)
│   └── module.sv
├── docs/                   # Documentation and specifications
│   └── specification.md
├── verif/                  # Verification files (testbenches)
│   └── module_tb.sv
├── src/                    # Test harness (Python scripts, etc.)
│   └── test_runner.py
│   └── .env
└── docker-compose.yml      # Docker configuration for testing
```

### Docker Image Placeholders

In `docker-compose.yml`, use template placeholders that get substituted at runtime:

| Placeholder | Default Image | Use Case |
|-------------|---------------|----------|
| `__OSS_SIM_IMAGE__` | `ghcr.io/hdl/sim/osvb` | Simulation tasks (iverilog, verilator) |
| `__PORTLE_OSS_CAD_IMAGE__` | `portle-oss-cad-suite:latest` | Full toolchain (iverilog + yosys + nextpnr) |

**Example for simulation-only task:**
```yaml
services:
  direct:
    image: __OSS_SIM_IMAGE__
    # ...
```

**Example for synthesis/PnR task:**
```yaml
services:
  direct:
    image: __PORTLE_OSS_CAD_IMAGE__
    # ...
```

> **Note:** To use `__PORTLE_OSS_CAD_IMAGE__`, first build the central images: `cd docker && ./build.sh`

When `src/create_diy_task.py` ingests the directory it classifies files automatically:

| Classification | What goes inside |
| -------------- | ---------------- |
| **context**    | RTL (`rtl/`), docs, verification, and every file that is not part of the harness |
| **prompt**     | The `input.jsonl` file contains the prompt to give the model, and task data|
| **harness**    | `src/`, `.env`, `docker-compose.yml`, and any other runtime/test infrastructure (CocoTB)|
| **patch**      | For the golden solution. Not yet implemlented |


## Template Reference & Customization

The maintained template (`diy_tasks/template`) contains:

- Placeholder RTL (`rtl/my_module.sv`)
- Specification stub (`docs/specification.md`)
- Testbench and harness (`verif/my_module_tb.sv`, `src/test_runner.py`, `src/.env`)
- Docker compose file
- `input.jsonl` pre-filled with  defaults

Customize the copy by editing whichever folders you need, keeping `input.jsonl` in sync with the prompt and metadata you want published. The template harness always passes, so you can verify the pipeline before introducing real logic or CocoTB-based tests.

## Registering a DIY Task

### UI Flow

1. Launch `streamlit run src/ui/portle_ui.py`
2. Navigate to **Commands → Create DIY Task**
3. Select a benchmark/dataset (or create new ones), paste the path to your task directory, and click **📂 Load from input.jsonl** to populate fields
4. Adjust prompt, categories, or metadata if needed
5. Click **Create Task** to append to `benchmark_datasets/<benchmark>/<dataset>.jsonl`

### CLI Flow

**Option A: With `input.jsonl` (recommended)**

```bash
python src/create_diy_task.py \
    --task-dir path/to/task \
    --benchmark-id your_benchmark \
    --dataset-id your_dataset
```

**Option B: Fully manual overrides**

```bash
python src/create_diy_task.py \
    --task-dir path/to/task \
    --prompt "Describe what the agent should do" \
    --benchmark-id your_benchmark \
    --dataset-id your_dataset \
    --task-id my_task_001 \
    --categories cid004,easy
```

CLI flags win over values inside `input.jsonl`, so you can selectively override the prompt, categories, or system message for experimentation.

## After Registration

1. **Build the benchmark JSON** so downstream tooling can load the dataset:

   ```bash
   python src/build_benchmark_json.py your_benchmark
   ```

2. **Rebuild the analytics/database cache** (always do this before running reports or loading the UI):

   ```bash
   python src/build_datatable.py
   ```

3. **Run the dataset with an agent** (optional check):

   ```bash
   # Update src/run.yaml first
   python src/run_dataset.py
   ```

4. **Refresh the Streamlit UI** if it is already open so it reads the new JSON and SQLite artifacts.

## Example: Simple Counter (`diy_tasks/simple_counter`)

- 8-bit counter with enable, reset, and rollover checks
- Includes RTL, spec, SystemVerilog testbench, and Python harness

Create or update its dataset entry:

```bash
python src/create_diy_task.py \
    --task-dir diy_tasks/simple_counter \
    --prompt "Verify the counter module meets the specification. The counter should correctly handle enable, reset, and rollover behavior." \
    --benchmark-id my_benchmarks \
    --dataset-id verification_tasks \
    --task-id simple_counter_001 \
    --categories verification,counter,easy
```

Then follow the “After Registration” steps to rebuild the benchmark JSON, regenerate the database, and refresh the UI.

## DIY Task Types

- **Verification tasks** — Provide RTL in context and ask the agent to confirm behavior or add tests
- **Debug/Fix tasks** — Deliver buggy RTL; the agent patches it
- **Design tasks** — Provide specs and expect fresh RTL. Golden reference checking is not automated yet, but you can still score solutions using your harness

## `input.jsonl` Configuration

Include an `input.jsonl` in your task directory to store metadata:

```json
{
  "id": "cvdp_custom_task_0001",
  "categories": ["cid004", "easy"],
  "system_message": "System guidance for the agent...",
  "prompt": "Describe the task for the agent..."
}
```

Only `--benchmark-id` and `--dataset-id` are mandatory CLI flags when this file exists. Command-line arguments override the JSONL values if you need to experiment.

## CVDP Task Format Requirements

- Task IDs must end in digits (e.g., `cvdp_custom_task_0001`)
- Categories should start with the CID (`cid004`) followed by the difficulty (`easy`, `medium`, `hard`)
- The UI auto-populates these fields using CVDP-compliant defaults; adjust them carefully if you deviate

Misformatted IDs or categories cause downstream validators to reject the task, so double-check before publishing.

## Future Development

Golden solutions/patches are planned but not yet wired in. The current format mirrors CVDP by placing RTL content in `context` and empty placeholders in `patch`, indicating which files agents may modify.

## Cleaning Task Directories

DIY task directories should only contain source code and text files. Binary files (PDFs, images, Git repositories, compiled binaries) cause JSONL bloat and harness failures.

### Automatic Cleaning (UI)

When creating a task in the UI, check **"🧹 Clean task directory before creating"** to automatically remove problematic files.

### Manual Cleaning (CLI)

Use the cleaning script to prepare your task directory:

```bash
# Preview what would be removed
python diy_tasks/clean_task.py diy_tasks/my_task --dry-run

# Actually clean the directory
python diy_tasks/clean_task.py diy_tasks/my_task
```

### What Gets Removed

The cleaning script removes:

- **Version control**: `.git`, `.gitignore`, `.gitmodules`
- **Binary docs**: PDFs, images (PNG, JPG, SVG, etc.)
- **Office files**: Word, Excel, PowerPoint, LibreOffice
- **Simulation artifacts**: `.qdb`, `.qtl`, `.wlf`, `.vcd`
- **Compiled binaries**: `.o`, `.a`, `.so`, executables
- **Archives**: `.zip`, `.tar`, `.gz`
- **RISC-V test binaries**: `rv32*`, `rv64*` compiled tests

### What Gets Kept

- Source code: `.sv`, `.v`, `.vh`, `.svh`
- Documentation: `.md`, `.txt`
- Scripts: `.py`, `.sh`
- Configuration: `.yml`, `.yaml`, `.env`

## Tips

1. **Always clean before creating** - Use the cleaning script or UI option to avoid binary file issues
2. Favor the template for new tasks—it is already wired with a passing harness
3. Keep `input.jsonl` authoritative so you can reproduce metadata quickly
4. Provide thorough specs and verification to reduce ambiguity for agents
5. Make harness scripts deterministic and ensure they exit non-zero on failure
6. Use Docker for consistent EDA tool installation if applicable
7. **Never clone Git repositories** into task directories—extract only the needed files

## Notes

- DIY task directories are not automatically added to any benchmark without running the UI/CLI steps above
- Dataset entries live under `benchmark_datasets/<benchmark>/<dataset>.jsonl`
- Rerun `build_benchmark_json.py` and `build_datatable.py` whenever you change tasks so agents/UI pick up the latest state
