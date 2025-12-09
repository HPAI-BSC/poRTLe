# poRTLe UI

A comprehensive Streamlit-based user interface for the Portable RTL Evaluation (poRTLe) benchmarking tool.

## Features

- **🔍 Search & Browse**: Find and view any database entry (agents, benchmarks, datasets, tasks, runs, datapoints)
- **📊 Interactive Heatmaps**: Visualize agent performance across tasks with Plotly-based interactive charts
- **✏️ Metadata Editor**: Add notes, keys, and custom fields to any entry
- **⚙️ Command Runner**: Execute Python scripts from the UI (build database, add agents, run benchmarks)

## Installation

### Prerequisites

- Python 3.8+
- poRTLe database (`results/portle.db`) - run `build_datatable.py` if not present

### Install Dependencies

```bash
cd src/ui
pip install -r requirements.txt
```

Or install globally from repo root:

```bash
pip install streamlit plotly pandas numpy pyyaml
```

## Usage

### Running the UI

From the repository root:

```bash
streamlit run src/ui/portle_ui.py
```

The UI will open in your default browser at `http://localhost:8501`

### First Time Setup

1. **Build Database** (if not already done):
   - Go to the **Commands** page
   - Click "Build Database" tab
   - Click "🔨 Build Database" button
   - Wait for completion

2. **Verify Database**:
   - Check the sidebar - it should show "✓ Connected"
   - The About page will show database statistics

## Pages

### 🔍 Search

**Find and browse database entries**

- **Select entry type**: Agent, Benchmark, Dataset, Task, Run, or Datapoint
- **Search by ID**: Enter specific ID for direct lookup
- **Filter**: Use dropdown filters to narrow results
- **View details**: Click on table rows to see full entry details
- **Edit metadata**: Click "✏️ Edit Metadata" button on detail view

**Example workflow:**
1. Select "Run" from entry type
2. Filter by benchmark and agent
3. Click on a run to see details
4. Click "Edit Metadata" to add notes

### 📊 Heatmap

**Interactive performance visualization**

- **Select benchmark and dataset**: Use sidebar filters
- **View heatmap**: Red (fail) → Yellow (partial) → Green (pass)
- **Click cells**: Click any cell to see all datapoints for that agent/task combination
- **Summary stats**: View pass rate, average score, and total datapoints
- **Filter agents**: Optionally filter to specific agents

**Example workflow:**
1. Select benchmark: "cvdp_example"
2. Select dataset: "cvdp_example__cvdp_example_no_commercial_1"
3. Click on a cell with score < 1.0
4. Review failed datapoints
5. Click "View Full Details" to investigate

### ✏️ Metadata Editor

**Edit metadata for database entries**

Available from:
- Search page → View details → Edit Metadata button
- Metadata Editor page → Quick Edit form

**Editable fields:**
- **Keys**: Add/remove keys for categorization
- **Notes**: Add dated notes with author tracking
- **Custom fields**: Add domain-specific key-value pairs

**Example workflow:**
1. Search for an agent
2. Click "Edit Metadata"
3. Add key "production-ready"
4. Add note: "Passed all tests on 01-15-25"
5. Add custom field: `fairness_score: 0.95`
6. Click "Save Changes"
7. Go to Commands → Build Database to sync changes

### ⚙️ Commands

**Execute Python scripts from the UI**

#### Build Database
- Rebuild SQLite database from all JSON files
- Required after editing metadata or adding new runs
- Shows real-time output

#### Add Agent
- Add new agent to registry via form
- Specify: ID, description, model, folder path, custom config
- Automatically saves to `results/json/agents.json`

#### Run Benchmark
- Execute benchmark runs from the UI
- Select benchmark, dataset, agent, hardware
- Configure number of runs and threads
- Creates `run.yaml` and executes `run_dataset.py`
- Shows real-time execution output

#### Build Benchmark
- Create benchmark JSON files from JSONL datasets
- (Coming soon: full CLI support)

## Data Flow

### Viewing Data
1. Database (`portle.db`) provides fast queries
2. UI reads from database for display
3. Click interactions load related data

### Editing Data
1. Edit metadata in UI
2. Changes saved to JSON files
3. **Important**: Click "Build Database" to sync changes to DB
4. UI shows banner when database is out of sync

## Tips & Tricks

### Navigation
- Use related entries links to jump between connected items
- Click on IDs in detail views to navigate
- Use browser back button to return to search results

### Performance
- Heatmap rendering may be slow for >100 tasks
- Use agent filters to reduce heatmap size
- Search results limited to 1000 rows

### Metadata Notes Best Practices
1. Set your name in Metadata Editor → User Settings
2. Add notes when making significant changes
3. Use keys for categorization (e.g., "verified", "production", "experimental")
4. Use custom fields for metrics (e.g., `fairness_score`, `latency`)

### Keyboard Shortcuts
- Streamlit standard shortcuts apply
- `R`: Rerun the app
- `C`: Clear cache
- Settings (⋮ menu) for theme and more

## Troubleshooting

### "Database not found"
**Solution**: Run `build_datatable.py` first:
```bash
python src/build_datatable.py
```

### "No benchmarks/datasets/agents found"
**Solution**: Check that JSON files exist in `results/json/`:
```bash
ls -R results/json/
```

If empty, you need to:
1. Build benchmark: `python src/build_cvdp_benchmark_json.py <benchmark_name>` (expects datasets in `benchmark_datasets/<benchmark_name>`)
2. Add agent: Use UI Commands page or edit `add_agent_to_json.py`
3. Run benchmark: Use UI or `python src/run_dataset.py`

### "Database out of sync" warning
**Solution**: Go to Commands → Build Database and click rebuild

### Heatmap not loading
**Possible causes:**
- No runs exist for selected benchmark/dataset
- Run JSON files missing from `results/json/<benchmark>/<dataset>/`

**Solution**: Run a benchmark first via Commands page

### Import errors
**Solution**: Install requirements:
```bash
pip install -r src/ui/requirements.txt
```

### Port already in use
**Solution**: Specify different port:
```bash
streamlit run src/ui/portle_ui.py --server.port 8502
```

## Architecture

```
src/ui/
├── portle_ui.py              # Main Streamlit app
├── components/                # UI components
│   ├── search.py             # Search & filter interface
│   ├── detail_view.py        # Entry detail display
│   ├── heatmap.py            # Interactive Plotly heatmap
│   ├── metadata_editor.py    # Metadata editing forms
│   └── command_runner.py     # Script execution interface
└── utils/                     # Utility modules
    ├── db_manager.py         # Database queries
    └── json_manager.py       # JSON file operations
```

## Development

### Adding New Features

1. **New component**:
   - Create file in `components/`
   - Import in `components/__init__.py`
   - Add navigation in `portle_ui.py`

2. **New command**:
   - Add tab in `command_runner.py`
   - Implement `render_*_tab()` function

3. **New database query**:
   - Add function to `utils/db_manager.py`

### Testing Components Standalone

Each component can run independently:

```bash
streamlit run src/ui/components/search.py
streamlit run src/ui/components/heatmap.py
```

## License

Part of the poRTLe project. See repository LICENSE for details.

## Contributing

See main poRTLe repository for contribution guidelines.
