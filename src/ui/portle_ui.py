#!/usr/bin/env python3
"""
poRTLe UI - Portable RTL Evaluation Tool User Interface

Main Streamlit application for viewing, analyzing, and editing poRTLe benchmark data.

Usage:
    streamlit run src/ui/portle_ui.py
"""

import streamlit as st
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

DEMO_MODE = os.environ.get("PORTLE_DEMO_MODE", "").lower() == "true"

# Import components (will be created in subsequent phases)
try:
    from ui.components import search, heatmap, command_runner, process_monitor, scoreboard
    components_available = True
except ImportError:
    components_available = False

# Configure Streamlit page
st.set_page_config(
    page_title="poRTLe - Portable RTL Evaluation Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


def check_password():
    """Returns True if password is correct or no password is configured."""
    # Only require password in demo mode
    if not DEMO_MODE:
        return True
    
    # Check if password is configured in secrets
    try:
        configured_password = st.secrets.get("password", "")
    except Exception:
        configured_password = ""
    
    # If no password configured, allow access
    if not configured_password:
        return True
    
    # Check if already authenticated
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return True
    
    # Show login form
    st.title("🔐 poRTLe Login")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if password == configured_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


# Check password before proceeding
if not check_password():
    st.stop()


def initialize_session_state():
    """Initialize session state variables."""
    if 'db_path' not in st.session_state:
        # Default database path
        repo_root = Path(__file__).parent.parent.parent
        st.session_state.db_path = repo_root / "results" / "portle.db"

    if 'selected_entry' not in st.session_state:
        st.session_state.selected_entry = None

    if 'selected_table' not in st.session_state:
        st.session_state.selected_table = None

    if 'db_sync_needed' not in st.session_state:
        st.session_state.db_sync_needed = False

    # Clean up orphaned processes on first run
    if 'processes_cleaned' not in st.session_state:
        try:
            from ui.utils.process_manager import get_manager
            manager = get_manager()
            cleaned = manager.cleanup_orphaned_processes()
            st.session_state.processes_cleaned = True
            if cleaned > 0:
                # Store cleanup message to show in sidebar
                st.session_state.cleanup_message = f"Cleaned up {cleaned} orphaned process(es)"
        except Exception:
            # Silently fail if process manager not available
            st.session_state.processes_cleaned = True


def render_sidebar():
    """Render sidebar with navigation and settings."""
    if DEMO_MODE:
        st.sidebar.title("🎯 poRTLe (demo)")
    else:
        st.sidebar.title("🎯 poRTLe")

    st.sidebar.markdown("---")  

    # Navigation
    st.sidebar.subheader("Navigation")

    if components_available:
        page = st.sidebar.radio(
            "Select Page",
            ["🏆 Scoreboard", "🔍 Search", "📊 Plots", "⚙️ Commands", "🔄 Process Monitor", "ℹ️ About"],
            label_visibility="collapsed"
        )
    else:
        # Fallback when components not yet created
        page = st.sidebar.radio(
            "Select Page",
            ["ℹ️ About"],
            label_visibility="collapsed"
        )

    st.sidebar.markdown("---")

    # Database status
    st.sidebar.subheader("Database")
    if st.session_state.db_path.exists():
        st.sidebar.success(f"✓ Connected")
        st.sidebar.caption(f"Path: {st.session_state.db_path}")
    else:
        st.sidebar.error(f"✗ Not found")
        st.sidebar.caption(f"Expected: {st.session_state.db_path}")

    # Sync status
    if st.session_state.db_sync_needed:
        st.sidebar.warning("⚠️ Database out of sync")
        if st.sidebar.button("Rebuild Database Now"):
            # Will be implemented in command_runner component
            st.sidebar.info("Database rebuild will be implemented in Phase 5")
            st.session_state.db_sync_needed = False

    st.sidebar.markdown("---")

    # Settings
    with st.sidebar.expander("⚙️ Settings"):
        # Database path selector
        custom_db = st.text_input(
            "Custom DB Path",
            value=str(st.session_state.db_path),
            help="Path to portle.db file"
        )
        if custom_db != str(st.session_state.db_path):
            st.session_state.db_path = Path(custom_db)
            st.rerun()

    return page


def render_about_page():
    """Render the about page."""
    st.title("About 🎯 poRTLe")

    st.markdown("""
    **poRTLe** is a platform for running, tracking, and analyzing AI agent performance across RTL benchmarks, datasets, and tasks.
    ### Overview

    poRTLe provides a unified interface for:

    - Running AI agents against multiple benchmarks (CVDP, TuRTLe, DIY, etc.)
    - Creating custom DIY tasks from your own RTL code
    - Tracking results in a structured database with rich metadata
    - Visualizing performance with an interactive UI
    - Managing benchmarks with persistent background execution
    - Filtering and analyzing tasks with advanced query capabilities

    ### Pages

    - **🏆 Scoreboard** - Agent and task leaderboards with ranked performance metrics
    - **🔍 Search** - Find and view any entry (agents, benchmarks, datasets, tasks, runs, datapoints)
    - **📊 Plots** - Interactive heatmaps and visualizations with drill-down capabilities
    - **⚙️ Commands** - Execute all Python scripts from the UI with rich configuration options
    - **🔄 Process Monitor** - View status and logs for background benchmark executions
    - **ℹ️ About** - System overview and database statistics

    ### How to Use

    1. **Build Benchmark**: Use Commands ▸ Build Benchmark to create benchmark JSON from JSONL datasets
    2. **Add Agent**: Use Commands ▸ Add Agent to register new agents with metadata
    3. **Run Benchmark**: Use Commands ▸ Run Benchmark to execute benchmarks with task filtering and background execution
    4. **Monitor**: Use Process Monitor to view status and logs for running benchmarks
    5. **Analyze**: Use Plots and Search to visualize and query results
    6. **Share**: Use Metadata Editor to add notes, keys, or variables to any DataType to share with collaborators. 
    7. **Create DIY Tasks**: Use Commands ▸ Create DIY Task to add custom tasks from your RTL code

    ### Data Hierarchy

    ```
    Benchmarks (CVDP, TuRTLe, etc.)
      ↓ contains
    Datasets (Commercial/Non-Commercial, Agentic/Non-Agentic)
      ↓ contains
    Tasks (Individual problems/circuits)
      ↓ tested by
    Agents (AI models/systems)
      ↓ produces
    Runs (Benchmark execution instances)
      ↓ generates
    Datapoints (Results: score, time, tokens, errors)
    ```
    ### Key Features

    **Metadata System**
    Add rich metadata to any entity with keys, timestamped notes, and custom fields

    **Background Execution**
    Run benchmarks as persistent background processes that continue even if you close the UI

    **Advanced Task Filtering**
    Filter tasks by difficulty, categories, or custom fields to tailor benchmark runs

    **LLM Mode (force-copilot)**
    For CVDP benchmarks, run in LLM-only mode without Docker agents

    **DIY Task Creation**
    Create custom tasks from your own RTL code with automatic JSONL conversion

    ### Quick Start

    Clone the external projects that live in gitignored directories (only the runner is required for the quick start):

    - **`benchmark_runners/`** – *Required for Quick Start.* Follow `benchmark_runners/README.md` to clone the CVDP runner (`cvdp_benchmark`) **and** set up its `cvdp_env` virtual environment.
    - **`benchmark_datasets/`** – Benchmark task datasets (optional extras for Quick Start; see `benchmark_datasets/README.md`.
    - **`agents/`** – Agent implementations (optional; see `agents/README.md`.
    - **`results/`** – Shared results repository (optional for Quick Start; see `results/README.md`.

    **Note:** `results/` is its **own git repository** so teams can share benchmark data independently of the main codebase.
                
    ### Current Status
    """)

    if not components_available:
        st.info("""
        **UI Components Loading...**

        Some components are still being implemented:
        - ⏳ Search component
        - ⏳ Plots component
        - ⏳ Metadata editor
        - ⏳ Command runner

        Check back soon!
        """)
    else:
        st.success("✓ All components loaded successfully!")

    st.markdown("---")

    # Database statistics
    if st.session_state.db_path.exists():
        st.subheader("📊 Database Statistics")

        try:
            import sqlite3
            conn = sqlite3.connect(str(st.session_state.db_path))
            cursor = conn.cursor()

            tables = ["agents", "benchmarks", "datasets", "tasks", "runs", "datapoints"]
            stats = {}

            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[table] = count

            conn.close()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Agents", stats.get("agents", 0))
                st.metric("Benchmarks", stats.get("benchmarks", 0))

            with col2:
                st.metric("Datasets", stats.get("datasets", 0))
                st.metric("Tasks", stats.get("tasks", 0))

            with col3:
                st.metric("Runs", stats.get("runs", 0))
                st.metric("Datapoints", stats.get("datapoints", 0))

        except Exception as e:
            st.error(f"Error loading database statistics: {e}")
    else:
        st.warning("Database not found. Run `build_datatable.py` to create it.")

    st.markdown("---")

    # Version info
    st.caption("poRTLe UI v1.0 | Built with Streamlit + Plotly")


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Route to appropriate page
    if page == "🏆 Scoreboard" and components_available:
        scoreboard.render()
    elif page == "🔍 Search" and components_available:
        search.render()
    elif page == "📊 Plots" and components_available:
        heatmap.render()
    elif page == "⚙️ Commands" and components_available:
        command_runner.render()
    elif page == "🔄 Process Monitor" and components_available:
        process_monitor.render()
    else:  # About page or fallback
        render_about_page()


if __name__ == "__main__":
    main()
