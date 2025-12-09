"""
Process Monitor Component for poRTLe UI

Displays status of background processes and allows log viewing and process control.
"""

import streamlit as st
from datetime import datetime
import json
from pathlib import Path
import sys
import os

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ui.utils.process_manager import get_manager

DEMO_MODE = os.environ.get("PORTLE_DEMO_MODE", "").lower() == "true"

def format_duration(start_time: str, end_time: str = None) -> str:
    """Calculate and format duration."""
    try:
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time) if end_time else datetime.now()
        duration = end - start

        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except:
        return "Unknown"


def extract_run_id_from_log(process_id: str) -> str:
    """Extract run_id from process log if available."""
    import re
    manager = get_manager()
    success, log_content = manager.get_log(process_id, tail=None)

    if not success:
        return None

    # Try to find run_id in the log output
    # Patterns: "Generated run_id: <id>" or "run_id=<id>"
    run_id_match = re.search(r"Generated run_id:\s*([A-Za-z0-9\-_]+)", log_content)
    if not run_id_match:
        run_id_match = re.search(r"run_id=([A-Za-z0-9\-_]+)", log_content)
    if not run_id_match:
        run_id_match = re.search(r"run_id[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9\-_]+)", log_content)

    if run_id_match:
        return run_id_match.group(1)

    return None


def render_process_table(processes, show_logs=True):
    """Render a table of processes with actions."""
    if not processes:
        st.info("No processes to display")
        return

    for proc in processes:
        process_id = proc['process_id']

        # Parse metadata if available
        metadata = {}
        if proc.get('metadata'):
            try:
                metadata = json.loads(proc['metadata'])
            except:
                pass

        # Try to extract run_id from log
        run_id = extract_run_id_from_log(process_id)
        if run_id:
            metadata['run_id'] = run_id

        # Status badge with color
        status = proc['status']
        status_colors = {
            'starting': '🟡',
            'running': '🟢',
            'completed': '✅',
            'failed': '❌',
            'killed': '🔴',
            'error': '⚠️',
            'orphaned': '💀'
        }
        status_badge = status_colors.get(status, '❓')

        # Create expandable section for each process
        with st.expander(
            f"{status_badge} {proc['description']} - {status.upper()} - {format_duration(proc['start_time'], proc.get('end_time'))}",
            expanded=False
        ):
            # Action buttons at the top
            action_col1, action_col2, action_col3 = st.columns([2, 2, 6])

            with action_col1:
                # Kill button (only for running processes)
                if status in ['starting', 'running']:
                    if st.button("🛑 Kill Process", key=f"kill_{process_id}"):
                        manager = get_manager()
                        success, message = manager.kill_process(process_id)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

            with action_col2:
                # Refresh logs button
                if st.button("🔄 Refresh", key=f"refresh_{process_id}", disabled=DEMO_MODE):
                    st.rerun()

            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Process Info**")
                st.text(f"ID: {process_id}")
                st.text(f"PID: {proc['pid'] or 'N/A'}")
                st.text(f"Status: {status}")

            with col2:
                st.markdown("**Timing**")
                st.text(f"Started: {proc['start_time'][:19]}")
                if proc.get('end_time'):
                    st.text(f"Ended: {proc['end_time'][:19]}")
                st.text(f"Duration: {format_duration(proc['start_time'], proc.get('end_time'))}")

            with col3:
                st.markdown("**Result**")
                if proc.get('exit_code') is not None:
                    st.text(f"Exit code: {proc['exit_code']}")
                if proc.get('cwd'):
                    st.text(f"Working dir: {proc['cwd']}")

            # Command
            st.markdown("**Command**")
            try:
                command = json.loads(proc['command'])
                st.code(' '.join(command), language='bash')
            except:
                st.code(proc['command'])

            # Metadata and Run ID
            if metadata:
                st.markdown("**Metadata**")

                # Special handling for run_id - display it prominently
                if 'run_id' in metadata:
                    st.markdown(f"**Run ID:** `{metadata['run_id']}`")
                    st.caption("💡 Go to the **🔍 Search** page and search for this Run ID to view details")
                    # Display remaining metadata
                    other_metadata = {k: v for k, v in metadata.items() if k != 'run_id'}
                    if other_metadata:
                        st.json(other_metadata)
                else:
                    st.json(metadata)

            # Log viewer
            if show_logs:
                st.markdown("**Output Log**")

                # Tail options
                tail_options = {
                    "Last 50 lines": 50,
                    "Last 100 lines": 100,
                    "Last 500 lines": 500,
                    "Full log": None
                }

                tail_choice = st.selectbox(
                    "Log view",
                    options=list(tail_options.keys()),
                    key=f"tail_{process_id}",
                    label_visibility="collapsed"
                )

                tail = tail_options[tail_choice]

                # Get and display log
                manager = get_manager()
                success, log_content = manager.get_log(process_id, tail=tail)

                if success:
                    st.code(log_content, language='text')
                else:
                    st.error(log_content)


def render_running_processes():
    """Render the running processes section."""
    st.markdown("### 🟢 Running Processes")

    manager = get_manager()

    # Get running and starting processes
    running = manager.list_processes(status="running", limit=100)
    starting = manager.list_processes(status="starting", limit=100)

    all_active = starting + running

    # Manual refresh button
    if st.button("🔄 Refresh", key="refresh_running_processes", disabled=DEMO_MODE):
        st.rerun()

    if all_active:
        st.info(f"Found {len(all_active)} active process(es)")
        render_process_table(all_active, show_logs=True)
    else:
        st.success("No processes currently running")


def render_completed_processes():
    """Render the completed processes section."""
    st.markdown("### 📋 Process History")

    manager = get_manager()

    # Filter options
    status_filter = st.selectbox(
        "Filter by status",
        options=["All", "Completed", "Failed", "Killed", "Error", "Orphaned"],
        key="history_status_filter"
    )

    limit = st.number_input("Max results", min_value=10, max_value=1000, value=50, step=10)

    # Map filter to database status
    status_map = {
        "All": None,
        "Completed": "completed",
        "Failed": "failed",
        "Killed": "killed",
        "Error": "error",
        "Orphaned": "orphaned"
    }

    status = status_map[status_filter]

    # Get processes
    if status:
        processes = manager.list_processes(status=status, limit=limit)
    else:
        # Get all non-running processes
        all_processes = manager.list_processes(limit=limit)
        processes = [p for p in all_processes if p['status'] not in ['starting', 'running']]

    if processes:
        st.info(f"Found {len(processes)} process(es)")
        render_process_table(processes, show_logs=True)
    else:
        st.info(f"No {status_filter.lower()} processes found")


def render_cleanup_section():
    """Render the cleanup section."""
    st.markdown("### 🧹 Cleanup")

    st.markdown("""
    Clean up orphaned processes that are marked as running but have actually terminated.
    This can happen if the UI crashes or processes are killed externally.
    """)

    if st.button("🧹 Clean Up Orphaned Processes", type="secondary", disabled=DEMO_MODE):
        manager = get_manager()
        cleaned = manager.cleanup_orphaned_processes()

        if cleaned > 0:
            st.success(f"Cleaned up {cleaned} orphaned process(es)")
            st.rerun()
        else:
            st.info("No orphaned processes found")


def render():
    """Main render function for process monitor page."""
    st.title("🔄 Process Monitor")
    st.markdown("View and manage background processes")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs([
        "🟢 Running",
        "📋 History",
        "🧹 Cleanup"
    ])

    with tab1:
        render_running_processes()

    with tab2:
        render_completed_processes()

    with tab3:
        render_cleanup_section()


if __name__ == "__main__":
    # For testing component standalone
    render()
