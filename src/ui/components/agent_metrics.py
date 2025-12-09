"""
Agent Metrics Component for poRTLe UI

Displays an agent's performance across tasks with bar charts
and a comprehensive datapoint metrics table.

Flow:
1. Select Agent (required)
2. Optional filters for Benchmark, Dataset, Task, Run
3. Run Selection (like sample selection - Average or specific run)
4. Performance charts and metrics table
"""

import streamlit as st
from pathlib import Path
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import os

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ui.utils import db_manager, json_manager
from ui.utils.filter_engine import FilterSpec
from ui.components.heatmap import render_generic_column_filters, apply_entity_filters

DEMO_MODE = os.environ.get("PORTLE_DEMO_MODE", "").lower() == "true"


def truncate_label(label: str, max_length: int) -> str:
    """Truncate label with ellipsis if needed."""
    if len(label) <= max_length:
        return label
    if max_length <= 1:
        return label[-1:]
    return "…" + label[-(max_length - 1):]


def get_agent_datapoints(conn, agent_id: str) -> List[Dict[str, Any]]:
    """
    Get all datapoints for a specific agent with run context.

    Returns list of dicts with datapoint info plus benchmark_id, dataset_id from run.
    """
    import json

    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            d.datapoint_id, d.task_id, d.run_id, d.agent_id,
            d.score, d.tokens, d.time, d.error, d.metadata,
            r.benchmark_id, r.dataset_id
        FROM datapoints d
        JOIN runs r ON d.run_id = r.run_id
        WHERE d.agent_id = ?
        ORDER BY r.run_start DESC, d.task_id
    """, (agent_id,))

    results = []
    for row in cursor.fetchall():
        # Extract sample_index from metadata
        metadata_str = row[8]
        sample_index = 0
        if metadata_str:
            try:
                metadata = json.loads(metadata_str)
                sample_index = metadata.get("custom", {}).get("sample_index", 0)
            except (json.JSONDecodeError, TypeError):
                pass

        results.append({
            "datapoint_id": row[0],
            "task_id": row[1],
            "run_id": row[2],
            "agent_id": row[3],
            "score": row[4],
            "tokens": row[5],
            "time": row[6],
            "error": row[7],
            "metadata": metadata_str,
            "n": sample_index + 1,  # Convert 0-indexed to 1-indexed for display
            "benchmark_id": row[9],
            "dataset_id": row[10]
        })
    return results


def get_agent_benchmarks(conn, agent_id: str) -> List[str]:
    """Get benchmarks where this agent has data."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT r.benchmark_id
        FROM runs r
        WHERE r.agent_id = ? AND r.benchmark_id IS NOT NULL
        ORDER BY r.benchmark_id
    """, (agent_id,))
    return [row[0] for row in cursor.fetchall()]


def get_agent_datasets(conn, agent_id: str, benchmark_ids: Optional[List[str]] = None) -> List[str]:
    """Get datasets where this agent has data, optionally filtered by benchmark."""
    cursor = conn.cursor()
    if benchmark_ids:
        placeholders = ",".join("?" * len(benchmark_ids))
        cursor.execute(f"""
            SELECT DISTINCT r.dataset_id
            FROM runs r
            WHERE r.agent_id = ? AND r.dataset_id IS NOT NULL
              AND r.benchmark_id IN ({placeholders})
            ORDER BY r.dataset_id
        """, (agent_id, *benchmark_ids))
    else:
        cursor.execute("""
            SELECT DISTINCT r.dataset_id
            FROM runs r
            WHERE r.agent_id = ? AND r.dataset_id IS NOT NULL
            ORDER BY r.dataset_id
        """, (agent_id,))
    return [row[0] for row in cursor.fetchall()]


def get_agent_tasks(conn, agent_id: str, dataset_ids: Optional[List[str]] = None) -> List[str]:
    """Get tasks where this agent has data, optionally filtered by dataset."""
    cursor = conn.cursor()
    if dataset_ids:
        placeholders = ",".join("?" * len(dataset_ids))
        cursor.execute(f"""
            SELECT DISTINCT d.task_id
            FROM datapoints d
            JOIN runs r ON d.run_id = r.run_id
            WHERE d.agent_id = ? AND r.dataset_id IN ({placeholders})
            ORDER BY d.task_id
        """, (agent_id, *dataset_ids))
    else:
        cursor.execute("""
            SELECT DISTINCT d.task_id
            FROM datapoints d
            WHERE d.agent_id = ?
            ORDER BY d.task_id
        """, (agent_id,))
    return [row[0] for row in cursor.fetchall()]


def get_agent_runs(conn, agent_id: str, benchmark_ids: Optional[List[str]] = None,
                   dataset_ids: Optional[List[str]] = None) -> List[str]:
    """Get runs for this agent, optionally filtered by benchmark/dataset."""
    cursor = conn.cursor()
    query = "SELECT DISTINCT r.run_id FROM runs r WHERE r.agent_id = ?"
    params = [agent_id]

    if benchmark_ids:
        placeholders = ",".join("?" * len(benchmark_ids))
        query += f" AND r.benchmark_id IN ({placeholders})"
        params.extend(benchmark_ids)

    if dataset_ids:
        placeholders = ",".join("?" * len(dataset_ids))
        query += f" AND r.dataset_id IN ({placeholders})"
        params.extend(dataset_ids)

    query += " ORDER BY r.run_start DESC"
    cursor.execute(query, params)
    return [row[0] for row in cursor.fetchall()]


def apply_datapoint_filters(
    datapoints: List[Dict],
    benchmark_ids: Optional[List[str]] = None,
    dataset_ids: Optional[List[str]] = None,
    task_ids: Optional[List[str]] = None,
    run_ids: Optional[List[str]] = None
) -> List[Dict]:
    """Filter datapoints by various criteria."""
    filtered = datapoints

    if benchmark_ids:
        filtered = [dp for dp in filtered if dp.get("benchmark_id") in benchmark_ids]
    if dataset_ids:
        filtered = [dp for dp in filtered if dp.get("dataset_id") in dataset_ids]
    if task_ids:
        filtered = [dp for dp in filtered if dp.get("task_id") in task_ids]
    if run_ids:
        filtered = [dp for dp in filtered if dp.get("run_id") in run_ids]

    return filtered


def aggregate_by_task(datapoints: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate datapoints by task_id, averaging metrics where multiple exist.

    Returns dict: task_id -> {"score": avg, "tokens": avg, "time": avg}
    """
    task_data = {}

    for dp in datapoints:
        task_id = dp["task_id"]
        if task_id not in task_data:
            task_data[task_id] = {"scores": [], "tokens": [], "times": []}

        if dp["score"] is not None and dp["score"] != -1:
            task_data[task_id]["scores"].append(dp["score"])
        if dp["tokens"] is not None and dp["tokens"] != -1:
            task_data[task_id]["tokens"].append(dp["tokens"])
        if dp["time"] is not None and dp["time"] != -1:
            task_data[task_id]["times"].append(dp["time"])

    result = {}
    for task_id, data in task_data.items():
        result[task_id] = {
            "score": np.mean(data["scores"]) if data["scores"] else np.nan,
            "tokens": np.mean(data["tokens"]) if data["tokens"] else np.nan,
            "time": np.mean(data["times"]) if data["times"] else np.nan
        }

    return result


def render_agent_metrics_tab():
    """
    Render the Agent Metrics tab.

    This view shows an agent's performance with:
    - Agent selection (required)
    - Optional filters for benchmark, dataset, task, run
    - Run selection (like sample selection)
    - Bar charts for Score, Tokens, Time (sorted by priority)
    - Comprehensive metrics table for all datapoints
    - Easy access to task/agent/datapoint metadata editing
    """
    # Check if we're viewing a detail page from this tab
    if st.session_state.get("agent_metrics_viewing_detail") and st.session_state.get("selected_entry"):
        from ui.components import detail_view

        # Show detail view with back button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Back to Agent Metrics"):
                st.session_state.agent_metrics_viewing_detail = False
                st.rerun()

        with col2:
            st.title(f"📄 {st.session_state.selected_table.capitalize()} Details")

        st.markdown("---")

        # Render full detail view
        detail_view.render_detail_panel(
            st.session_state.selected_table,
            st.session_state.selected_entry
        )
        return

    st.markdown("### 🤖 Agent Metrics")
    st.markdown("Analyze an agent's performance across tasks and runs")

    # Check database
    if not st.session_state.db_path.exists():
        st.error(f"Database not found at: {st.session_state.db_path}")
        return

    try:
        conn = db_manager.connect_db(st.session_state.db_path)

        # Get all agents
        all_agents = db_manager.get_agents(conn)

        if not all_agents:
            st.warning("No agents found in database")
            conn.close()
            return

        # =================================================================
        # STEP 1: Agent Selection (Required)
        # =================================================================
        st.markdown("#### 🤖 Select Agent")

        # Determine default index - restore from saved state if available
        default_agent_index = 0
        if st.session_state.get("agent_metrics_return_agent"):
            saved_agent = st.session_state.agent_metrics_return_agent
            if saved_agent in all_agents:
                default_agent_index = all_agents.index(saved_agent)

        agent_id = st.selectbox(
            "Select an agent to analyze",
            options=all_agents,
            index=default_agent_index,
            key="agent_metrics_agent_selector",
            help="Choose an agent to view its performance metrics"
        )

        if not agent_id:
            st.info("Please select an agent to continue")
            conn.close()
            return

        # Check if agent changed - if so, clear all filters to avoid invalid selections
        previous_agent = st.session_state.get("agent_metrics_previous_agent")
        if previous_agent and previous_agent != agent_id:
            # Agent changed - clear all filter state
            for entity_type in ["benchmark", "dataset", "task", "run"]:
                prefix = f"agent_metrics_{entity_type}"
                st.session_state[f"{prefix}_field_filters"] = []
                st.session_state[f"{prefix}_metadata_filters"] = []
                st.session_state[f"{prefix}_metadata_keys"] = []
                st.session_state[f"{prefix}_manual_selection"] = []
                if f"{prefix}_multiselect" in st.session_state:
                    del st.session_state[f"{prefix}_multiselect"]
                if f"{prefix}_tag_multiselect" in st.session_state:
                    del st.session_state[f"{prefix}_tag_multiselect"]

        # Store current agent for next comparison
        st.session_state["agent_metrics_previous_agent"] = agent_id

        # Get agent info
        agent_row = db_manager.query_entry(conn, "agent", agent_id)

        st.markdown("---")

        # =================================================================
        # STEP 2: Agent Information
        # =================================================================
        with st.expander("📋 Agent Information", expanded=True):
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(f"**Agent ID:** `{agent_id}`")
                if agent_row and "about" in agent_row.keys() and agent_row["about"]:
                    st.markdown(f"**About:** {agent_row['about']}")
            with info_col2:
                if agent_row:
                    agent_metadata = json_manager.parse_metadata(agent_row["metadata"] if "metadata" in agent_row.keys() else "{}")
                    agent_keys = agent_metadata.get("keys", [])
                    if agent_keys:
                        st.markdown(f"**Keys:** {', '.join(agent_keys[:5])}")
                    agent_notes = agent_metadata.get("notes", [])
                    if agent_notes:
                        st.markdown(f"**Notes:** {len(agent_notes)} note(s)")

            # View agent details button
            if st.button("View Full Agent Details", key="view_agent_details", use_container_width=True):
                st.session_state.agent_metrics_return_agent = agent_id
                st.session_state.selected_entry = agent_id
                st.session_state.selected_table = "agent"
                st.session_state.agent_metrics_viewing_detail = True
                st.rerun()

        # =================================================================
        # STEP 3: Optional Filters (using same interface as heatmap)
        # =================================================================
        # Get data scoped to this agent for filter options
        agent_benchmarks = get_agent_benchmarks(conn, agent_id)
        agent_datasets = get_agent_datasets(conn, agent_id)
        agent_tasks = get_agent_tasks(conn, agent_id)
        agent_runs = get_agent_runs(conn, agent_id)

        # Get existing tags for each entity type
        existing_tags_by_entity = {}
        for entity_key in ["benchmark", "dataset", "task", "run"]:
            try:
                existing_tags_by_entity[entity_key] = db_manager.get_existing_keys(conn, entity_key)
            except Exception:
                existing_tags_by_entity[entity_key] = []

        # Entity filter configurations (scoped to this agent's data)
        entity_configs = {
            "benchmark": {
                "columns": ["benchmark_id", "about"],
                "available_items": agent_benchmarks,
                "available_tags": existing_tags_by_entity.get("benchmark", [])
            },
            "dataset": {
                "columns": ["dataset_id", "benchmark_id", "about", "task_count"],
                "available_items": agent_datasets,
                "available_tags": existing_tags_by_entity.get("dataset", [])
            },
            "task": {
                "columns": ["task_id", "benchmark_id", "dataset_id"],
                "available_items": agent_tasks,
                "available_tags": existing_tags_by_entity.get("task", [])
            },
            "run": {
                "columns": ["run_id", "benchmark_id", "dataset_id", "agent_id", "hardware_info", "n", "threads", "run_start", "run_end", "total_time"],
                "available_items": agent_runs,
                "available_tags": existing_tags_by_entity.get("run", [])
            }
        }

        # Render filters for each entity type
        filter_specs = {}
        for entity_type, config in entity_configs.items():
            filter_result = render_generic_column_filters(
                entity_type=entity_type,
                entity_columns=config["columns"],
                session_key_prefix=f"agent_metrics_{entity_type}",
                available_items=config.get("available_items"),
                show_completeness_toggle=False,
                available_tags=config.get("available_tags")
            )
            if filter_result and (filter_result[0] or filter_result[1]):
                filter_specs[entity_type] = filter_result

        # Check if any filters are active
        has_any_filters = bool(filter_specs)
        has_manual_selection = any(
            st.session_state.get(f"agent_metrics_{entity_type}_manual_selection", [])
            for entity_type in entity_configs.keys()
        )

        if has_any_filters or has_manual_selection:
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("🗑️ Remove All Filters", type="secondary", use_container_width=True, key="agent_metrics_clear_all"):
                    for entity_type in entity_configs.keys():
                        prefix = f"agent_metrics_{entity_type}"
                        st.session_state[f"{prefix}_field_filters"] = []
                        st.session_state[f"{prefix}_metadata_filters"] = []
                        st.session_state[f"{prefix}_metadata_keys"] = []
                        st.session_state[f"{prefix}_manual_selection"] = []
                        if f"{prefix}_multiselect" in st.session_state:
                            del st.session_state[f"{prefix}_multiselect"]
                    st.rerun()

        st.markdown("---")

        # Extract filter selections for datapoint filtering
        selected_benchmarks = None
        selected_datasets = None
        selected_tasks = None
        selected_runs_filter = None

        # Get manual selections from filter UI
        benchmark_manual = st.session_state.get("agent_metrics_benchmark_manual_selection", [])
        dataset_manual = st.session_state.get("agent_metrics_dataset_manual_selection", [])
        task_manual = st.session_state.get("agent_metrics_task_manual_selection", [])
        run_manual = st.session_state.get("agent_metrics_run_manual_selection", [])

        # Apply FilterSpec filters if present
        if "benchmark" in filter_specs:
            filter_spec, manual_sel = filter_specs["benchmark"]
            if filter_spec:
                matching = apply_entity_filters(conn, "benchmark", "benchmark_id", filter_spec, lambda c: agent_benchmarks)
                selected_benchmarks = matching if matching else None
            if manual_sel:
                if selected_benchmarks:
                    selected_benchmarks = [b for b in selected_benchmarks if b in manual_sel]
                else:
                    selected_benchmarks = manual_sel
        elif benchmark_manual:
            selected_benchmarks = benchmark_manual

        if "dataset" in filter_specs:
            filter_spec, manual_sel = filter_specs["dataset"]
            if filter_spec:
                matching = apply_entity_filters(conn, "dataset", "dataset_id", filter_spec, lambda c: agent_datasets)
                selected_datasets = matching if matching else None
            if manual_sel:
                if selected_datasets:
                    selected_datasets = [d for d in selected_datasets if d in manual_sel]
                else:
                    selected_datasets = manual_sel
        elif dataset_manual:
            selected_datasets = dataset_manual

        if "task" in filter_specs:
            filter_spec, manual_sel = filter_specs["task"]
            if filter_spec:
                matching = apply_entity_filters(conn, "task", "task_id", filter_spec, lambda c: agent_tasks)
                selected_tasks = matching if matching else None
            if manual_sel:
                if selected_tasks:
                    selected_tasks = [t for t in selected_tasks if t in manual_sel]
                else:
                    selected_tasks = manual_sel
        elif task_manual:
            selected_tasks = task_manual

        if "run" in filter_specs:
            filter_spec, manual_sel = filter_specs["run"]
            if filter_spec:
                matching = apply_entity_filters(conn, "run", "run_id", filter_spec, lambda c: agent_runs)
                selected_runs_filter = matching if matching else None
            if manual_sel:
                if selected_runs_filter:
                    selected_runs_filter = [r for r in selected_runs_filter if r in manual_sel]
                else:
                    selected_runs_filter = manual_sel
        elif run_manual:
            selected_runs_filter = run_manual

        # =================================================================
        # STEP 4: Get and Filter Datapoints
        # =================================================================
        all_datapoints = get_agent_datapoints(conn, agent_id)

        if not all_datapoints:
            st.warning(f"No datapoints found for agent: {agent_id}")
            conn.close()
            return

        # Apply filters
        filtered_datapoints = apply_datapoint_filters(
            all_datapoints,
            benchmark_ids=selected_benchmarks if selected_benchmarks else None,
            dataset_ids=selected_datasets if selected_datasets else None,
            task_ids=selected_tasks if selected_tasks else None,
            run_ids=selected_runs_filter if selected_runs_filter else None
        )

        if not filtered_datapoints:
            st.warning("No datapoints match the current filters")
            conn.close()
            return

        # =================================================================
        # STEP 5: Run Selection (replaces Sample Selection)
        # =================================================================
        # Get unique runs from filtered datapoints
        available_runs = sorted(set(dp["run_id"] for dp in filtered_datapoints))

        st.markdown("#### 🏃 Run Selection")

        run_options = ["Average (all runs)"] + available_runs

        # Determine default run index - restore from saved state if available
        default_run_index = 0
        if st.session_state.get("agent_metrics_return_run"):
            saved_run = st.session_state.agent_metrics_return_run
            if saved_run in run_options:
                default_run_index = run_options.index(saved_run)
            # Clear the return state after using it
            st.session_state.agent_metrics_return_run = None
            st.session_state.agent_metrics_return_agent = None

        selected_run_option = st.radio(
            "Select run to display in charts",
            options=run_options,
            index=default_run_index,
            horizontal=True,
            key="agent_metrics_run_selector",
            help="Choose to view average across all runs or a specific run for the bar charts"
        )

        is_average_mode = selected_run_option == "Average (all runs)"
        selected_run_id = None if is_average_mode else selected_run_option

        st.markdown("---")

        # =================================================================
        # STEP 6: Build Chart Data
        # =================================================================
        # First, calculate metrics from ALL filtered datapoints (for consistent axis ranges)
        all_task_metrics = aggregate_by_task(filtered_datapoints)
        all_tasks = sorted(all_task_metrics.keys())

        if not all_tasks:
            st.warning("No tasks found in filtered datapoints")
            conn.close()
            return

        # Calculate global axis ranges from all data (used for both average and individual run views)
        all_token_values = [all_task_metrics[t]["tokens"] for t in all_tasks]
        all_time_values = [all_task_metrics[t]["time"] for t in all_tasks]

        global_token_min = np.nanmin(all_token_values) if not all(np.isnan(all_token_values)) else 0
        global_token_max = np.nanmax(all_token_values) if not all(np.isnan(all_token_values)) else 1
        global_time_min = np.nanmin(all_time_values) if not all(np.isnan(all_time_values)) else 0
        global_time_max = np.nanmax(all_time_values) if not all(np.isnan(all_time_values)) else 1

        # Add padding to max values for better visualization
        token_range_max = global_token_max * 1.15 if global_token_max > 0 else 1
        time_range_max = global_time_max * 1.15 if global_time_max > 0 else 1

        # Now filter datapoints by run for the actual chart display
        if is_average_mode:
            chart_datapoints = filtered_datapoints
            task_metrics = all_task_metrics
        else:
            chart_datapoints = [dp for dp in filtered_datapoints if dp["run_id"] == selected_run_id]
            task_metrics = aggregate_by_task(chart_datapoints)

        # Fetch task metadata for hover display
        task_metadata_cache = {}
        for task_id in all_tasks:
            task_row = db_manager.query_entry(conn, "task", task_id)
            if task_row:
                metadata = json_manager.parse_metadata(task_row["metadata"] if "metadata" in task_row.keys() else "{}")
                task_metadata_cache[task_id] = metadata
            else:
                task_metadata_cache[task_id] = {}

        # Sort tasks by: score (desc), tokens (desc), time (asc) - ALWAYS use all_task_metrics for consistent ordering
        def sort_key(task):
            m = all_task_metrics[task]
            score = m["score"] if not np.isnan(m["score"]) else -float('inf')
            tokens = m["tokens"] if not np.isnan(m["tokens"]) else -float('inf')
            time_val = m["time"] if not np.isnan(m["time"]) else float('inf')
            return (-score, -tokens, time_val)

        tasks = sorted(all_tasks, key=sort_key)

        # Build ordered value lists (handle missing tasks in individual run view)
        score_values = [task_metrics.get(t, {}).get("score", np.nan) for t in tasks]
        token_values = [task_metrics.get(t, {}).get("tokens", np.nan) for t in tasks]
        time_values = [task_metrics.get(t, {}).get("time", np.nan) for t in tasks]

        # Create bar charts
        task_labels = [truncate_label(t, 20) for t in tasks]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[1, 1, 1],
            subplot_titles=("Score (0-1)", "Tokens", "Time (seconds)")
        )

        # Helper functions for colors
        def get_score_colors(values):
            colors = []
            for v in values:
                if np.isnan(v):
                    colors.append('rgba(128,128,128,0.3)')
                elif v >= 1.0:
                    colors.append('#1a9850')
                elif v > 0:
                    colors.append('#fee08b')
                else:
                    colors.append('#d73027')
            return colors

        def get_time_colors(values, time_min, time_max):
            colors = []
            for v in values:
                if np.isnan(v):
                    colors.append('rgba(128,128,128,0.3)')
                elif time_max == time_min:
                    colors.append('#fee08b')
                else:
                    normalized = (v - time_min) / (time_max - time_min)
                    if normalized < 0.33:
                        colors.append('#1a9850')
                    elif normalized < 0.67:
                        colors.append('#fee08b')
                    else:
                        colors.append('#d73027')
            return colors

        # Helper function to build hover text with task metadata
        def build_hover_text(task_id: str, metric_name: str, metric_value, format_str: str = "{:.2f}") -> str:
            """Build rich hover text including task metadata."""
            lines = [f"Task: {task_id}"]

            # Add metric value
            if metric_value is not None and not (isinstance(metric_value, float) and np.isnan(metric_value)):
                lines.append(f"{metric_name}: {format_str.format(metric_value)}")
            else:
                lines.append(f"{metric_name}: No data")

            # Add task metadata
            meta = task_metadata_cache.get(task_id, {})

            # Add keys if present
            keys = meta.get("keys", [])
            if keys:
                lines.append(f"Keys: {', '.join(keys[:5])}{'...' if len(keys) > 5 else ''}")

            # Add custom metadata fields
            custom = meta.get("custom", {})
            if custom:
                for key, value in list(custom.items())[:3]:
                    if value is not None and value != "":
                        lines.append(f"{key}: {value}")

            return "<br>".join(lines)

        # Score bar chart
        score_colors = get_score_colors(score_values)
        score_hover = [
            build_hover_text(tasks[i], "Score", score_values[i], "{:.2f}")
            for i in range(len(tasks))
        ]
        fig.add_trace(go.Bar(
            x=task_labels,
            y=[v if not np.isnan(v) else 0 for v in score_values],
            marker_color=score_colors,
            text=[f"{v:.2f}" if not np.isnan(v) else "N/A" for v in score_values],
            textposition='outside',
            textfont=dict(size=8),
            hovertext=score_hover,
            hovertemplate='%{hovertext}<extra></extra>',
            name="Score",
            showlegend=False
        ), row=1, col=1)

        # Tokens bar chart
        token_min = np.nanmin(token_values) if not all(np.isnan(token_values)) else 0
        token_max = np.nanmax(token_values) if not all(np.isnan(token_values)) else 1
        token_hover = [
            build_hover_text(tasks[i], "Tokens", token_values[i], "{:.0f}")
            for i in range(len(tasks))
        ]
        token_colors = []
        for v in token_values:
            if np.isnan(v):
                token_colors.append('rgba(128,128,128,0.3)')
            elif token_max == token_min:
                token_colors.append('#6baed6')
            else:
                normalized = (v - token_min) / (token_max - token_min)
                r = int(239 - normalized * 200)
                g = int(248 - normalized * 120)
                b = int(253 - normalized * 70)
                token_colors.append(f'rgb({r},{g},{b})')

        fig.add_trace(go.Bar(
            x=task_labels,
            y=[v if not np.isnan(v) else 0 for v in token_values],
            marker_color=token_colors,
            text=[f"{int(v)}" if not np.isnan(v) else "N/A" for v in token_values],
            textposition='outside',
            textfont=dict(size=8),
            hovertext=token_hover,
            hovertemplate='%{hovertext}<extra></extra>',
            name="Tokens",
            showlegend=False
        ), row=2, col=1)

        # Time bar chart
        time_min = np.nanmin(time_values) if not all(np.isnan(time_values)) else 0
        time_max = np.nanmax(time_values) if not all(np.isnan(time_values)) else 1
        time_colors = get_time_colors(time_values, time_min, time_max)
        time_hover = [
            build_hover_text(tasks[i], "Time", time_values[i], "{:.1f}s")
            for i in range(len(tasks))
        ]
        fig.add_trace(go.Bar(
            x=task_labels,
            y=[v if not np.isnan(v) else 0 for v in time_values],
            marker_color=time_colors,
            text=[f"{v:.1f}s" if not np.isnan(v) else "N/A" for v in time_values],
            textposition='outside',
            textfont=dict(size=8),
            hovertext=time_hover,
            hovertemplate='%{hovertext}<extra></extra>',
            name="Time",
            showlegend=False
        ), row=3, col=1)

        # Update layout
        width = max(900, len(tasks) * 40 + 200)

        # Build title
        if is_average_mode:
            chart_title = f"Agent: {agent_id[:40]}{'...' if len(agent_id) > 40 else ''}"
            if len(available_runs) > 1:
                chart_title += f" (avg of {len(available_runs)} runs)"
        else:
            chart_title = f"Agent: {agent_id[:30]}{'...' if len(agent_id) > 30 else ''} - {selected_run_id[:20]}{'...' if len(selected_run_id) > 20 else ''}"

        fig.update_layout(
            title=dict(
                text=chart_title,
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            height=700,
            width=width,
            margin=dict(l=80, r=50, t=80, b=150),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            template='plotly_dark',
            bargap=0.15
        )

        # Use fixed axis ranges based on all data (for consistent comparison across runs)
        fig.update_yaxes(title_text="Score", range=[0, 1.15], row=1, col=1)
        fig.update_yaxes(title_text="Tokens", range=[0, token_range_max], row=2, col=1)
        fig.update_yaxes(title_text="Time (s)", range=[0, time_range_max], row=3, col=1)

        fig.update_xaxes(tickangle=-45, tickfont=dict(size=9), title_text="Tasks", row=3, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=1)

        # Display charts
        st.markdown("#### 📊 Performance Metrics")
        st.plotly_chart(fig, use_container_width=True, key="agent_metrics_chart")

        st.markdown("---")

        # =================================================================
        # STEP 7: Summary Statistics
        # =================================================================
        st.markdown("#### 📈 Summary Statistics")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        valid_scores = [s for s in score_values if not np.isnan(s)]
        valid_times = [t for t in time_values if not np.isnan(t)]
        valid_tokens = [t for t in token_values if not np.isnan(t)]

        with stat_col1:
            avg_score = np.mean(valid_scores) if valid_scores else 0
            pass_rate = (sum(1 for s in valid_scores if s >= 1.0) / len(valid_scores) * 100) if valid_scores else 0
            st.metric("Avg Score", f"{avg_score:.2f}")
            st.caption(f"Pass rate: {pass_rate:.1f}%")

        with stat_col2:
            st.metric("Tasks", len(tasks))
            st.caption(f"With data: {len(valid_scores)}")

        with stat_col3:
            avg_time = np.mean(valid_times) if valid_times else 0
            st.metric("Avg Time", f"{avg_time:.1f}s")
            if valid_times:
                st.caption(f"Range: {min(valid_times):.1f}s - {max(valid_times):.1f}s")

        with stat_col4:
            avg_tokens = np.mean(valid_tokens) if valid_tokens else 0
            st.metric("Avg Tokens", f"{avg_tokens:.0f}")
            if valid_tokens:
                st.caption(f"Total: {sum(valid_tokens):.0f}")

        st.markdown("---")

        # =================================================================
        # STEP 8: Comprehensive Metrics Table
        # =================================================================
        st.markdown("#### 📊 Comprehensive Metrics Table")
        st.caption(f"Showing **{len(filtered_datapoints)} datapoints** matching filters - Click any row's action buttons to view/edit")

        # Build table data from filtered datapoints
        table_data = []
        for i, dp in enumerate(filtered_datapoints):
            task_id = dp["task_id"]
            run_id = dp["run_id"]
            sample_n = dp["n"]
            full_datapoint_id = dp["datapoint_id"]

            dp_score = dp["score"] if dp["score"] is not None else np.nan
            dp_time = dp["time"] if dp["time"] is not None and dp["time"] != -1 else np.nan
            dp_tokens = dp["tokens"] if dp["tokens"] is not None and dp["tokens"] != -1 else np.nan

            metadata = json_manager.parse_metadata(dp.get("metadata", "{}"))
            keys = metadata.get("keys", [])
            keys_str = ", ".join(keys) if keys else ""
            notes_count = len(metadata.get("notes", []))
            notes_count_str = f"{notes_count} note(s)" if notes_count > 0 else ""

            if not np.isnan(dp_score):
                if dp_score >= 1.0:
                    status = "✅ Pass"
                    status_sort = 2
                elif dp_score > 0:
                    status = "⚠️ Partial"
                    status_sort = 1
                else:
                    status = "❌ Fail"
                    status_sort = 0
            else:
                status = "❓ N/A"
                status_sort = -1

            table_data.append({
                "#": i + 1,
                "Run": run_id[:25] + "..." if len(run_id) > 25 else run_id,
                "Sample": sample_n,
                "Task ID": task_id,
                "Status": status,
                "Score": f"{dp_score:.2f}" if not np.isnan(dp_score) else "N/A",
                "Time (s)": f"{dp_time:.1f}" if not np.isnan(dp_time) else "N/A",
                "Tokens": f"{int(dp_tokens)}" if not np.isnan(dp_tokens) else "N/A",
                "Keys": keys_str[:30] + "..." if len(keys_str) > 30 else keys_str,
                "Notes": notes_count_str,
                "_status_sort": status_sort,
                "_score_val": dp_score if not np.isnan(dp_score) else -1,
                "_tokens_val": dp_tokens if not np.isnan(dp_tokens) else -1,
                "_time_val": dp_time if not np.isnan(dp_time) else float('inf'),
                "_task_id": task_id,
                "_run_id": run_id,
                "_full_datapoint_id": full_datapoint_id,
            })

        if table_data:
            df = pd.DataFrame(table_data)

            # Sorting controls
            sort_col1, sort_col2 = st.columns([1, 3])
            with sort_col1:
                sort_by = st.selectbox(
                    "Sort by",
                    options=["Status", "Score", "Tokens", "Time (s)", "Task ID", "Run", "Sample"],
                    index=0,
                    key="table_sort_by"
                )
            with sort_col2:
                sort_order = st.radio(
                    "Order",
                    options=["Descending", "Ascending"],
                    index=0,
                    horizontal=True,
                    key="table_sort_order"
                )

            # Apply sorting
            if sort_by == "Status":
                df_sorted = df.sort_values("_status_sort", ascending=(sort_order == "Ascending"))
            elif sort_by == "Score":
                df_sorted = df.sort_values("_score_val", ascending=(sort_order == "Ascending"))
            elif sort_by == "Tokens":
                df_sorted = df.sort_values("_tokens_val", ascending=(sort_order == "Ascending"))
            elif sort_by == "Time (s)":
                df_sorted = df.sort_values("_time_val", ascending=(sort_order == "Ascending"))
            elif sort_by == "Sample":
                df_sorted = df.sort_values("Sample", ascending=(sort_order == "Ascending"))
            elif sort_by == "Run":
                df_sorted = df.sort_values("Run", ascending=(sort_order == "Ascending"))
            else:
                df_sorted = df.sort_values(sort_by, ascending=(sort_order == "Ascending"))

            # Display datapoints as expanders
            for _, row in df_sorted.iterrows():
                task_id = row["_task_id"]
                run_id = row["_run_id"]
                full_dp_id = row["_full_datapoint_id"]
                sample_n = row["Sample"]

                # Get full metadata for notes display
                dp_metadata = None
                for dp in filtered_datapoints:
                    if dp["datapoint_id"] == full_dp_id:
                        dp_metadata = json_manager.parse_metadata(dp.get("metadata", "{}"))
                        break
                if dp_metadata is None:
                    dp_metadata = {}

                notes_list = dp_metadata.get("notes", [])
                keys_list = dp_metadata.get("keys", [])

                # Include run in expander label
                run_short = run_id[:20] + "..." if len(run_id) > 20 else run_id
                expander_label = f"{row['Status']} | {run_short} | Task: {task_id}"

                with st.expander(expander_label, expanded=False):
                    main_col, action_col = st.columns([4, 1])

                    with main_col:
                        # Task, Run, and Datapoint ID
                        st.markdown(f"**Task:** `{task_id}`")
                        st.markdown(f"**Run:** `{run_id}`")
                        st.markdown(f"**Datapoint ID:** `{full_dp_id}`")

                        # Metrics on one line
                        st.markdown(f"**Score:** {row['Score']}  &nbsp;&nbsp;&nbsp; **Tokens:** {row['Tokens']}  &nbsp;&nbsp;&nbsp; **Time:** {row['Time (s)']}  &nbsp;&nbsp;&nbsp; **Sample #:** {sample_n}")

                        # Keys
                        keys_display = ", ".join(keys_list) if keys_list else "*None*"
                        st.markdown(f"**Keys:** {keys_display}")

                        # Notes - show actual content
                        st.markdown("**Notes:**")
                        if notes_list:
                            for note in notes_list:
                                date_added = note.get("date_added", "Unknown date")
                                author = note.get("author", "Unknown")
                                text = note.get("text", "")
                                st.markdown(f"- *{date_added}* — **{author}**: {text}")
                        else:
                            st.markdown("*No notes*")

                    with action_col:
                        st.markdown("**Quick Actions:**")
                        if st.button("📝 Task", key=f"view_task_table_{full_dp_id}", use_container_width=True, help="View task details"):
                            st.session_state.agent_metrics_return_agent = agent_id
                            st.session_state.agent_metrics_return_run = selected_run_option
                            st.session_state.selected_entry = task_id
                            st.session_state.selected_table = "task"
                            st.session_state.agent_metrics_viewing_detail = True
                            st.rerun()

                        if st.button("📊 View", key=f"view_dp_table_{full_dp_id}", use_container_width=True, help="View datapoint details"):
                            st.session_state.agent_metrics_return_agent = agent_id
                            st.session_state.agent_metrics_return_run = selected_run_option
                            st.session_state.selected_entry = full_dp_id
                            st.session_state.selected_table = "datapoint"
                            st.session_state.agent_metrics_viewing_detail = True
                            st.rerun()

                        if st.button("✏️ Edit", key=f"edit_dp_table_{full_dp_id}", use_container_width=True, help="Edit datapoint metadata", disabled=DEMO_MODE):
                            st.session_state.agent_metrics_return_agent = agent_id
                            st.session_state.agent_metrics_return_run = selected_run_option
                            st.session_state.selected_entry = full_dp_id
                            st.session_state.selected_table = "datapoint"
                            edit_mode_key = f"edit_mode_datapoint_{full_dp_id}"
                            st.session_state[edit_mode_key] = True
                            st.session_state.agent_metrics_viewing_detail = True
                            st.rerun()
        else:
            st.warning("No datapoints available")

        st.markdown("---")
        conn.close()

    except Exception as e:
        st.error(f"Error rendering agent metrics: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
