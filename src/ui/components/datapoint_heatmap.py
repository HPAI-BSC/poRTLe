"""
Datapoint Heatmap Component for poRTLe UI

Interactive Plotly-based heatmap for visualizing agent performance across tasks.
Allows filtering by agent, task, run, and datapoint with click-to-view details.
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import sys
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any, Tuple, Optional
import base64
from uuid import uuid4
import time
import json

try:
    from streamlit_plotly_events import plotly_events
except ImportError:  # pragma: no cover - optional dependency for interactive clicks
    plotly_events = None

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ui.utils import db_manager, json_manager
from ui.utils.filter_engine import FilterSpec


def load_run_jsons_for_filters(
    repo_root: Path,
    benchmark_id: str,
    dataset_id: str
) -> List[Dict[str, Any]]:
    """
    Load run JSONs for selected benchmark/dataset.

    Args:
        repo_root: Repository root directory
        benchmark_id: Benchmark ID
        dataset_id: Dataset ID

    Returns:
        List of run dictionaries
    """
    return json_manager.load_run_jsons_for_dataset(repo_root, benchmark_id, dataset_id)


def open_datapoints_view(
    selected_agent: str,
    selected_task: str,
    runs: List[Dict[str, Any]],
    benchmark_ids: List[str],
    dataset_ids: List[str],
    results: Dict[str, Dict[str, List[float]]]
):
    """Navigate to datapoint view if the combination exists."""
    if selected_agent in results and selected_task in results[selected_agent]:
        st.session_state.heatmap_agent = selected_agent
        st.session_state.heatmap_task = selected_task
        # Store lists - runs will be re-fetched from all combinations when viewing
        st.session_state.heatmap_benchmarks_for_view = benchmark_ids
        st.session_state.heatmap_datasets_for_view = dataset_ids
        st.session_state.viewing_datapoints = True
        st.rerun()
    else:
        st.warning(f"⚠️ No datapoints found for {selected_agent} × {selected_task}")


def render_full_heatmap_button(fig: go.Figure):
    """Render a styled link that opens the figure in a new tab."""
    inner_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    full_html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          body {{
            margin: 0;
            background-color: #0b0b15;
            color: #f5f5f5;
            font-family: 'Inter', sans-serif;
          }}
        </style>
      </head>
      <body>
        {inner_html}
      </body>
    </html>
    """
    b64 = base64.b64encode(full_html.encode('utf-8')).decode('utf-8')
    button_id = f"heatmap-full-btn-{uuid4().hex}"
    components.html(
        f"""
        <style>
        .heatmap-full-btn {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.35rem 0.85rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(255,255,255,0.2);
            background: linear-gradient(90deg, #262730, #1f1f29);
            color: #f5f5f5;
            font-size: 0.9rem;
            cursor: pointer;
        }}
        .heatmap-full-btn:hover {{
            border-color: #ff4b4b;
        }}
        </style>
        <button class="heatmap-full-btn" id="{button_id}">🖥️ Open Full Heatmap</button>
        <script>
        (function() {{
            const payload = "{b64}";
            const btn = document.getElementById("{button_id}");
            if (btn) {{
                btn.addEventListener("click", () => {{
                    const tab = window.open("about:blank", "_blank");
                    if (tab) {{
                        tab.document.open();
                        tab.document.write(atob(payload));
                        tab.document.close();
                    }}
                }});
            }}
        }})();
        </script>
        """,
        height=80,
    )


def truncate_label(label: str, max_length: int) -> str:
    """Truncate label with ellipsis if needed."""
    if len(label) <= max_length:
        return label
    if max_length <= 1:
        return label[-1:]
    return "…" + label[-(max_length - 1):]


def aggregate_results(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Aggregate results by agent and task.

    Args:
        runs: List of run dictionaries

    Returns:
        Nested dict: {agent_id: {task_id: [datapoint_dicts]}}
    """
    results = {}

    for run in runs:
        agent_id = run["agent_id"]

        if agent_id not in results:
            results[agent_id] = {}

        for datapoint in run.get("datapoints", []):
            task_id = datapoint["task_id"]

            if task_id not in results[agent_id]:
                results[agent_id][task_id] = []

            results[agent_id][task_id].append(datapoint)

    return results


def create_heatmap_data(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]],
    task_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    metric: str = "score"
) -> Tuple[np.ndarray, List[str], List[str], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Create heatmap matrix and labels from aggregated results.

    Args:
        results: Aggregated results dict
        task_metadata: Optional dict mapping task_id to task metadata
        metric: Metric to display ("score", "tokens", "time", or "error")

    Returns:
        Tuple of (matrix, agent_labels, task_labels, agent_stats, task_stats, cell_metadata)
    """
    # Get unique agents and tasks (initially unsorted)
    all_agents = list(results.keys())
    all_tasks = set()
    for agent_results in results.values():
        all_tasks.update(agent_results.keys())
    all_tasks = list(all_tasks)

    # First pass: calculate balanced statistics for all agents and tasks
    agent_stats_temp = {}
    task_stats_temp = {task: {"task_avg_scores": []} for task in all_tasks}

    for agent in all_agents:
        agent_task_avg_scores = []

        for task in all_tasks:
            if task in results[agent]:
                datapoints = results[agent][task]
                # Calculate average metric for this agent-task combination
                if metric == "error":
                    task_values = []
                    for dp in datapoints:
                        error_val = dp[metric]
                        if error_val and error_val not in [0, 1, "0", "1"]:
                            task_values.append(1.0)
                        else:
                            task_values.append(0.0)
                elif metric in ["tokens", "time"]:
                    task_values = [dp[metric] for dp in datapoints if dp[metric] != -1]
                else:
                    task_values = [dp[metric] for dp in datapoints]

                if task_values:
                    avg_value = np.mean(task_values)
                    agent_task_avg_scores.append(avg_value)
                    task_stats_temp[task]["task_avg_scores"].append(avg_value)

        if agent_task_avg_scores:
            total_score = sum(agent_task_avg_scores)
            total_tasks = len(agent_task_avg_scores)
            agent_stats_temp[agent] = {
                "total_score": total_score,
                "total": total_tasks,
                "percentage": (total_score / total_tasks * 100) if total_tasks > 0 else 0.0
            }
        else:
            agent_stats_temp[agent] = {
                "total_score": 0.0,
                "total": 0,
                "percentage": 0.0
            }

    for task in all_tasks:
        agent_avg_scores = task_stats_temp[task]["task_avg_scores"]
        if agent_avg_scores:
            total_score = sum(agent_avg_scores)
            total_agents = len(agent_avg_scores)
            task_stats_temp[task]["total_score"] = total_score
            task_stats_temp[task]["total"] = total_agents
            task_stats_temp[task]["percentage"] = (total_score / total_agents * 100) if total_agents > 0 else 0.0
        else:
            task_stats_temp[task]["total_score"] = 0.0
            task_stats_temp[task]["total"] = 0
            task_stats_temp[task]["percentage"] = 0.0

    # Sort agents and tasks
    agents = sorted(all_agents, key=lambda a: agent_stats_temp[a]["percentage"], reverse=False)
    tasks = sorted(all_tasks, key=lambda t: task_stats_temp[t]["percentage"], reverse=True)

    # Create matrix with sorted agents and tasks
    matrix = np.full((len(agents), len(tasks)), np.nan)
    agent_stats = {}
    task_stats = {}
    cell_metadata = {}

    for i, agent in enumerate(agents):
        if agent not in cell_metadata:
            cell_metadata[agent] = {}

        for j, task in enumerate(tasks):
            if task in results[agent]:
                datapoints = results[agent][task]
                if metric == "error":
                    values = []
                    for dp in datapoints:
                        error_val = dp[metric]
                        if error_val and error_val not in [0, 1, "0", "1"]:
                            values.append(1.0)
                        else:
                            values.append(0.0)
                elif metric in ["tokens", "time"]:
                    values = [dp[metric] for dp in datapoints if dp[metric] != -1]
                else:
                    values = [dp[metric] for dp in datapoints]

                if values:
                    matrix[i, j] = np.mean(values)
                else:
                    matrix[i, j] = np.nan

                cvdp_cid = "N/A"
                cvdp_difficulty = "N/A"
                if task_metadata and task in task_metadata:
                    task_meta = task_metadata[task]
                    if "metadata" in task_meta and "custom" in task_meta["metadata"]:
                        custom = task_meta["metadata"]["custom"]
                        cvdp_cid = custom.get("cvdp_cid", "N/A")
                        cvdp_difficulty = custom.get("cvdp_difficulty", "N/A")

                cell_metadata[agent][task] = {
                    "cvdp_cid": cvdp_cid,
                    "cvdp_difficulty": cvdp_difficulty,
                    "num_datapoints": len(datapoints)
                }

        agent_stats[agent] = agent_stats_temp[agent]

    for task in tasks:
        task_stats[task] = task_stats_temp[task]

    return matrix, agents, tasks, agent_stats, task_stats, cell_metadata


def create_plotly_heatmap(
    matrix: np.ndarray,
    agents: List[str],
    tasks: List[str],
    title: str = "Agent Performance Heatmap",
    min_cell_width: int = 90,
    min_cell_height: int = 55,
    task_label_truncate: Optional[int] = None,
    base_width: int = 900,
    agent_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    task_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    cell_metadata: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    metric: str = "score"
) -> Tuple[go.Figure, Dict[str, int]]:
    """
    Create Plotly heatmap figure.

    Args:
        matrix: 2D array of metric values
        agents: Agent labels (rows)
        tasks: Task labels (columns)
        title: Plot title
        min_cell_width: Minimum cell width in pixels
        min_cell_height: Minimum cell height in pixels
        task_label_truncate: Maximum length for task labels
        base_width: Base width for the figure
        agent_stats: Statistics per agent (passing/total/percentage)
        task_stats: Statistics per task (passing/total/percentage)
        cell_metadata: Metadata for each cell (cvdp_cid, cvdp_difficulty, etc.)
        metric: Metric being displayed ("score", "tokens", "time", or "error")

    Returns:
        Tuple containing the Plotly figure and layout sizing info
    """
    metric_display = metric.capitalize()
    hover_text = []
    for i, agent in enumerate(agents):
        hover_row = []
        for j, task in enumerate(tasks):
            if not np.isnan(matrix[i, j]):
                hover_parts = [
                    f"Agent: {agent}",
                    f"Task: {task}",
                ]

                if metric == "score":
                    hover_parts.append(f"Score: {matrix[i, j]:.2f}")
                elif metric == "tokens":
                    hover_parts.append(f"Avg Tokens: {matrix[i, j]:.0f}")
                elif metric == "time":
                    hover_parts.append(f"Avg Time: {matrix[i, j]:.2f}s")
                elif metric == "error":
                    hover_parts.append(f"Error Rate: {matrix[i, j]:.2%}")

                if cell_metadata and agent in cell_metadata and task in cell_metadata[agent]:
                    metadata = cell_metadata[agent][task]
                    hover_parts.append(f"CVDP CID: {metadata.get('cvdp_cid', 'N/A')}")
                    hover_parts.append(f"CVDP Difficulty: {metadata.get('cvdp_difficulty', 'N/A')}")
                    if metadata.get('num_datapoints', 1) > 1:
                        hover_parts.append(f"Datapoints: {metadata['num_datapoints']}")

                hover_row.append("<br>".join(hover_parts))
            else:
                if metric in ["tokens", "time"]:
                    hover_row.append(f"No {metric} measurements")
                elif metric == "error":
                    hover_row.append(f"No execution errors")
                else:
                    hover_row.append(f"No data")
        hover_text.append(hover_row)

    def _to_serializable(data: np.ndarray) -> List[List[Any]]:
        serializable = []
        for row in data:
            serializable.append([
                None if (isinstance(val, float) and np.isnan(val)) else float(val)
                if isinstance(val, (np.floating, float, int)) else val
                for val in row
            ])
        return serializable

    # Detect theme for styling
    is_dark_mode = True  # Default to dark

    # Check st.context.theme (returns dict like {'type': 'light'} or {'type': 'dark'})
    if hasattr(st, 'context') and hasattr(st.context, 'theme'):
        theme_info = st.context.theme
        if theme_info:
            # It's a dict, check 'type' key
            if isinstance(theme_info, dict):
                theme_type = theme_info.get('type', 'dark')
            else:
                # Fallback for object-style access
                theme_type = getattr(theme_info, 'type', None) or getattr(theme_info, 'base', 'dark')
            is_dark_mode = theme_type != 'light'

    if is_dark_mode:
        grid_color = 'rgba(255, 255, 255, 0.5)'
        text_color = '#f5f5f5'
    else:
        grid_color = 'rgba(0, 0, 0, 0.5)'
        text_color = '#1f1f1f'

    # Configure colorscale based on metric
    if metric == "score":
        colorscale = [
            [0.0, '#d73027'],
            [0.5, '#fee08b'],
            [1.0, '#1a9850']
        ]
        zmid = 0.5
        zmin = 0.0
        zmax = 1.0
        colorbar_config = dict(
            title=dict(text="Score", font=dict(color=text_color)),
            tickvals=[0, 1.0],
            ticktext=["0.0 (Fail)", "1.0 (Pass)"],
            tickfont=dict(color=text_color)
        )
    elif metric == "tokens":
        colorscale = "Blues"
        zmid = None
        zmin = None
        zmax = None
        colorbar_config = dict(
            title=dict(text="Tokens", font=dict(color=text_color)),
            tickfont=dict(color=text_color)
        )
    elif metric == "time":
        colorscale = [
            [0.0, '#1a9850'],
            [0.5, '#fee08b'],
            [1.0, '#d73027']
        ]
        zmid = None
        zmin = None
        zmax = None
        colorbar_config = dict(
            title=dict(text="Time (s)", font=dict(color=text_color)),
            tickfont=dict(color=text_color)
        )
    elif metric == "error":
        colorscale = [
            [0.0, '#1a9850'],
            [0.5, '#fee08b'],
            [1.0, '#d73027']
        ]
        zmid = 0.5
        zmin = 0.0
        zmax = 1.0
        colorbar_config = dict(
            title=dict(text="Error Rate", font=dict(color=text_color)),
            tickvals=[0, 0.5, 1.0],
            ticktext=["0% (None)", "50%", "100% (All)"],
            tickfont=dict(color=text_color)
        )

    fig = go.Figure(data=go.Heatmap(
        z=_to_serializable(matrix),
        x=tasks,
        y=agents,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorscale=colorscale,
        zmid=zmid,
        zmin=zmin,
        zmax=zmax,
        colorbar=colorbar_config,
        showscale=True
    ))

    width = max(base_width, len(tasks) * min_cell_width + 250)
    height = max(500, len(agents) * min_cell_height + 250)

    max_score_width = 0
    max_total_width = 0
    if agent_stats:
        for stats in agent_stats.values():
            score_str = f"{stats['total_score']:.1f}"
            max_score_width = max(max_score_width, len(score_str))
            max_total_width = max(max_total_width, len(str(stats['total'])))
    if task_stats:
        for stats in task_stats.values():
            score_str = f"{stats['total_score']:.1f}"
            max_score_width = max(max_score_width, len(score_str))
            max_total_width = max(max_total_width, len(str(stats['total'])))

    task_labels_display = []
    for task in tasks:
        if task_stats and task in task_stats:
            stats = task_stats[task]
            label = (f"{truncate_label(task, task_label_truncate) if task_label_truncate else task}\n"
                    f"({stats['total_score']:>{max_score_width}.1f}/{stats['total']:>{max_total_width}}, "
                    f"{stats['percentage']:>5.1f}%)")
        else:
            label = truncate_label(task, task_label_truncate) if task_label_truncate else task
        task_labels_display.append(label)

    agent_labels_display = []
    for agent in agents:
        if agent_stats and agent in agent_stats:
            stats = agent_stats[agent]
            label = (f"{agent} "
                    f"({stats['total_score']:>{max_score_width}.1f}/{stats['total']:>{max_total_width}}, "
                    f"{stats['percentage']:>5.1f}%)")
        else:
            label = agent
        agent_labels_display.append(label)

    # Theme colors were already detected at the start of this function
    # Calculate dynamic margins based on label lengths
    # Agent labels (y-axis) appear on the left - need left margin
    max_agent_label_len = max(len(label) for label in agent_labels_display) if agent_labels_display else 10
    # Task labels (x-axis) appear at the bottom at -45 degrees
    max_task_label_len = max(len(label) for label in task_labels_display) if task_labels_display else 10

    # Estimate pixels needed: ~6px per character for the font sizes used
    # Agent labels use font size 10, task labels use font size 9
    left_margin = max(100, int(max_agent_label_len * 6) + 50)
    # Task labels are rotated -45 degrees, so they extend both down and to the side
    # The vertical extent is roughly label_length * char_width * sin(45) ≈ 0.7 * length * 6
    bottom_margin = max(100, int(max_task_label_len * 4.5) + 50)

    # Apply layout with explicit colors (no template - we control all colors)
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18, color=text_color)
        ),
        xaxis=dict(
            title="Tasks",
            tickangle=-45,
            tickfont=dict(size=9, color=text_color),
            side='bottom',
            showgrid=False,
            type='category',
            categoryorder='array',
            categoryarray=tasks,
            tickvals=tasks,
            ticktext=task_labels_display,
            title_font=dict(color=text_color)
        ),
        yaxis=dict(
            title="Agents",
            tickfont=dict(size=10, color=text_color),
            showgrid=False,
            type='category',
            categoryorder='array',
            categoryarray=agents,
            tickvals=agents,
            ticktext=agent_labels_display,
            title_font=dict(color=text_color)
        ),
        height=height,
        width=width,
        margin=dict(l=left_margin, r=100, t=80, b=bottom_margin),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # Add gridlines
    shapes = []
    for i in range(len(tasks) + 1):
        shapes.append(dict(
            type='line',
            x0=i - 0.5,
            x1=i - 0.5,
            y0=-0.5,
            y1=len(agents) - 0.5,
            line=dict(color=grid_color, width=1),
            layer='above'
        ))
    for i in range(len(agents) + 1):
        shapes.append(dict(
            type='line',
            x0=-0.5,
            x1=len(tasks) - 0.5,
            y0=i - 0.5,
            y1=i - 0.5,
            line=dict(color=grid_color, width=1),
            layer='above'
        ))
    fig.update_layout(shapes=shapes)

    return fig, {"width": width, "height": height}


def calculate_summary_stats(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, Any]:
    """Calculate summary statistics for the heatmap."""
    all_scores = []
    pass_count = 0
    fail_count = 0
    total_datapoints = 0

    for agent_results in results.values():
        for datapoints in agent_results.values():
            for dp in datapoints:
                score = dp["score"]
                all_scores.append(score)
                total_datapoints += 1
                if score == 1.0:
                    pass_count += 1
                elif score == 0.0:
                    fail_count += 1

    stats = {
        "total_datapoints": total_datapoints,
        "avg_score": np.mean(all_scores) if all_scores else 0.0,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": (pass_count / total_datapoints * 100) if total_datapoints > 0 else 0.0
    }

    return stats


def render_summary_stats(stats: Dict[str, Any]):
    """Render summary statistics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Datapoints", stats["total_datapoints"])
    with col2:
        st.metric("Average Score", f"{stats['avg_score']:.2f}")
    with col3:
        st.metric("Pass Count", stats["pass_count"])
    with col4:
        st.metric("Pass Rate", f"{stats['pass_rate']:.1f}%")


def render_datapoint_details(runs: List[Dict], agent_id: str, task_id: str):
    """Render details for all datapoints matching agent/task."""
    datapoints_found = []

    for run in runs:
        if run["agent_id"] == agent_id:
            for dp in run["datapoints"]:
                if dp["task_id"] == task_id:
                    datapoints_found.append(dp)

    if not datapoints_found:
        st.warning(f"No datapoints found for {agent_id} × {task_id}")
        return

    st.subheader(f"Datapoints: {agent_id} × {task_id}")
    st.write(f"Found {len(datapoints_found)} datapoint(s)")

    benchmark_id = st.session_state.get("heatmap_benchmark")
    dataset_id = st.session_state.get("heatmap_dataset")

    for i, dp in enumerate(datapoints_found, 1):
        with st.expander(f"Datapoint [{i}]: {dp['datapoint_id']}", expanded=(i == 1)):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Run ID:** {dp['run_id']}")
                st.write(f"**Score:** {dp['score']:.2f}")
                st.write(f"**Time:** {dp['time']:.2f}s" if dp['time'] != -1 else "**Time:** N/A")

            with col2:
                st.write(f"**Tokens:** {dp['tokens']}" if dp['tokens'] != -1 else "**Tokens:** N/A")
                st.write(f"**Error:** {dp['error']}")

            st.markdown("---")

            if st.button(f"View Full Details", key=f"view_dp_{dp['datapoint_id']}"):
                st.session_state.selected_entry = dp['datapoint_id']
                st.session_state.selected_table = "datapoint"
                st.session_state.viewing_detail = True
                st.session_state.viewing_datapoints = False
                st.rerun()


def render_datapoint_heatmap_tab():
    """Render the Datapoint Heatmap tab."""
    # Import filter and entity application functions from heatmap module
    from ui.components.heatmap import (
        render_heatmap_filters,
        apply_agent_filters,
        apply_task_filters,
        apply_run_filters,
        apply_datapoint_filters
    )
    
    # Check if we're viewing a detail page
    if st.session_state.get("viewing_detail") and st.session_state.get("selected_entry"):
        from ui.components import detail_view

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Back to Datapoints"):
                st.session_state.viewing_detail = False
                st.session_state.viewing_datapoints = True
                st.rerun()

        with col2:
            st.title(f"📄 {st.session_state.selected_table.capitalize()} Details")

        st.markdown("---")
        detail_view.render_detail_panel(
            st.session_state.selected_table,
            st.session_state.selected_entry
        )
        return

    # Check if we're viewing datapoints from a clicked cell
    if st.session_state.get("viewing_datapoints"):
        required_keys = [
            "heatmap_benchmarks_for_view",
            "heatmap_datasets_for_view",
            "heatmap_agent",
            "heatmap_task",
        ]
        missing = [key for key in required_keys
                   if key not in st.session_state or st.session_state.get(key) is None]

        if missing:
            st.warning("Datapoint context missing. Returning to heatmap view.")
            st.session_state.viewing_datapoints = False
            st.rerun()
            return

        benchmark_ids = st.session_state.get("heatmap_benchmarks_for_view")
        dataset_ids = st.session_state.get("heatmap_datasets_for_view")
        agent = st.session_state.get("heatmap_agent")
        task = st.session_state.get("heatmap_task")
        
        # Re-fetch runs data from all benchmark/dataset combinations
        repo_root = Path(__file__).parent.parent.parent.parent
        all_runs = []
        for benchmark_id in benchmark_ids:
            for dataset_id in dataset_ids:
                all_runs.extend(load_run_jsons_for_filters(repo_root, benchmark_id, dataset_id))
        runs = all_runs

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Back to Heatmap"):
                st.session_state.viewing_datapoints = False
                st.rerun()

        with col2:
            st.title(f"📊 Datapoints")

        st.markdown("**Context:**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Benchmark(s):**")
            benchmark_label = benchmark_ids[0] if len(benchmark_ids) == 1 else f"{len(benchmark_ids)} benchmarks"
            if len(benchmark_ids) == 1:
                if st.button(f"📚 {benchmark_label}", key="nav_benchmark", use_container_width=True):
                    st.session_state.selected_entry = benchmark_ids[0]
                    st.session_state.selected_table = "benchmark"
                    st.session_state.viewing_detail = True
                    st.session_state.viewing_datapoints = False
                    st.rerun()
            else:
                st.write(f"📚 {benchmark_label}")

        with col2:
            st.markdown("**Dataset(s):**")
            dataset_label = dataset_ids[0] if len(dataset_ids) == 1 else f"{len(dataset_ids)} datasets"
            if len(dataset_ids) == 1:
                if st.button(f"📊 {dataset_label}", key="nav_dataset", use_container_width=True):
                    st.session_state.selected_entry = dataset_ids[0]
                    st.session_state.selected_table = "dataset"
                    st.session_state.viewing_detail = True
                    st.session_state.viewing_datapoints = False
                    st.rerun()
            else:
                st.write(f"📊 {dataset_label}")

        with col3:
            st.markdown("**Agent:**")
            if st.button(f"🤖 {agent}", key="nav_agent", use_container_width=True):
                st.session_state.selected_entry = agent
                st.session_state.selected_table = "agent"
                st.session_state.viewing_detail = True
                st.session_state.viewing_datapoints = False
                st.rerun()

        with col4:
            st.markdown("**Task:**")
            if st.button(f"📝 {task}", key="nav_task", use_container_width=True):
                st.session_state.selected_entry = task
                st.session_state.selected_table = "task"
                st.session_state.viewing_detail = True
                st.session_state.viewing_datapoints = False
                st.rerun()

        st.markdown("---")
        render_datapoint_details(runs, agent, task)
        return

    # Normal heatmap interface
    st.markdown("### 📊 Datapoint Performance Heatmap")
    st.markdown("Visualize agent performance across tasks with clickable cells")

    if not st.session_state.db_path.exists():
        st.error(f"Database not found at: {st.session_state.db_path}")
        return

    selection, filter_specs = render_heatmap_filters()

    if selection is None or selection[0] is None or selection[1] is None:
        st.info("Select a benchmark and dataset to view heatmap")
        return

    st.markdown("---")

    st.subheader("📈 Display Metric")
    metric_option = st.selectbox(
        "Select metric to display on heatmap",
        options=["Score", "Tokens", "Time", "Error"],
        index=0,
        key="heatmap_metric_selector",
        help="Choose which datapoint metric to visualize in the heatmap"
    )

    st.markdown("---")

    benchmark_ids, dataset_ids = selection

    try:
        repo_root = Path(__file__).parent.parent.parent.parent

        # Load runs from all benchmark/dataset combinations
        all_runs = []
        with st.spinner(f"Loading data from {len(benchmark_ids)} benchmark(s), {len(dataset_ids)} dataset(s)..."):
            for benchmark_id in benchmark_ids:
                for dataset_id in dataset_ids:
                    runs = load_run_jsons_for_filters(repo_root, benchmark_id, dataset_id)
                    all_runs.extend(runs)

        runs = all_runs

        if not runs:
            st.warning(f"No runs found for selected benchmark/dataset combinations")
            return

        # Apply entity filters if specified
        if filter_specs:
            conn = db_manager.connect_db(st.session_state.db_path)

            if "agent" in filter_specs:
                filter_spec, manual_selection = filter_specs["agent"]
                if filter_spec and manual_selection:
                    filtered_agent_ids = apply_agent_filters(conn, filter_spec)
                    filtered_agent_ids = [aid for aid in filtered_agent_ids if aid in manual_selection]
                elif filter_spec:
                    filtered_agent_ids = apply_agent_filters(conn, filter_spec)
                elif manual_selection:
                    filtered_agent_ids = manual_selection
                else:
                    filtered_agent_ids = None

                if filtered_agent_ids is not None:
                    runs = [r for r in runs if r["agent_id"] in filtered_agent_ids]

            if "task" in filter_specs:
                filter_spec, manual_selection = filter_specs["task"]
                filtered_task_ids = apply_task_filters(conn, filter_spec) if filter_spec else manual_selection

                if filtered_task_ids:
                    for run in runs:
                        run["datapoints"] = [dp for dp in run.get("datapoints", []) if dp["task_id"] in filtered_task_ids]
                    runs = [r for r in runs if r.get("datapoints")]

            if "run" in filter_specs:
                filter_spec, manual_selection = filter_specs["run"]
                filtered_run_ids = apply_run_filters(conn, filter_spec) if filter_spec else manual_selection

                if filtered_run_ids:
                    runs = [r for r in runs if r["run_id"] in filtered_run_ids]

            if "datapoint" in filter_specs:
                filter_spec, manual_selection = filter_specs["datapoint"]
                filtered_datapoint_ids = apply_datapoint_filters(conn, filter_spec) if filter_spec else manual_selection

                if filtered_datapoint_ids:
                    for run in runs:
                        run["datapoints"] = [dp for dp in run.get("datapoints", []) if dp["datapoint_id"] in filtered_datapoint_ids]
                    runs = [r for r in runs if r.get("datapoints")]

            conn.close()

            if not runs:
                st.warning("No runs found matching filters")
                return

        results = aggregate_results(runs)

        if not results:
            st.warning("No datapoints found in runs")
            return

        # Apply completeness filters (toggles are in the Agent/Task filter dropdowns)
        complete_agents_only = st.session_state.get("heatmap_agent_complete_only", False)
        complete_tasks_only = st.session_state.get("heatmap_task_complete_only", False)

        if complete_agents_only or complete_tasks_only:
            # Get all unique agents and tasks from results
            all_agents = set(results.keys())
            all_tasks = set()
            for agent_tasks in results.values():
                all_tasks.update(agent_tasks.keys())

            # Iteratively filter until stable (since filtering one affects the other)
            prev_agents, prev_tasks = None, None
            while (prev_agents, prev_tasks) != (all_agents, all_tasks):
                prev_agents, prev_tasks = all_agents.copy(), all_tasks.copy()

                if complete_agents_only and all_tasks:
                    # Keep only agents that have results for ALL current tasks
                    all_agents = {
                        agent for agent in all_agents
                        if all(task in results.get(agent, {}) for task in all_tasks)
                    }

                if complete_tasks_only and all_agents:
                    # Keep only tasks that have results from ALL current agents
                    all_tasks = {
                        task for task in all_tasks
                        if all(task in results.get(agent, {}) for agent in all_agents)
                    }

            # Filter results to only include complete agents/tasks
            filtered_results = {}
            for agent in all_agents:
                if agent in results:
                    filtered_results[agent] = {
                        task: datapoints
                        for task, datapoints in results[agent].items()
                        if task in all_tasks
                    }

            results = filtered_results

            if not results:
                st.warning("No complete agent-task combinations found with current filters")
                return

        # Load task metadata from database for all selected benchmark/dataset combinations
        conn = db_manager.connect_db(st.session_state.db_path)
        cursor = conn.cursor()
        task_metadata = {}
        for benchmark_id in benchmark_ids:
            for dataset_id in dataset_ids:
                cursor.execute(
                    "SELECT task_id, metadata FROM tasks WHERE benchmark_id = ? AND dataset_id = ?",
                    (benchmark_id, dataset_id)
                )
                for row in cursor.fetchall():
                    task_id = row[0]
                    metadata_json = row[1]
                    if metadata_json and task_id not in task_metadata:
                        task_metadata[task_id] = {"metadata": json.loads(metadata_json)}
        conn.close()

        metric_key = metric_option.lower()
        matrix, agents, tasks, agent_stats, task_stats, cell_metadata = create_heatmap_data(results, task_metadata, metric_key)

        # Generate dynamic title based on selection
        if len(benchmark_ids) == 1 and len(dataset_ids) == 1:
            title = f"{benchmark_ids[0]} / {dataset_ids[0]}"
        else:
            title = f"{len(benchmark_ids)} benchmark(s) / {len(dataset_ids)} dataset(s)"

        stats = calculate_summary_stats(results)
        cols_stats = st.columns([3, 1])
        with cols_stats[0]:
            render_summary_stats(stats)
        with cols_stats[1]:
            # Calculate cell sizes to prevent axis label overlap
            # Use smaller cells when there are many items
            num_tasks = len(tasks)
            num_agents = len(agents)

            # Target: cells small enough that axis labels don't overlap
            # Axis labels need ~10px per character, and labels are angled at -45 degrees
            # Task labels appear below, agent labels on the left
            full_cell_width = max(20, min(60, 1800 // max(1, num_tasks)))
            full_cell_height = max(20, min(60, 1200 // max(1, num_agents)))

            full_fig, _ = create_plotly_heatmap(
                matrix,
                agents,
                tasks,
                title,
                min_cell_width=full_cell_width,
                min_cell_height=full_cell_height,
                task_label_truncate=None,
                base_width=1200,
                agent_stats=agent_stats,
                task_stats=task_stats,
                cell_metadata=cell_metadata,
                metric=metric_key
            )
            render_full_heatmap_button(full_fig)

        st.markdown("---")
        preview_cell_width = max(12, int(800 / max(1, len(tasks))))
        preview_cell_height = max(12, int(400 / max(1, len(agents))))
        fig, layout_dims = create_plotly_heatmap(
            matrix,
            agents,
            tasks,
            title,
            min_cell_width=preview_cell_width,
            min_cell_height=preview_cell_height,
            task_label_truncate=18,
            base_width=800,
            agent_stats=agent_stats,
            task_stats=task_stats,
            cell_metadata=cell_metadata,
            metric=metric_key
        )

        selected_points: List[Dict[str, Any]] = []
        if plotly_events:
            component_height = layout_dims["height"]
            component_width = layout_dims["width"]
            selected_points = plotly_events(
                fig,
                click_event=True,
                hover_event=False,
                select_event=False,
                key="heatmap_chart",
                override_height=component_height,
                override_width=component_width,
            ) or []
        else:
            st.info(
                "Install `streamlit-plotly-events` (see requirements) to enable"
                " click-to-select on the heatmap."
            )
            st.plotly_chart(
                fig,
                use_container_width=False,
                key="heatmap_chart",
                config={'responsive': False}
            )

        if selected_points:
            point = selected_points[0]
            clicked_task = point.get("x")
            clicked_agent = point.get("y")
            if clicked_agent in agents and clicked_task in tasks:
                now = time.time()
                last_click = st.session_state.get("heatmap_last_click")
                st.session_state["datapoint_agent_selector"] = clicked_agent
                st.session_state["datapoint_task_selector"] = clicked_task
                st.session_state["heatmap_last_click"] = {
                    "agent": clicked_agent,
                    "task": clicked_task,
                    "timestamp": now
                }
                if (
                    last_click
                    and last_click.get("agent") == clicked_agent
                    and last_click.get("task") == clicked_task
                    and now - last_click.get("timestamp", 0) < 0.9
                ):
                    open_datapoints_view(
                        clicked_agent,
                        clicked_task,
                        runs,
                        benchmark_ids,
                        dataset_ids,
                        results
                    )

        st.markdown("---")
        st.subheader("🔍 View Datapoints")
        st.markdown("Select an agent and task combination to view detailed datapoints:")

        col1, col2, col3 = st.columns([2, 2, 1])

        if agents:
            current_agent = st.session_state.get("datapoint_agent_selector")
            if current_agent not in agents:
                st.session_state["datapoint_agent_selector"] = agents[0]
        if tasks:
            current_task = st.session_state.get("datapoint_task_selector")
            if current_task not in tasks:
                st.session_state["datapoint_task_selector"] = tasks[0]

        with col1:
            selected_agent = st.selectbox(
                "Agent",
                options=agents,
                key="datapoint_agent_selector"
            )

        with col2:
            selected_task = st.selectbox(
                "Task",
                options=tasks,
                key="datapoint_task_selector"
            )

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📊 View Datapoints", type="primary", use_container_width=True):
                open_datapoints_view(
                    selected_agent,
                    selected_task,
                    runs,
                    benchmark_ids,
                    dataset_ids,
                    results
                )

    except Exception as e:
        st.error(f"Error rendering heatmap: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
