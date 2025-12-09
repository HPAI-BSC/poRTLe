"""
Scoreboard Component for poRTLe UI

Displays ranked leaderboards for agents and tasks with medals, color-coded rows,
and sortable metrics. Uses the same filtering mechanisms as the heatmap.
"""

import streamlit as st
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ui.components.heatmap import (
    render_heatmap_filters,
    apply_agent_filters,
    apply_task_filters,
    apply_run_filters,
    apply_datapoint_filters,
)
from ui.components.datapoint_heatmap import (
    load_run_jsons_for_filters,
    aggregate_results,
)
from ui.utils.db_manager import connect_db


# =============================================================================
# Styling Helpers
# =============================================================================

def get_rank_display(rank: int) -> str:
    """Return rank display with medals for top 3."""
    if rank == 1:
        return "🥇 1"
    elif rank == 2:
        return "🥈 2"
    elif rank == 3:
        return "🥉 3"
    else:
        return str(rank)


def get_row_class(rank: int, total: int) -> str:
    """Return CSS class name for row based on rank position."""
    if rank == 1:
        return "scoreboard-gold"
    elif rank == 2:
        return "scoreboard-silver"
    elif rank == 3:
        return "scoreboard-bronze"
    elif rank <= total * 0.25:
        return "scoreboard-top-quarter"
    elif rank <= total * 0.5:
        return "scoreboard-top-half"
    else:
        return "scoreboard-bottom-half"


def inject_scoreboard_css():
    """Inject custom CSS for scoreboard styling."""
    st.markdown("""
    <style>
    /* Medal rows */
    .scoreboard-gold {
        background: linear-gradient(90deg, rgba(255,215,0,0.2), rgba(255,215,0,0.05));
        border-left: 4px solid #FFD700;
        padding: 0.75rem 0.5rem;
        margin: 2px 0;
        border-radius: 4px;
    }

    .scoreboard-silver {
        background: linear-gradient(90deg, rgba(192,192,192,0.2), rgba(192,192,192,0.05));
        border-left: 4px solid #C0C0C0;
        padding: 0.75rem 0.5rem;
        margin: 2px 0;
        border-radius: 4px;
    }

    .scoreboard-bronze {
        background: linear-gradient(90deg, rgba(205,127,50,0.2), rgba(205,127,50,0.05));
        border-left: 4px solid #CD7F32;
        padding: 0.75rem 0.5rem;
        margin: 2px 0;
        border-radius: 4px;
    }

    /* Tier rows */
    .scoreboard-top-quarter {
        background-color: rgba(26, 152, 80, 0.08);
        padding: 0.75rem 0.5rem;
        margin: 2px 0;
        border-radius: 4px;
        border-left: 4px solid rgba(26, 152, 80, 0.3);
    }

    .scoreboard-top-half {
        background-color: rgba(254, 224, 139, 0.08);
        padding: 0.75rem 0.5rem;
        margin: 2px 0;
        border-radius: 4px;
        border-left: 4px solid rgba(254, 224, 139, 0.3);
    }

    .scoreboard-bottom-half {
        background-color: rgba(215, 48, 39, 0.05);
        padding: 0.75rem 0.5rem;
        margin: 2px 0;
        border-radius: 4px;
        border-left: 4px solid rgba(215, 48, 39, 0.2);
    }

    /* Header row */
    .scoreboard-header {
        background-color: rgba(255,255,255,0.08);
        font-weight: bold;
        border-bottom: 2px solid rgba(255,255,255,0.2);
        padding: 0.75rem 0.5rem;
        margin-bottom: 4px;
        border-radius: 4px 4px 0 0;
    }

    /* Score highlighting */
    .score-high { color: #2ecc71; font-weight: bold; }
    .score-medium { color: #f1c40f; }
    .score-low { color: #e74c3c; }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# Data Aggregation Functions
# =============================================================================

def calculate_agent_rankings(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> List[Dict[str, Any]]:
    """
    Calculate ranking statistics for each agent.

    The score is calculated as the average of per-task scores across ALL tasks
    in the filtered selection. If an agent hasn't run on a particular task,
    that task contributes a score of 0 to the agent's average.

    Args:
        results: Aggregated results from aggregate_results()
                 {agent_id: {task_id: [datapoint_dicts]}}

    Returns:
        List of agent stats dicts, sorted by avg_score descending
    """
    # First, collect all unique tasks across all agents
    all_tasks = set()
    for agent_id, task_results in results.items():
        all_tasks.update(task_results.keys())
    all_tasks = list(all_tasks)
    total_task_count = len(all_tasks)

    agent_stats = []

    for agent_id, task_results in results.items():
        all_tokens = []
        all_times = []
        error_count = 0
        total_datapoints = 0
        task_pass_count = 0
        tasks_run_count = 0

        # Calculate per-task average scores
        task_avg_scores = []

        for task_id in all_tasks:
            if task_id in task_results:
                # Agent has run this task
                datapoints = task_results[task_id]
                task_scores = []
                tasks_run_count += 1

                for dp in datapoints:
                    total_datapoints += 1

                    # Score
                    score = dp.get("score")
                    if score is not None:
                        task_scores.append(float(score))

                    # Tokens (exclude -1 and None)
                    tokens = dp.get("tokens")
                    if tokens is not None and tokens != -1:
                        all_tokens.append(float(tokens))

                    # Time (exclude -1 and None)
                    time_val = dp.get("time")
                    if time_val is not None and time_val != -1:
                        all_times.append(float(time_val))

                    # Error check - any non-empty, non-zero error
                    error_val = dp.get("error")
                    if error_val and error_val not in [0, 1, "0", "1", "", None]:
                        error_count += 1

                # Calculate task average score
                task_avg = float(np.mean(task_scores)) if task_scores else 0.0
                task_avg_scores.append(task_avg)

                # Check if task passed (avg score = 1.0 for this task)
                if task_scores and np.mean(task_scores) == 1.0:
                    task_pass_count += 1
            else:
                # Agent has NOT run this task - count as 0
                task_avg_scores.append(0.0)

        # Average score is the mean of all per-task averages (including 0s for missing tasks)
        avg_score = float(np.mean(task_avg_scores)) if task_avg_scores else 0.0

        agent_stats.append({
            "agent_id": agent_id,
            "avg_score": avg_score,
            "pass_rate": (task_pass_count / total_task_count * 100) if total_task_count > 0 else 0.0,
            "avg_tokens": float(np.mean(all_tokens)) if all_tokens else 0.0,
            "avg_time": float(np.mean(all_times)) if all_times else 0.0,
            "error_rate": (error_count / total_datapoints * 100) if total_datapoints > 0 else 0.0,
            "task_count": tasks_run_count,
            "total_tasks": total_task_count,
            "datapoint_count": total_datapoints
        })

    # Sort by avg_score descending
    agent_stats.sort(key=lambda x: x["avg_score"], reverse=True)

    return agent_stats


def calculate_task_rankings(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> List[Dict[str, Any]]:
    """
    Calculate ranking statistics for each task.

    The score is calculated as the average of per-agent scores across ALL agents
    in the filtered selection. If an agent hasn't run on a particular task,
    that agent contributes a score of 0 to the task's average.

    Args:
        results: Aggregated results {agent_id: {task_id: [datapoint_dicts]}}

    Returns:
        List of task stats dicts, sorted by avg_score descending
    """
    # Collect all unique agents
    all_agents = list(results.keys())
    total_agent_count = len(all_agents)

    # Reorganize by task: {task_id: {agent_id: [datapoints]}}
    task_data = {}

    for agent_id, task_results in results.items():
        for task_id, datapoints in task_results.items():
            if task_id not in task_data:
                task_data[task_id] = {}
            task_data[task_id][agent_id] = datapoints

    task_stats = []

    for task_id, agent_results in task_data.items():
        all_tokens = []
        all_times = []
        error_count = 0
        total_datapoints = 0
        agent_pass_count = 0
        agents_run_count = 0

        # Calculate per-agent average scores
        agent_avg_scores = []

        for agent_id in all_agents:
            if agent_id in agent_results:
                # This agent has run this task
                datapoints = agent_results[agent_id]
                agent_scores = []
                agents_run_count += 1

                for dp in datapoints:
                    total_datapoints += 1

                    # Score
                    score = dp.get("score")
                    if score is not None:
                        agent_scores.append(float(score))

                    # Tokens
                    tokens = dp.get("tokens")
                    if tokens is not None and tokens != -1:
                        all_tokens.append(float(tokens))

                    # Time
                    time_val = dp.get("time")
                    if time_val is not None and time_val != -1:
                        all_times.append(float(time_val))

                    # Error
                    error_val = dp.get("error")
                    if error_val and error_val not in [0, 1, "0", "1", "", None]:
                        error_count += 1

                # Calculate agent average score for this task
                agent_avg = float(np.mean(agent_scores)) if agent_scores else 0.0
                agent_avg_scores.append(agent_avg)

                # Check if this agent passed the task
                if agent_scores and np.mean(agent_scores) == 1.0:
                    agent_pass_count += 1
            else:
                # This agent has NOT run this task - count as 0
                agent_avg_scores.append(0.0)

        # Average score is the mean of all per-agent averages (including 0s for missing agents)
        avg_score = float(np.mean(agent_avg_scores)) if agent_avg_scores else 0.0

        task_stats.append({
            "task_id": task_id,
            "avg_score": avg_score,
            "pass_rate": (agent_pass_count / total_agent_count * 100) if total_agent_count > 0 else 0.0,
            "avg_tokens": float(np.mean(all_tokens)) if all_tokens else 0.0,
            "avg_time": float(np.mean(all_times)) if all_times else 0.0,
            "error_rate": (error_count / total_datapoints * 100) if total_datapoints > 0 else 0.0,
            "agent_count": agents_run_count,
            "total_agents": total_agent_count,
            "datapoint_count": total_datapoints
        })

    # Sort by avg_score descending
    task_stats.sort(key=lambda x: x["avg_score"], reverse=True)

    return task_stats


# =============================================================================
# Scoreboard Renderers
# =============================================================================

def render_scoreboard_header(is_agent_view: bool):
    """Render the header row for the scoreboard with tooltips."""
    cols = st.columns([0.6, 2.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7])

    # Define headers with tooltips for each view
    if is_agent_view:
        headers_with_tips = [
            ("Rank", None),
            ("Agent", None),
            ("Avg Score", "Average of per-task scores. Tasks not run count as 0."),
            ("Pass Rate", "Percentage of all tasks with a perfect score (1.0)."),
            ("Avg Tokens", "Average token usage across completed runs."),
            ("Avg Time", "Average execution time across completed runs."),
            ("Error Rate", "Percentage of datapoints with errors."),
            ("Tasks", "Tasks run / total tasks in selection."),
            ("DPs", "Total number of datapoints."),
        ]
    else:
        headers_with_tips = [
            ("Rank", None),
            ("Task", None),
            ("Avg Score", "Average of per-agent scores. Agents that haven't run count as 0."),
            ("Pass Rate", "Percentage of all agents with a perfect score (1.0)."),
            ("Avg Tokens", "Average token usage across completed runs."),
            ("Avg Time", "Average execution time across completed runs."),
            ("Error Rate", "Percentage of datapoints with errors."),
            ("Agents", "Agents run / total agents in selection."),
            ("DPs", "Total number of datapoints."),
        ]

    for col, (header, tooltip) in zip(cols, headers_with_tips):
        with col:
            if tooltip:
                st.markdown(f"**{header}**", help=tooltip)
            else:
                st.markdown(f"**{header}**")


def render_agent_scoreboard(agent_stats: List[Dict[str, Any]]):
    """Render the agent ranking scoreboard."""
    if not agent_stats:
        st.warning("No agent data available for the selected filters.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Agents", len(agent_stats))
    with col2:
        avg_score = np.mean([a["avg_score"] for a in agent_stats])
        st.metric("Avg Score", f"{avg_score:.3f}")
    with col3:
        top_agent = agent_stats[0]
        st.metric("Top Score", f"{top_agent['avg_score']:.3f}")
    with col4:
        # Truncate long names
        display_name = top_agent["agent_id"][:25] + "..." if len(top_agent["agent_id"]) > 25 else top_agent["agent_id"]
        st.metric("Top Agent", display_name)

    st.markdown("---")

    # Sort controls
    col1, col2 = st.columns([1, 3])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=["Avg Score", "Pass Rate", "Avg Tokens", "Avg Time", "Error Rate"],
            index=0,
            key="agent_scoreboard_sort"
        )
    with col2:
        sort_order = st.radio(
            "Order",
            options=["Descending", "Ascending"],
            horizontal=True,
            key="agent_scoreboard_order"
        )

    # Apply sorting
    sort_key_map = {
        "Avg Score": "avg_score",
        "Pass Rate": "pass_rate",
        "Avg Tokens": "avg_tokens",
        "Avg Time": "avg_time",
        "Error Rate": "error_rate"
    }
    sort_key = sort_key_map[sort_by]

    # Determine sort direction (for time/tokens/errors, lower is better when "descending")
    if sort_by in ["Avg Time", "Avg Tokens", "Error Rate"]:
        reverse = (sort_order == "Ascending")  # Lower is better, so ascending = best first
    else:
        reverse = (sort_order == "Descending")  # Higher is better

    sorted_stats = sorted(agent_stats, key=lambda x: x[sort_key], reverse=reverse)

    st.markdown("---")

    # Header row
    render_scoreboard_header(is_agent_view=True)

    # Data rows
    total = len(sorted_stats)
    for i, stats in enumerate(sorted_stats, 1):
        row_class = get_row_class(i, total)

        # Use markdown with HTML for styled rows
        cols = st.columns([0.6, 2.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7])

        with cols[0]:
            st.markdown(f"**{get_rank_display(i)}**")
        with cols[1]:
            st.markdown(f"`{stats['agent_id']}`")
        with cols[2]:
            score = stats['avg_score']
            if score >= 0.8:
                st.markdown(f"**:green[{score:.3f}]**")
            elif score >= 0.5:
                st.markdown(f"**:orange[{score:.3f}]**")
            else:
                st.markdown(f"**:red[{score:.3f}]**")
        with cols[3]:
            st.markdown(f"{stats['pass_rate']:.1f}%")
        with cols[4]:
            if stats['avg_tokens'] > 0:
                st.markdown(f"{stats['avg_tokens']:.0f}")
            else:
                st.markdown("N/A")
        with cols[5]:
            if stats['avg_time'] > 0:
                st.markdown(f"{stats['avg_time']:.2f}s")
            else:
                st.markdown("N/A")
        with cols[6]:
            error_rate = stats['error_rate']
            if error_rate == 0:
                st.markdown(":green[0.0%]")
            elif error_rate < 10:
                st.markdown(f":orange[{error_rate:.1f}%]")
            else:
                st.markdown(f":red[{error_rate:.1f}%]")
        with cols[7]:
            # Show tasks run vs total tasks
            tasks_run = stats['task_count']
            total_tasks = stats.get('total_tasks', tasks_run)
            if tasks_run < total_tasks:
                st.markdown(f":orange[{tasks_run}/{total_tasks}]")
            else:
                st.markdown(f"{tasks_run}/{total_tasks}")
        with cols[8]:
            st.markdown(str(stats['datapoint_count']))


def render_task_scoreboard(task_stats: List[Dict[str, Any]]):
    """Render the task ranking scoreboard."""
    if not task_stats:
        st.warning("No task data available for the selected filters.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tasks", len(task_stats))
    with col2:
        avg_score = np.mean([t["avg_score"] for t in task_stats])
        st.metric("Avg Score", f"{avg_score:.3f}")
    with col3:
        top_task = task_stats[0]
        st.metric("Top Score", f"{top_task['avg_score']:.3f}")
    with col4:
        display_name = top_task["task_id"][:25] + "..." if len(top_task["task_id"]) > 25 else top_task["task_id"]
        st.metric("Easiest Task", display_name)

    st.markdown("---")

    # Sort controls
    col1, col2 = st.columns([1, 3])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=["Avg Score", "Pass Rate", "Avg Tokens", "Avg Time", "Error Rate"],
            index=0,
            key="task_scoreboard_sort"
        )
    with col2:
        sort_order = st.radio(
            "Order",
            options=["Descending", "Ascending"],
            horizontal=True,
            key="task_scoreboard_order"
        )

    # Apply sorting
    sort_key_map = {
        "Avg Score": "avg_score",
        "Pass Rate": "pass_rate",
        "Avg Tokens": "avg_tokens",
        "Avg Time": "avg_time",
        "Error Rate": "error_rate"
    }
    sort_key = sort_key_map[sort_by]

    if sort_by in ["Avg Time", "Avg Tokens", "Error Rate"]:
        reverse = (sort_order == "Ascending")
    else:
        reverse = (sort_order == "Descending")

    sorted_stats = sorted(task_stats, key=lambda x: x[sort_key], reverse=reverse)

    st.markdown("---")

    # Header row
    render_scoreboard_header(is_agent_view=False)

    # Data rows
    total = len(sorted_stats)
    for i, stats in enumerate(sorted_stats, 1):
        cols = st.columns([0.6, 2.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.7])

        with cols[0]:
            st.markdown(f"**{get_rank_display(i)}**")
        with cols[1]:
            st.markdown(f"`{stats['task_id']}`")
        with cols[2]:
            score = stats['avg_score']
            if score >= 0.8:
                st.markdown(f"**:green[{score:.3f}]**")
            elif score >= 0.5:
                st.markdown(f"**:orange[{score:.3f}]**")
            else:
                st.markdown(f"**:red[{score:.3f}]**")
        with cols[3]:
            st.markdown(f"{stats['pass_rate']:.1f}%")
        with cols[4]:
            if stats['avg_tokens'] > 0:
                st.markdown(f"{stats['avg_tokens']:.0f}")
            else:
                st.markdown("N/A")
        with cols[5]:
            if stats['avg_time'] > 0:
                st.markdown(f"{stats['avg_time']:.2f}s")
            else:
                st.markdown("N/A")
        with cols[6]:
            error_rate = stats['error_rate']
            if error_rate == 0:
                st.markdown(":green[0.0%]")
            elif error_rate < 10:
                st.markdown(f":orange[{error_rate:.1f}%]")
            else:
                st.markdown(f":red[{error_rate:.1f}%]")
        with cols[7]:
            # Show agents run vs total agents
            agents_run = stats['agent_count']
            total_agents = stats.get('total_agents', agents_run)
            if agents_run < total_agents:
                st.markdown(f":orange[{agents_run}/{total_agents}]")
            else:
                st.markdown(f"{agents_run}/{total_agents}")
        with cols[8]:
            st.markdown(str(stats['datapoint_count']))


# =============================================================================
# Main Render Function
# =============================================================================

def render():
    """Main render function for scoreboard page."""
    # Inject custom CSS
    inject_scoreboard_css()

    st.title("🏆 Scoreboard")
    st.markdown("Ranked leaderboards for agents and tasks")

    # Database check
    if not st.session_state.db_path.exists():
        st.error(f"Database not found at: {st.session_state.db_path}")
        st.info("Run `build_datatable.py` to create the database.")
        return

    # Get repo root for data loading
    repo_root = Path(__file__).parent.parent.parent.parent
    db_path = st.session_state.db_path

    # Render filters (reused from heatmap)
    selection, filter_specs = render_heatmap_filters()

    if selection is None or selection[0] is None or selection[1] is None:
        st.info("Select a benchmark and dataset above to view the scoreboard.")
        return

    benchmark_ids, dataset_ids = selection

    # Load run data from all benchmark/dataset combinations
    runs = []
    with st.spinner(f"Loading data from {len(benchmark_ids)} benchmark(s), {len(dataset_ids)} dataset(s)..."):
        for benchmark_id in benchmark_ids:
            for dataset_id in dataset_ids:
                runs.extend(load_run_jsons_for_filters(repo_root, benchmark_id, dataset_id))

    if not runs:
        st.warning(f"No run data found for selected benchmark/dataset combinations")
        return

    # Apply filters if specified
    if filter_specs:
        conn = connect_db(db_path)

        # Filter agents
        if "agent" in filter_specs:
            filter_spec, manual_selection = filter_specs["agent"]
            if filter_spec and manual_selection:
                # Both filter spec and manual selection - intersect results
                filtered_agent_ids = apply_agent_filters(conn, filter_spec)
                filtered_agent_ids = [aid for aid in filtered_agent_ids if aid in manual_selection]
            elif filter_spec:
                # Only filter spec
                filtered_agent_ids = apply_agent_filters(conn, filter_spec)
            elif manual_selection:
                # Only manual selection
                filtered_agent_ids = manual_selection
            else:
                filtered_agent_ids = None

            if filtered_agent_ids is not None:
                runs = [r for r in runs if r["agent_id"] in filtered_agent_ids]

        # Filter tasks
        if "task" in filter_specs:
            filter_spec, manual_selection = filter_specs["task"]
            if filter_spec and manual_selection:
                filtered_task_ids = apply_task_filters(conn, filter_spec)
                filtered_task_ids = [tid for tid in filtered_task_ids if tid in manual_selection]
            elif filter_spec:
                filtered_task_ids = apply_task_filters(conn, filter_spec)
            elif manual_selection:
                filtered_task_ids = manual_selection
            else:
                filtered_task_ids = None

            if filtered_task_ids is not None:
                for run in runs:
                    run["datapoints"] = [
                        dp for dp in run.get("datapoints", [])
                        if dp["task_id"] in filtered_task_ids
                    ]
                runs = [r for r in runs if r.get("datapoints")]

        # Filter runs
        if "run" in filter_specs:
            filter_spec, manual_selection = filter_specs["run"]
            if filter_spec and manual_selection:
                filtered_run_ids = apply_run_filters(conn, filter_spec)
                filtered_run_ids = [rid for rid in filtered_run_ids if rid in manual_selection]
            elif filter_spec:
                filtered_run_ids = apply_run_filters(conn, filter_spec)
            elif manual_selection:
                filtered_run_ids = manual_selection
            else:
                filtered_run_ids = None

            if filtered_run_ids is not None:
                runs = [r for r in runs if r["run_id"] in filtered_run_ids]

        # Filter datapoints
        if "datapoint" in filter_specs:
            filter_spec, manual_selection = filter_specs["datapoint"]
            if filter_spec and manual_selection:
                filtered_datapoint_ids = apply_datapoint_filters(conn, filter_spec)
                filtered_datapoint_ids = [did for did in filtered_datapoint_ids if did in manual_selection]
            elif filter_spec:
                filtered_datapoint_ids = apply_datapoint_filters(conn, filter_spec)
            elif manual_selection:
                filtered_datapoint_ids = manual_selection
            else:
                filtered_datapoint_ids = None

            if filtered_datapoint_ids is not None:
                for run in runs:
                    run["datapoints"] = [
                        dp for dp in run.get("datapoints", [])
                        if dp["datapoint_id"] in filtered_datapoint_ids
                    ]
                runs = [r for r in runs if r.get("datapoints")]

        conn.close()

    # Remove runs with no datapoints after filtering
    runs = [r for r in runs if r.get("datapoints")]

    if not runs:
        st.warning("No data matches the selected filters.")
        return

    # Aggregate results
    results = aggregate_results(runs)

    if not results:
        st.warning("No results to display after aggregation.")
        return

    # Apply completeness filters (toggles are in the Agent/Task filter expanders)
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
            st.warning("No complete agent-task combinations found with current filters.")
            return

    st.markdown("---")

    # View toggle
    view_options = ["🤖 Agent Rankings", "📝 Task Rankings"]

    # Initialize session state for view
    if "scoreboard_view" not in st.session_state:
        st.session_state.scoreboard_view = "agents"

    selected_view = st.radio(
        "View",
        options=view_options,
        horizontal=True,
        key="scoreboard_view_selector",
        label_visibility="collapsed"
    )

    is_agent_view = "Agent" in selected_view

    st.markdown("---")

    # Calculate and render appropriate scoreboard
    if is_agent_view:
        agent_stats = calculate_agent_rankings(results)
        render_agent_scoreboard(agent_stats)
    else:
        task_stats = calculate_task_rankings(results)
        render_task_scoreboard(task_stats)


if __name__ == "__main__":
    # For testing component standalone
    render()
