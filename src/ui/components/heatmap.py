"""
Heatmap Component for poRTLe UI

Main entry point for the Plots page. This module:
1. Provides the render() function that creates the tabbed plot interface
2. Contains shared filter functions used by multiple plot components
3. Imports specialized plot renderers from submodules

Plot Components:
- datapoint_heatmap.py: Interactive heatmap for agent/task performance
- agent_metrics.py: Bar charts and metrics table for agent performance analysis
"""

import streamlit as st
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ui.utils import db_manager
from ui.utils.filter_engine import FilterSpec


# =============================================================================
# Shared Filter Functions
# =============================================================================

def render_generic_column_filters(
    entity_type: str,
    entity_columns: List[str],
    session_key_prefix: str,
    available_items: Optional[List[str]] = None,
    show_completeness_toggle: bool = False,
    available_tags: Optional[List[str]] = None
):
    """
    Generic function to render column filters for any entity type.
    Supports filtering by entity columns and metadata.

    Args:
        entity_type: Type of entity (e.g., "agent", "task", "run", "datapoint")
        entity_columns: List of column names for this entity
        session_key_prefix: Prefix for session state keys (e.g., "heatmap_agent")
        available_items: Optional list of available items for manual selection (e.g., agent IDs)
        show_completeness_toggle: If True, show a toggle for "complete only" filtering
        available_tags: Optional list of existing metadata keys/tags for this entity

    Returns:
        FilterSpec object if filters are defined, None otherwise
    """
    entity_display = entity_type.capitalize()

    with st.expander(f"🎯 {entity_display} Filters", expanded=False):
        st.markdown(
            f"Build filters for {entity_type}s using columns and metadata. "
            f"Filter by {entity_type} properties or metadata fields."
        )

        # Completeness toggle for agents/tasks
        if show_completeness_toggle:
            completeness_key = f"{session_key_prefix}_complete_only"
            if entity_type == "agent":
                toggle_label = "Complete agents only"
                toggle_help = "Only show agents that have results for ALL displayed tasks"
            else:  # task
                toggle_label = "Complete tasks only"
                toggle_help = "Only show tasks that have results from ALL displayed agents"

            st.toggle(
                toggle_label,
                value=st.session_state.get(completeness_key, False),
                key=completeness_key,
                help=toggle_help
            )
            st.markdown("---")

        # Manual selection section (only for entity types with available_items)
        if available_items:
            st.markdown(f"#### Quick Selection")
            st.caption(f"Manually select specific {entity_type}s to include")

            manual_selection_key = f"{session_key_prefix}_manual_selection"
            if manual_selection_key not in st.session_state:
                st.session_state[manual_selection_key] = []

            selected_items = st.multiselect(
                f"Select {entity_display}s",
                options=available_items,
                default=st.session_state[manual_selection_key],
                key=f"{session_key_prefix}_multiselect",
                help=f"Choose specific {entity_type}s to include in the heatmap"
            )

            st.session_state[manual_selection_key] = selected_items
            st.markdown("---")

        # Initialize session state for filters
        field_filters_key = f"{session_key_prefix}_field_filters"
        metadata_filters_key = f"{session_key_prefix}_metadata_filters"
        metadata_keys_key = f"{session_key_prefix}_metadata_keys"

        if field_filters_key not in st.session_state:
            st.session_state[field_filters_key] = []
        if metadata_filters_key not in st.session_state:
            st.session_state[metadata_filters_key] = []
        if metadata_keys_key not in st.session_state:
            st.session_state[metadata_keys_key] = []

        metadata_options = [
            "metadata.keys",
            "metadata.notes.author",
            "metadata.custom.<field>",
        ]

        # Custom CSS for outlined green button style
        st.markdown(
            f"""
            <style>
            div[data-testid="column"] button[data-testid*="remove_{session_key_prefix}_filter"],
            div[data-testid="column"] button[data-testid*="remove_{session_key_prefix}_filter"]:focus,
            div[data-testid="column"] button[data-testid*="remove_{session_key_prefix}_filter"]:active,
            div[data-testid="column"] button[data-testid*="remove_{session_key_prefix}_key"],
            div[data-testid="column"] button[data-testid*="remove_{session_key_prefix}_key"]:focus,
            div[data-testid="column"] button[data-testid*="remove_{session_key_prefix}_key"]:active {{
                background-color: transparent !important;
                border: 2px solid #28a745 !important;
                color: #28a745 !important;
                border-radius: 5px !important;
                padding: 6px 12px !important;
                font-weight: 500 !important;
            }}
            div[data-testid="column"] button[data-testid*="remove_{session_key_prefix}_filter"]:hover,
            div[data-testid="column"] button[data-testid*="remove_{session_key_prefix}_key"]:hover {{
                background-color: #28a745 !important;
                color: white !important;
                border: 2px solid #28a745 !important;
            }}
            .add-{session_key_prefix}-filter-button-container button,
            .add-{session_key_prefix}-filter-button-container button:focus,
            .add-{session_key_prefix}-filter-button-container button:active {{
                background-color: transparent !important;
                border: 2px solid #28a745 !important;
                color: #28a745 !important;
                border-radius: 5px !important;
            }}
            .add-{session_key_prefix}-filter-button-container button:hover {{
                background-color: #28a745 !important;
                color: white !important;
                border: 2px solid #28a745 !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Column Filters")
        st.caption(
            f"Filter by {entity_type} columns or metadata paths (e.g., `metadata.custom.status`, `metadata.keys`)"
        )

        # Help box
        st.markdown("""
        <details>
        <summary><b>ℹ️ How to use Column Filters</b> (click to expand)</summary>
        <br>

        **Column Filters** let you filter entries by table columns or nested metadata fields.

        **Basic Column Filtering:**
        - Type a column name from the current table
        - Select an operator (`==`, `>`, `<`, `contains`, etc.)
        - Enter a value to filter by
        - Click "Add" to apply the filter

        **Examples:**
        - `score > 0.8` - Find high-scoring datapoints
        - `agent_id contains gpt` - Find agents with "gpt" in their ID
        - `time < 10` - Find fast datapoints (under 10 seconds)
        - `error == ""` - Find datapoints with no errors

        **Metadata Filtering:**
        - **`metadata.keys`** - Filter by metadata keys
        - **`metadata.notes.author`** - Filter by note author
        - **`metadata.custom.<field>`** - Filter by custom metadata fields

        **Supported Operators:**
        - `==`, `!=` - Equals/Not equals
        - `>`, `<`, `>=`, `<=` - Comparison (for numbers)
        - `contains` - String contains (for text)
        - `excludes` - String does NOT contain (opposite of contains)
        - `in` - Value in list (enter comma-separated: `a,b,c`)

        </details>
        """, unsafe_allow_html=True)

        # Display current filters
        all_column_filters = []
        for f in st.session_state[field_filters_key]:
            all_column_filters.append(("field", f))
        for m in st.session_state[metadata_filters_key]:
            all_column_filters.append(("metadata", m))

        if all_column_filters:
            field_filters_to_remove = []
            metadata_filters_to_remove = []

            cols = st.columns(4)
            for i, (filter_type, f) in enumerate(all_column_filters):
                display_value = ", ".join(f["value"]) if isinstance(f["value"], list) else str(f["value"])

                if filter_type == "field":
                    field_name = f["field"]
                else:
                    field_name = f"metadata.{f['path']}"

                filter_text = f"{field_name} {f['op']} {display_value}"

                with cols[i % 4]:
                    if st.button(filter_text, key=f"remove_{session_key_prefix}_filter_{i}", type="secondary", help="Click to remove filter"):
                        if filter_type == "field":
                            field_filters_to_remove.append(f)
                        else:
                            metadata_filters_to_remove.append(f)

            if field_filters_to_remove or metadata_filters_to_remove:
                if field_filters_to_remove:
                    st.session_state[field_filters_key] = [
                        f for f in st.session_state[field_filters_key]
                        if f not in field_filters_to_remove
                    ]
                if metadata_filters_to_remove:
                    removed_keys = [
                        m.get("value")
                        for m in metadata_filters_to_remove
                        if m.get("path") == "keys" and m.get("op") in ("contains", "has_key")
                    ]
                    st.session_state[metadata_filters_key] = [
                        m for m in st.session_state[metadata_filters_key]
                        if m not in metadata_filters_to_remove
                    ]
                    # Keep Quick Tags multiselect in sync when a tag filter is removed via the pill button
                    if removed_keys:
                        updated_tags = [
                            k for k in st.session_state[metadata_keys_key]
                            if k not in removed_keys
                        ]
                        st.session_state[metadata_keys_key] = updated_tags
                        tag_widget_key = f"{session_key_prefix}_tag_multiselect"
                        if tag_widget_key in st.session_state:
                            st.session_state[tag_widget_key] = updated_tags
                st.rerun()
        else:
            st.caption("No filters yet")

        # Add new column filter
        st.markdown("**Add Column Filter:**")
        st.caption(f"📋 Available columns: {', '.join(entity_columns + metadata_options)}")

        col1, col2, col3, col4 = st.columns([2, 1, 2, 0.5])

        with col1:
            new_field = st.text_input(
                "Column or metadata path",
                placeholder=f"e.g., {entity_columns[0]} or metadata.custom.status",
                key=f"new_{session_key_prefix}_field_name",
                label_visibility="collapsed",
                help="Type a column name or metadata path"
            )
        with col2:
            new_op = st.selectbox(
                "Operator",
                options=["==", "!=", ">", "<", ">=", "<=", "contains", "excludes", "in"],
                key=f"new_{session_key_prefix}_field_op",
                label_visibility="collapsed"
            )
        with col3:
            new_value = st.text_input(
                "Value",
                placeholder="e.g., value or a,b,c for 'in'",
                key=f"new_{session_key_prefix}_field_value",
                label_visibility="collapsed"
            )
        with col4:
            st.markdown(f'<div class="add-{session_key_prefix}-filter-button-container">', unsafe_allow_html=True)
            if st.button("Add", key=f"add_{session_key_prefix}_field_filter", disabled=not (new_field and new_value)):
                # Parse value
                if new_op == "in":
                    parsed_value = [v.strip() for v in new_value.split(",")]
                elif new_op in [">", "<", ">=", "<="]:
                    try:
                        parsed_value = float(new_value)
                    except ValueError:
                        parsed_value = new_value
                else:
                    parsed_value = new_value

                # Map UI-friendly operator names to backend operators
                op_mapping = {"excludes": "not_contains"}
                backend_op = op_mapping.get(new_op, new_op)

                if new_field.startswith("metadata."):
                    metadata_path = new_field[len("metadata."):]

                    if metadata_path == "notes.author":
                        st.session_state[metadata_filters_key].append({
                            "path": "notes",
                            "op": "has_author",
                            "value": parsed_value
                        })
                    else:
                        st.session_state[metadata_filters_key].append({
                            "path": metadata_path,
                            "op": backend_op,
                            "value": parsed_value
                        })
                    st.rerun()
                else:
                    st.session_state[field_filters_key].append({
                        "field": new_field,
                        "op": backend_op,
                        "value": parsed_value
                    })
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if available_tags is not None:
            st.markdown("#### Quick Tags")
            st.caption("Browse existing tags and instantly add `metadata.keys contains <tag>` filters")

            # Sync Quick Tags selection with any existing metadata key filters
            existing_tag_filters = [
                f.get("value")
                for f in st.session_state[metadata_filters_key]
                if f.get("path") == "keys" and f.get("op") in ("contains", "has_key")
            ]
            current_tag_selection = list(st.session_state[metadata_keys_key])
            for tag in existing_tag_filters:
                if isinstance(tag, str) and tag not in current_tag_selection:
                    current_tag_selection.append(tag)

            # Remove tags that are no longer available
            if available_tags:
                current_tag_selection = [t for t in current_tag_selection if t in available_tags]
            else:
                current_tag_selection = []
            st.session_state[metadata_keys_key] = current_tag_selection
            previous_tags = set(current_tag_selection)

            if available_tags:
                selected_tags = st.multiselect(
                    f"Quick Tags for {entity_display}s",
                    options=available_tags,
                    default=current_tag_selection,
                    key=f"{session_key_prefix}_tag_multiselect",
                    help="Adds a metadata.keys contains <tag> filter for each selected tag",
                    label_visibility="collapsed"
                )
                st.session_state[metadata_keys_key] = selected_tags

                added_tags = set(selected_tags) - previous_tags
                removed_tags = previous_tags - set(selected_tags)

                if added_tags:
                    for tag in added_tags:
                        filter_dict = {"path": "keys", "op": "contains", "value": tag}
                        if filter_dict not in st.session_state[metadata_filters_key]:
                            st.session_state[metadata_filters_key].append(filter_dict)

                if removed_tags:
                    st.session_state[metadata_filters_key] = [
                        f for f in st.session_state[metadata_filters_key]
                        if not (
                            f.get("path") == "keys"
                            and f.get("op") in ("contains", "has_key")
                            and f.get("value") in removed_tags
                        )
                    ]
                if added_tags or removed_tags:
                    st.rerun()
            else:
                st.session_state[metadata_keys_key] = []
                st.caption("No tags found for this type yet.")

        st.markdown("---")

        all_metadata_filters = list(st.session_state[metadata_filters_key])

        logic = st.radio(
            "Combine filters with",
            options=["AND", "OR"],
            horizontal=True,
            key=f"{session_key_prefix}_filter_logic"
        )

        total_filters = len(st.session_state[field_filters_key]) + len(all_metadata_filters)
        if total_filters > 0:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"**Active filters:** {len(st.session_state[field_filters_key])} column + {len(all_metadata_filters)} metadata = {total_filters} total")
            with col2:
                if st.button("🗑️ Clear All", key=f"clear_all_{session_key_prefix}_filters"):
                    st.session_state[field_filters_key] = []
                    st.session_state[metadata_filters_key] = []
                    st.session_state[metadata_keys_key] = []
                    st.rerun()

        has_column_filters = st.session_state[field_filters_key] or all_metadata_filters
        has_manual_selection = available_items and st.session_state.get(f"{session_key_prefix}_manual_selection")

        if has_column_filters and has_manual_selection:
            filter_spec = FilterSpec(
                field_filters=st.session_state[field_filters_key],
                metadata_filters=all_metadata_filters,
                logic=logic
            )
            return (filter_spec, st.session_state[f"{session_key_prefix}_manual_selection"])
        elif has_column_filters:
            return (FilterSpec(
                field_filters=st.session_state[field_filters_key],
                metadata_filters=all_metadata_filters,
                logic=logic
            ), None)
        elif has_manual_selection:
            return (None, st.session_state[f"{session_key_prefix}_manual_selection"])

        return (None, None)


def apply_entity_filters(
    conn,
    entity_type: str,
    id_column: str,
    filter_spec: Optional[FilterSpec],
    fallback_getter=None
) -> List[str]:
    """
    Generic function to apply filters for any entity type.

    Args:
        conn: Database connection
        entity_type: Type of entity (e.g., "agent", "task", "run", "datapoint")
        id_column: Name of ID column to extract (e.g., "agent_id", "task_id")
        filter_spec: FilterSpec object with filters
        fallback_getter: Optional function to get all IDs if no filter

    Returns:
        List of IDs matching the filters
    """
    if not filter_spec:
        if fallback_getter:
            return fallback_getter(conn)
        return []

    try:
        matching_rows = db_manager.apply_filters(conn, entity_type, filter_spec)
        return [row[id_column] for row in matching_rows]
    except Exception as e:
        st.error(f"Error applying {entity_type} filters: {e}")
        if fallback_getter:
            return fallback_getter(conn)
        return []


def apply_agent_filters(conn, filter_spec: Optional[FilterSpec]) -> List[str]:
    """Apply agent filters and return matching agent IDs."""
    return apply_entity_filters(conn, "agent", "agent_id", filter_spec, db_manager.get_agents)


def apply_task_filters(conn, filter_spec: Optional[FilterSpec]) -> List[str]:
    """Apply task filters and return matching task IDs."""
    return apply_entity_filters(conn, "task", "task_id", filter_spec, db_manager.get_tasks)


def apply_run_filters(conn, filter_spec: Optional[FilterSpec]) -> List[str]:
    """Apply run filters and return matching run IDs."""
    return apply_entity_filters(conn, "run", "run_id", filter_spec)


def apply_datapoint_filters(conn, filter_spec: Optional[FilterSpec]) -> List[str]:
    """Apply datapoint filters and return matching datapoint IDs."""
    return apply_entity_filters(conn, "datapoint", "datapoint_id", filter_spec)


def render_heatmap_filters():
    """Render heatmap filter controls in main area."""
    st.subheader("Heatmap Filters")

    try:
        conn = db_manager.connect_db(st.session_state.db_path)

        all_benchmarks = db_manager.get_benchmarks(conn)
        if not all_benchmarks:
            st.warning("No benchmarks found")
            conn.close()
            return (None, None), {}

        # Multi-select for benchmarks with "All" option
        benchmark_options = ["All"] + all_benchmarks

        col1, col2 = st.columns(2)

        with col1:
            # Filter saved values to only include valid options
            saved_benchmarks = st.session_state.get("heatmap_benchmarks", [])
            valid_benchmark_defaults = [b for b in saved_benchmarks if b in benchmark_options]

            selected_benchmarks_raw = st.multiselect(
                "Benchmarks",
                options=benchmark_options,
                default=valid_benchmark_defaults,
                key="filter_benchmarks",
                help="Select one or more benchmarks, or 'All' to include all"
            )

            # Expand "All" to actual benchmark list
            if "All" in selected_benchmarks_raw:
                selected_benchmarks = all_benchmarks
            else:
                selected_benchmarks = [b for b in selected_benchmarks_raw if b != "All"]

            st.session_state.heatmap_benchmarks = selected_benchmarks_raw

        with col2:
            # Get datasets for all selected benchmarks
            all_datasets_for_benchmarks = []
            for benchmark in selected_benchmarks:
                all_datasets_for_benchmarks.extend(db_manager.get_datasets(conn, benchmark))
            unique_datasets = sorted(set(all_datasets_for_benchmarks))

            if not unique_datasets:
                if selected_benchmarks:
                    st.warning(f"No datasets found for selected benchmark(s)")
                conn.close()
                return (None, None), {}

            dataset_options = ["All"] + unique_datasets

            # Filter saved values to only include valid options
            saved_datasets = st.session_state.get("heatmap_datasets", [])
            valid_dataset_defaults = [d for d in saved_datasets if d in dataset_options]

            selected_datasets_raw = st.multiselect(
                "Datasets",
                options=dataset_options,
                default=valid_dataset_defaults,
                key="filter_datasets",
                help="Select one or more datasets, or 'All' to include all"
            )

            # Expand "All" to actual dataset list
            if "All" in selected_datasets_raw:
                selected_datasets = unique_datasets
            else:
                selected_datasets = [d for d in selected_datasets_raw if d != "All"]

            st.session_state.heatmap_datasets = selected_datasets_raw

        # Guard: require at least one benchmark and one dataset
        if not selected_benchmarks or not selected_datasets:
            st.info("Select at least one benchmark and one dataset to view the heatmap")
            conn.close()
            return (None, None), {}

        # Get available entities across all selected benchmark/dataset combinations
        available_agents = db_manager.get_agents(conn)

        available_tasks = []
        available_runs = []
        available_datapoints = []
        cursor = conn.cursor()

        for benchmark in selected_benchmarks:
            for dataset in selected_datasets:
                # Only query if this benchmark actually has this dataset
                if dataset in db_manager.get_datasets(conn, benchmark):
                    available_tasks.extend(db_manager.get_tasks(conn, benchmark, dataset))

                    cursor.execute(
                        "SELECT DISTINCT run_id FROM runs WHERE benchmark_id = ? AND dataset_id = ?",
                        (benchmark, dataset)
                    )
                    available_runs.extend([row[0] for row in cursor.fetchall()])

                    cursor.execute(
                        "SELECT DISTINCT datapoint_id FROM datapoints WHERE benchmark_id = ? AND dataset_id = ?",
                        (benchmark, dataset)
                    )
                    available_datapoints.extend([row[0] for row in cursor.fetchall()])

        available_tasks = sorted(set(available_tasks))
        available_runs = sorted(set(available_runs))
        available_datapoints = sorted(set(available_datapoints))

        existing_tags_by_entity = {}
        for entity_key in ["agent", "task", "run", "datapoint"]:
            try:
                existing_tags_by_entity[entity_key] = db_manager.get_existing_keys(conn, entity_key)
            except Exception:
                existing_tags_by_entity[entity_key] = []

        conn.close()

        entity_configs = {
            "agent": {
                "columns": ["agent_id", "about", "agent_config"],
                "available_items": available_agents,
                "show_completeness_toggle": True,
                "available_tags": existing_tags_by_entity.get("agent", [])
            },
            "task": {
                "columns": ["task_id", "benchmark_id", "dataset_id"],
                "available_items": available_tasks,
                "show_completeness_toggle": True,
                "available_tags": existing_tags_by_entity.get("task", [])
            },
            "run": {
                "columns": ["run_id", "benchmark_id", "dataset_id", "agent_id", "hardware_info", "n", "threads", "run_start", "run_end", "total_time"],
                "available_items": available_runs,
                "available_tags": existing_tags_by_entity.get("run", [])
            },
            "datapoint": {
                "columns": ["datapoint_id", "benchmark_id", "dataset_id", "task_id", "agent_id", "run_id", "score", "time", "tokens", "error"],
                "available_items": available_datapoints,
                "available_tags": existing_tags_by_entity.get("datapoint", [])
            }
        }

        filter_specs = {}
        for entity_type, config in entity_configs.items():
            filter_result = render_generic_column_filters(
                entity_type=entity_type,
                entity_columns=config["columns"],
                session_key_prefix=f"heatmap_{entity_type}",
                available_items=config.get("available_items"),
                show_completeness_toggle=config.get("show_completeness_toggle", False),
                available_tags=config.get("available_tags")
            )
            if filter_result and (filter_result[0] or filter_result[1]):
                filter_specs[entity_type] = filter_result

        # Check if any filters are active (including completeness toggles and manual selections)
        has_any_filters = bool(filter_specs)
        has_completeness_toggle = (
            st.session_state.get("heatmap_agent_complete_only", False) or
            st.session_state.get("heatmap_task_complete_only", False)
        )
        has_manual_selection = any(
            st.session_state.get(f"heatmap_{entity_type}_manual_selection", [])
            for entity_type in entity_configs.keys()
        )

        if has_any_filters or has_completeness_toggle or has_manual_selection:
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("🗑️ Remove All Filters", type="secondary", use_container_width=True):
                    for entity_type in entity_configs.keys():
                        prefix = f"heatmap_{entity_type}"
                        st.session_state[f"{prefix}_field_filters"] = []
                        st.session_state[f"{prefix}_metadata_filters"] = []
                        st.session_state[f"{prefix}_metadata_keys"] = []
                        st.session_state[f"{prefix}_manual_selection"] = []
                        # Also clear the multiselect widget key to reset the UI
                        if f"{prefix}_multiselect" in st.session_state:
                            del st.session_state[f"{prefix}_multiselect"]
                        # Also delete completeness toggle widget key if it exists
                        # (can't set value after widget instantiated, must delete)
                        if f"{prefix}_complete_only" in st.session_state:
                            del st.session_state[f"{prefix}_complete_only"]
                    st.rerun()

        return (selected_benchmarks, selected_datasets), filter_specs

    except FileNotFoundError:
        st.error("Database not found")
        return (None, None), {}


# =============================================================================
# Coming Soon Tab
# =============================================================================

def render_coming_soon_tab():
    """Render the Coming Soon tab for future plots."""
    st.markdown("### 🚧 More Plots Coming Soon")
    st.markdown("""
    This section will include additional visualization plots such as:

    - **Performance Trends**: Track agent performance over time
    - **Comparison Charts**: Side-by-side agent comparisons
    - **Task Difficulty Analysis**: Visualize task complexity distributions
    - **Custom Metrics**: User-defined metric visualizations

    **Want to add a new plot?**

    New plots can be added by creating a new file in `src/ui/components/` following the pattern
    of `datapoint_heatmap.py` or `agent_metrics.py`. Each plot should:
    1. Have its own `render_<plot_name>_tab()` function in a separate file
    2. Be imported and added to the tabs in the `render()` function in `heatmap.py`
    3. Support filters and interactive features

    Contact your poRTLe administrator or check the documentation for details on
    adding custom visualization plots.
    """)

    st.info("💡 **Tip:** More plots will be added based on user needs and feedback.")


# =============================================================================
# Main Render Function
# =============================================================================

def render():
    """Main render function for plots page with multiple tabs."""
    # Import plot renderers from submodules
    from ui.components.datapoint_heatmap import render_datapoint_heatmap_tab
    from ui.components.agent_metrics import render_agent_metrics_tab
    
    st.title("📊 Plots")
    st.markdown("Interactive data visualizations and analytics")

    # Use radio buttons instead of tabs to prevent all tabs from rendering
    # This ensures only the selected plot is queried/rendered
    plot_options = ["📊 Datapoint Heatmap", "🤖 Agent Metrics", "🚧 More Plots (Coming Soon)"]
    
    # Restore last selected plot or default to first
    default_index = 0
    if st.session_state.get("selected_plot_tab"):
        saved_plot = st.session_state.selected_plot_tab
        if saved_plot in plot_options:
            default_index = plot_options.index(saved_plot)
    
    selected_plot = st.radio(
        "Select Plot Type",
        options=plot_options,
        index=default_index,
        horizontal=True,
        key="plot_type_selector",
        label_visibility="collapsed"
    )
    
    # Save selection for restoration
    st.session_state.selected_plot_tab = selected_plot
    
    st.markdown("---")

    # Only render the selected plot (not all of them)
    if selected_plot == "📊 Datapoint Heatmap":
        render_datapoint_heatmap_tab()
    elif selected_plot == "🤖 Agent Metrics":
        render_agent_metrics_tab()
    else:
        render_coming_soon_tab()


if __name__ == "__main__":
    # For testing component standalone
    render()
