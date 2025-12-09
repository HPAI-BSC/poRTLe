"""
Search Component for poRTLe UI

Provides search and filter interface for finding database entries.
"""

import json
import streamlit as st
from pathlib import Path
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ui.utils import db_manager, json_manager
from ui.utils.filter_engine import FilterSpec
from ui.components import detail_view
import pandas as pd
import os

DEMO_MODE = os.environ.get("PORTLE_DEMO_MODE", "").lower() == "true"

def _metadata_matches_query(metadata_value, query: str) -> bool:
    """Return True if metadata contains the query in any key/value."""
    if not query:
        return True

    normalized_query = query.lower()

    def search(value):
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if normalized_query in str(key).lower() or search(nested_value):
                    return True
            return False
        if isinstance(value, list):
            return any(search(item) for item in value)
        if value is None:
            return False
        return normalized_query in str(value).lower()

    if isinstance(metadata_value, str):
        if not metadata_value.strip():
            return False
        try:
            parsed_value = json.loads(metadata_value)
        except json.JSONDecodeError:
            parsed_value = metadata_value
    else:
        parsed_value = metadata_value

    return search(parsed_value)


def filter_results_by_metadata(results: pd.DataFrame, query: str) -> pd.DataFrame:
    """Filter results DataFrame to only rows whose metadata matches the query."""
    if not query or "metadata" not in results.columns:
        return results

    matches = results["metadata"].apply(
        lambda value: _metadata_matches_query(value, query)
    )
    return results[matches]


def render_advanced_filter_builder(table_type: str):
    """
    Render advanced filter builder UI using FilterSpec engine.

    Args:
        table_type: Type of table to filter

    Returns:
        FilterSpec object if filters are defined, None otherwise
    """
    with st.expander("🎯 Advanced Filters (Filter Engine)", expanded=False):
        st.markdown(
            "Build complex filters for nested metadata fields and custom attributes. "
            "Use this for queries like `custom.status == 'active'` or `rating > 3`."
        )

        metadata_keys_key = f"search_{table_type}_metadata_keys"
        # Initialize session state for filter builder
        if "field_filters" not in st.session_state:
            st.session_state.field_filters = []
        if "metadata_filters" not in st.session_state:
            st.session_state.metadata_filters = []
        if metadata_keys_key not in st.session_state:
            st.session_state[metadata_keys_key] = []

        # Get available columns for the selected table type
        available_tags = []
        try:
            conn = db_manager.connect_db(st.session_state.db_path)

            # Get columns for current table only
            tbl_columns = db_manager.get_table_columns(conn, table_type, exclude_json=True)
            column_names = [col['name'] for col in tbl_columns]

            available_tags = db_manager.get_existing_keys(conn, table_type)
            conn.close()

            # Add metadata path options
            metadata_options = [
                "metadata.keys",
                "metadata.notes.author",
                "metadata.custom.<field>",  # Placeholder to show custom metadata syntax
            ]
            column_names.extend(metadata_options)

        except Exception:
            # Fallback if database not available
            column_names = []
            try:
                conn.close()
            except Exception:
                pass

        # Field Filters Section
        st.markdown("#### Column Filters")
        st.caption(
            "Filter by table columns (e.g., `agent_id`, `score`) "
            "or metadata paths (e.g., `metadata.custom.status`, `metadata.keys`, `metadata.notes.author`)"
        )

        # Add help box using HTML details element (collapsible without nested expander)
        st.markdown("""
        <details>
        <summary><b>ℹ️ How to use Column Filters</b> (click to expand)</summary>
        <br>

        **Column Filters** let you filter entries by regular table columns or nested metadata fields.

        **Basic Column Filtering:**
        - Type a column name from the current table (e.g., `score`, `agent_id`)
        - Select an operator (`==`, `>`, `<`, `contains`, etc.)
        - Enter a value to filter by
        - Example: `score > 0.8` to find high-scoring entries

        **Metadata Filtering:**

        You can also filter by metadata fields using the `metadata.` prefix:

        - **`metadata.keys`** - Filter by metadata keys
          - Example: `metadata.keys contains production`
          - Use with `contains` operator to check if a key exists

        - **`metadata.notes.author`** - Filter by note author
          - Example: `metadata.notes.author == Dakota`
          - Finds entries with notes written by specific author

        - **`metadata.custom.<field>`** - Filter by custom metadata fields
          - Example: `metadata.custom.status == active`
          - Example: `metadata.custom.rating > 3`
          - Replace `<field>` with your actual custom field name

        **Supported Operators:**
        - `==` - Equals
        - `!=` - Not equals
        - `>`, `<`, `>=`, `<=` - Comparison (for numbers)
        - `contains` - String contains (for text)
        - `in` - Value in list (enter comma-separated: `a,b,c`)

        **Tips:**
        - Click any active filter button to remove it
        - Use the "Clear All" button to reset all filters
        - Combine multiple filters with AND/OR logic (see bottom of filter section)

        </details>
        """, unsafe_allow_html=True)


        # Display current field filters (includes both regular columns and metadata)
        # Combine field filters and metadata filters for display
        all_column_filters = []
        for f in st.session_state.field_filters:
            all_column_filters.append(("field", f))
        for m in st.session_state.metadata_filters:
            all_column_filters.append(("metadata", m))

        if all_column_filters:
            field_filters_to_remove = []
            metadata_filters_to_remove = []

            # Display filters as clickable buttons (like metadata keys)
            cols = st.columns(4)
            for i, (filter_type, f) in enumerate(all_column_filters):
                display_value = ", ".join(f["value"]) if isinstance(f["value"], list) else str(f["value"])

                # Get field/path name for display
                if filter_type == "field":
                    field_name = f["field"]
                else:  # metadata
                    # Show as "metadata.path" for clarity
                    field_name = f"metadata.{f['path']}"

                # Format filter as "field op value"
                filter_text = f"{field_name} {f['op']} {display_value}"

                with cols[i % 4]:
                    if st.button(filter_text, key=f"remove_filter_{i}", type="secondary", help=f"Click to remove filter"):
                        if filter_type == "field":
                            field_filters_to_remove.append(f)
                        else:
                            metadata_filters_to_remove.append(f)

            # Remove filters
            if field_filters_to_remove or metadata_filters_to_remove:
                if field_filters_to_remove:
                    st.session_state.field_filters = [
                        f for f in st.session_state.field_filters
                        if f not in field_filters_to_remove
                    ]
                if metadata_filters_to_remove:
                    removed_keys = [
                        m.get("value")
                        for m in metadata_filters_to_remove
                        if m.get("path") == "keys" and m.get("op") in ("contains", "has_key")
                    ]
                    st.session_state.metadata_filters = [
                        m for m in st.session_state.metadata_filters
                        if m not in metadata_filters_to_remove
                    ]
                    if removed_keys:
                        updated_tags = [
                            t for t in st.session_state[metadata_keys_key]
                            if t not in removed_keys
                        ]
                        st.session_state[metadata_keys_key] = updated_tags
                        tag_widget_key = f"search_tag_multiselect_{table_type}"
                        if tag_widget_key in st.session_state:
                            st.session_state[tag_widget_key] = updated_tags
                st.rerun()
        else:
            st.caption("No filters yet")

        # Add new field filter
        st.markdown("**Add Column Filter:**")

        # Show available columns as helper text
        if column_names:
            st.caption(f"📋 Available columns: {', '.join(column_names)}")

        col1, col2, col3, col4 = st.columns([2, 1, 2, 0.5])

        with col1:
            # Use text input to allow both column names and metadata paths
            new_field = st.text_input(
                "Column or metadata path",
                placeholder="e.g., score or metadata.custom.status",
                key="new_field_name",
                label_visibility="collapsed",
                help="Type a column name or metadata path (metadata.custom.*, metadata.keys, metadata.notes.author)"
            )
        with col2:
            new_op = st.selectbox(
                "Operator",
                options=["==", "!=", ">", "<", ">=", "<=", "contains", "in"],
                key="new_field_op",
                label_visibility="collapsed"
            )
        with col3:
            new_value = st.text_input(
                "Value",
                placeholder="e.g., 0.8 or a,b,c for 'in'",
                key="new_field_value",
                label_visibility="collapsed"
            )
        with col4:
            # Wrap button in styled container
            st.markdown('<div class="add-filter-button-container">', unsafe_allow_html=True)
            if st.button("Add", key="add_field_filter", disabled=not (new_field and new_value)):
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

                # Detect if this is a metadata path (starts with "metadata.")
                if new_field.startswith("metadata."):
                    # Extract the path after "metadata."
                    metadata_path = new_field[len("metadata."):]

                    # Handle special cases
                    if metadata_path == "notes.author":
                        # Convert to has_author operator for notes
                        st.session_state.metadata_filters.append({
                            "path": "notes",
                            "op": "has_author",
                            "value": parsed_value
                        })
                    else:
                        # Regular metadata filter
                        st.session_state.metadata_filters.append({
                            "path": metadata_path,
                            "op": new_op,
                            "value": parsed_value
                        })
                    st.rerun()
                else:
                    # Check if field has table prefix (e.g., "datapoint.score")
                    if "." in new_field and not new_field.startswith("metadata."):
                        # Has table prefix - support cross-table filtering
                        field_parts = new_field.split(".", 1)
                        if len(field_parts) == 2:
                            prefix_table, column_name = field_parts
                            if prefix_table != table_type:
                                # Cross-table filter - keep the prefix for JOIN query
                                st.session_state.field_filters.append({
                                    "field": new_field,  # Keep full "table.column" format
                                    "op": new_op,
                                    "value": parsed_value
                                })
                                st.rerun()
                            else:
                                # Same table - use column name without prefix
                                st.session_state.field_filters.append({
                                    "field": column_name,
                                    "op": new_op,
                                    "value": parsed_value
                                })
                                st.rerun()
                        else:
                            # Just use field as-is
                            st.session_state.field_filters.append({
                                "field": new_field,
                                "op": new_op,
                                "value": parsed_value
                            })
                            st.rerun()
                    else:
                        # Regular column filter without prefix
                        st.session_state.field_filters.append({
                            "field": new_field,
                            "op": new_op,
                            "value": parsed_value
                        })
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Quick Tags for metadata.keys
        st.markdown("#### Quick Tags")
        st.caption("Browse existing keys and instantly add `metadata.keys contains <tag>` filters")

        existing_tag_filters = [
            f.get("value")
            for f in st.session_state.metadata_filters
            if f.get("path") == "keys" and f.get("op") in ("contains", "has_key")
        ]
        current_tag_selection = list(st.session_state[metadata_keys_key])
        for tag in existing_tag_filters:
            if isinstance(tag, str) and tag not in current_tag_selection:
                current_tag_selection.append(tag)

        if available_tags:
            current_tag_selection = [t for t in current_tag_selection if t in available_tags]
            st.session_state[metadata_keys_key] = current_tag_selection
            previous_tags = set(current_tag_selection)

            selected_tags = st.multiselect(
                f"Quick Tags for {table_type.capitalize()}s",
                options=available_tags,
                default=current_tag_selection,
                key=f"search_tag_multiselect_{table_type}",
                help="Adds a metadata.keys contains <tag> filter for each selected tag",
                label_visibility="collapsed"
            )
            st.session_state[metadata_keys_key] = selected_tags

            added_tags = set(selected_tags) - previous_tags
            removed_tags = previous_tags - set(selected_tags)

            if added_tags:
                for tag in added_tags:
                    filter_dict = {"path": "keys", "op": "contains", "value": tag}
                    if filter_dict not in st.session_state.metadata_filters:
                        st.session_state.metadata_filters.append(filter_dict)

            if removed_tags:
                st.session_state.metadata_filters = [
                    f for f in st.session_state.metadata_filters
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

        # Custom CSS for outlined green button style
        st.markdown(
            """
            <style>
            /* Style for column filter buttons */
            div[data-testid="column"] button[data-testid*="remove_filter"],
            div[data-testid="column"] button[data-testid*="remove_filter"]:focus,
            div[data-testid="column"] button[data-testid*="remove_filter"]:active {
                background-color: transparent !important;
                border: 2px solid #28a745 !important;
                color: #28a745 !important;
                border-radius: 5px !important;
                padding: 6px 12px !important;
                font-weight: 500 !important;
            }
            div[data-testid="column"] button[data-testid*="remove_filter"]:hover {
                background-color: #28a745 !important;
                color: white !important;
                border: 2px solid #28a745 !important;
            }
            /* Style for Add filter buttons */
            .add-filter-button-container button,
            .add-filter-button-container button:focus,
            .add-filter-button-container button:active {
                background-color: transparent !important;
                border: 2px solid #28a745 !important;
                color: #28a745 !important;
                border-radius: 5px !important;
            }
            .add-filter-button-container button:hover {
                background-color: #28a745 !important;
                color: white !important;
                border: 2px solid #28a745 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Collect all metadata filters (only from explicit column filters)
        all_metadata_filters = list(st.session_state.metadata_filters)

        # Logic selector
        logic = st.radio(
            "Combine filters with",
            options=["AND", "OR"],
            horizontal=True,
            help="AND: All filters must match. OR: Any filter can match."
        )

        # Show filter summary and clear button
        total_filters = len(st.session_state.field_filters) + len(all_metadata_filters)
        if total_filters > 0:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"**Active filters:** {len(st.session_state.field_filters)} column + {len(all_metadata_filters)} metadata = {total_filters} total")
            with col2:
                if st.button("🗑️ Clear All", key="clear_all_filters"):
                    st.session_state.field_filters = []
                    st.session_state.metadata_filters = []
                    st.session_state[metadata_keys_key] = []
                    st.rerun()

        # Return FilterSpec if any filters are defined
        if st.session_state.field_filters or all_metadata_filters:
            return FilterSpec(
                field_filters=st.session_state.field_filters,
                metadata_filters=all_metadata_filters,
                logic=logic
            )

        return None


def render_search_filters():
    """Render search filter controls in the main content area."""
    st.subheader("Search Filters")

    # Get options
    options = list(db_manager.TABLES.keys())

    # Initialize saved selection if not present or invalid
    if "saved_search_table_type" not in st.session_state or st.session_state.saved_search_table_type not in options:
        st.session_state.saved_search_table_type = options[0] if options else "agent"

    # Find index of saved selection
    try:
        default_index = options.index(st.session_state.saved_search_table_type)
    except ValueError:
        default_index = 0

    # Table type selector
    table_type = st.selectbox(
        "Entry Type",
        options=options,
        index=default_index,
        format_func=lambda x: x.capitalize()
    )

    # Update saved state
    st.session_state.saved_search_table_type = table_type

    st.markdown("---")

    # ID search
    search_id = st.text_input(
        "Search by ID",
        placeholder=f"Enter {table_type} ID...",
        help=f"Search for specific {table_type} by ID"
    )

    # Initialize selected_ids
    selected_ids = []

    # Quick Selection for multiple IDs
    try:
        conn = db_manager.connect_db(st.session_state.db_path)

        # Get available IDs based on table type
        id_col = db_manager.TABLES[table_type][1]
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT {id_col} FROM {db_manager.TABLES[table_type][0]} ORDER BY {id_col}")
        available_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        if available_ids:
            # Initialize session state for quick selection
            quick_selection_key = f"search_quick_selection_{table_type}"
            if quick_selection_key not in st.session_state:
                st.session_state[quick_selection_key] = []

            selected_ids = st.multiselect(
                f"Quick Selection - Select Multiple {table_type.capitalize()}s",
                options=available_ids,
                default=st.session_state[quick_selection_key],
                key=f"search_multiselect_{table_type}",
                help=f"Select multiple {table_type}s to filter results"
            )

            # Update session state
            st.session_state[quick_selection_key] = selected_ids
    except Exception:
        selected_ids = []

    # Advanced filter builder
    filter_spec = render_advanced_filter_builder(table_type)

    st.markdown("---")

    # Advanced filters based on table type
    filters = {}

    try:
        conn = db_manager.connect_db(st.session_state.db_path)

        if table_type in ["dataset", "task", "run", "datapoint"]:
            # Benchmark filter - with persistence
            benchmarks = db_manager.get_benchmarks(conn)
            if benchmarks:
                # Initialize session state
                filter_key = f"search_filter_benchmark_{table_type}"
                if filter_key not in st.session_state:
                    st.session_state[filter_key] = "All"
                # Validate saved value still exists
                saved_val = st.session_state[filter_key]
                if saved_val != "All" and saved_val not in benchmarks:
                    st.session_state[filter_key] = "All"
                
                benchmark_filter = st.selectbox(
                    "Benchmark",
                    options=["All"] + benchmarks,
                    index=(["All"] + benchmarks).index(st.session_state[filter_key]),
                    key=f"search_benchmark_select_{table_type}"
                )
                st.session_state[filter_key] = benchmark_filter
                if benchmark_filter != "All":
                    filters["benchmark_id"] = benchmark_filter

        if table_type in ["task", "run", "datapoint"]:
            # Dataset filter - with persistence
            benchmark_id = filters.get("benchmark_id")
            datasets = db_manager.get_datasets(conn, benchmark_id)
            if datasets:
                filter_key = f"search_filter_dataset_{table_type}"
                if filter_key not in st.session_state:
                    st.session_state[filter_key] = "All"
                saved_val = st.session_state[filter_key]
                if saved_val != "All" and saved_val not in datasets:
                    st.session_state[filter_key] = "All"
                
                dataset_filter = st.selectbox(
                    "Dataset",
                    options=["All"] + datasets,
                    index=(["All"] + datasets).index(st.session_state[filter_key]),
                    key=f"search_dataset_select_{table_type}"
                )
                st.session_state[filter_key] = dataset_filter
                if dataset_filter != "All":
                    filters["dataset_id"] = dataset_filter

        if table_type == "datapoint":
            # Task filter - with persistence
            benchmark_id = filters.get("benchmark_id")
            dataset_id = filters.get("dataset_id")
            tasks = db_manager.get_tasks(conn, benchmark_id, dataset_id)
            if tasks:
                filter_key = f"search_filter_task_{table_type}"
                if filter_key not in st.session_state:
                    st.session_state[filter_key] = "All"
                saved_val = st.session_state[filter_key]
                if saved_val != "All" and saved_val not in tasks:
                    st.session_state[filter_key] = "All"
                
                task_filter = st.selectbox(
                    "Task",
                    options=["All"] + tasks,
                    index=(["All"] + tasks).index(st.session_state[filter_key]),
                    key=f"search_task_select_{table_type}"
                )
                st.session_state[filter_key] = task_filter
                if task_filter != "All":
                    filters["task_id"] = task_filter

        if table_type in ["run", "datapoint"]:
            # Agent filter - with persistence
            agents = db_manager.get_agents(conn)
            if agents:
                filter_key = f"search_filter_agent_{table_type}"
                if filter_key not in st.session_state:
                    st.session_state[filter_key] = "All"
                saved_val = st.session_state[filter_key]
                if saved_val != "All" and saved_val not in agents:
                    st.session_state[filter_key] = "All"
                
                agent_filter = st.selectbox(
                    "Agent",
                    options=["All"] + agents,
                    index=(["All"] + agents).index(st.session_state[filter_key]),
                    key=f"search_agent_select_{table_type}"
                )
                st.session_state[filter_key] = agent_filter
                if agent_filter != "All":
                    filters["agent_id"] = agent_filter

        conn.close()

    except FileNotFoundError:
        st.error("Database not found")
        return None, None, None, None, None

    return table_type, search_id, filters, filter_spec, selected_ids


def display_search_results(table_type: str, results: pd.DataFrame):
    """
    Display search results - default as table, with optional expandable view for <50 results.

    Args:
        table_type: Type of table being displayed
        results: DataFrame with search results
    """
    if results.empty:
        st.info(f"No {table_type}s found matching the filters.")
        return

    result_count = len(results)
    st.write(f"Found **{result_count}** {table_type}(s)")

    # Determine ID column
    id_col = db_manager.TABLES[table_type][1]

    # Show toggle for expandable view if < 50 results
    show_expandable = False
    if result_count < 50:
        col1, col2 = st.columns([3, 1])
        with col2:
            show_expandable = st.toggle("Expandable View", value=False, key="search_expandable_toggle")

    if show_expandable:
        # Display each result as an expander (compact format with colored badges)
        for idx, row in results.iterrows():
            entry_id = row[id_col]
            
            with st.expander(entry_id, expanded=False):
                # Get all columns except metadata and id (handle separately)
                regular_cols = [c for c in results.columns if c not in ["metadata", id_col]]
                
                # Define highlight columns with colors (for datapoint mainly)
                highlight_styles = {
                    "score": ("🎯", "#28a745", "#d4edda"),      # Green
                    "tokens": ("🔢", "#6f42c1", "#e2d9f3"),     # Purple  
                    "time": ("⏱️", "#fd7e14", "#ffeeba"),       # Orange
                    "error": ("⚠️", "#dc3545", "#f8d7da"),      # Red
                }
                
                # Build compact inline display of fields
                highlighted_parts = []
                regular_parts = []
                
                for col in regular_cols:
                    val = row[col]
                    if val is not None and str(val) != "nan" and str(val) != "":
                        # Format value based on type
                        if isinstance(val, float):
                            if col == "score":
                                formatted_val = f"{val:.2f}"
                            elif col == "time" and val != -1:
                                formatted_val = f"{val:.1f}s"
                            else:
                                formatted_val = str(val)
                        elif col == "tokens" and val != -1:
                            formatted_val = str(int(val))
                        else:
                            formatted_val = str(val)
                        
                        # Check if this is a highlight column
                        if col in highlight_styles:
                            icon, text_color, bg_color = highlight_styles[col]
                            # Skip if value indicates missing/error (like -1 for time/tokens)
                            if col in ["time", "tokens"] and val == -1:
                                continue
                            highlighted_parts.append(
                                f'<span style="background-color:{bg_color};color:{text_color};'
                                f'padding:2px 6px;border-radius:4px;font-weight:600;margin-right:4px;">'
                                f'{icon} {col}: {formatted_val}</span>'
                            )
                        else:
                            regular_parts.append(f"**{col}:** {formatted_val}")
                
                # Display highlighted fields first (as HTML)
                if highlighted_parts:
                    st.markdown(" ".join(highlighted_parts), unsafe_allow_html=True)
                
                # Display regular fields
                if regular_parts:
                    st.markdown(" · ".join(regular_parts))
                
                # Display metadata compactly if present
                if "metadata" in results.columns and row["metadata"]:
                    try:
                        if isinstance(row["metadata"], str):
                            metadata = json.loads(row["metadata"])
                        else:
                            metadata = row["metadata"]
                        
                        meta_parts = []
                        
                        # Display keys with colored badges
                        if metadata.get("keys"):
                            key_badges = []
                            for key in metadata["keys"]:
                                key_badges.append(
                                    f'<span style="background-color:#cfe2ff;color:#084298;'
                                    f'padding:2px 6px;border-radius:4px;font-weight:500;margin-right:4px;">'
                                    f'🏷️ {key}</span>'
                                )
                            if key_badges:
                                st.markdown(" ".join(key_badges), unsafe_allow_html=True)
                        
                        # Display custom fields inline
                        if metadata.get("custom"):
                            custom_items = [f"{k}: {v}" for k, v in metadata["custom"].items() if v is not None and v != ""]
                            if custom_items:
                                st.markdown(f"📋 {', '.join(custom_items)}")
                        
                        # Display notes formatted nicely
                        if metadata.get("notes"):
                            for note in metadata["notes"]:
                                if isinstance(note, dict):
                                    # Format note dict: author, date, text
                                    author = note.get("author", "Unknown")
                                    date = note.get("date_added", "")
                                    text = note.get("text", "")
                                    if text:
                                        st.markdown(f"📝 **{author}** ({date}): {text}")
                                else:
                                    # Plain string note
                                    st.markdown(f"📝 {note}")
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                # Compact action buttons on same line
                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
                with btn_col1:
                    if st.button(f"📄 View", key=f"view_{table_type}_{entry_id}"):
                        st.session_state.selected_entry = entry_id
                        st.session_state.selected_table = table_type
                        st.session_state.viewing_detail = True
                        st.rerun()
                with btn_col2:
                    if st.button(f"✏️ Edit", key=f"edit_{table_type}_{entry_id}", disabled=DEMO_MODE):
                        st.session_state.selected_entry = entry_id
                        st.session_state.selected_table = table_type
                        st.session_state[f"edit_mode_{table_type}_{entry_id}"] = True
                        st.session_state.viewing_detail = True
                        st.rerun()
    else:
        # Default table view
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="search_results_table"
        )
        
        # Handle row selection for detail view
        if st.session_state.get("search_results_table") and st.session_state.search_results_table.selection.rows:
            selected_idx = st.session_state.search_results_table.selection.rows[0]
            selected_id = results.iloc[selected_idx][id_col]
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"📄 View Details: {selected_id}", use_container_width=True):
                    st.session_state.selected_entry = selected_id
                    st.session_state.selected_table = table_type
                    st.session_state.viewing_detail = True
                    st.rerun()
            with col2:
                if st.button(f"✏️ Edit: {selected_id}", use_container_width=True, disabled=DEMO_MODE):
                    st.session_state.selected_entry = selected_id
                    st.session_state.selected_table = table_type
                    st.session_state[f"edit_mode_{table_type}_{selected_id}"] = True
                    st.session_state.viewing_detail = True
                    st.rerun()


def render():
    """Main render function for search page."""
    # Check if we're viewing a detail page
    if st.session_state.get("viewing_detail") and st.session_state.get("selected_entry"):
        # Show detail view with back button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Back to Search"):
                st.session_state.viewing_detail = False
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

    # Normal search interface
    st.title("🔍 Search Database")
    st.markdown("Search and browse poRTLe database entries")

    # Check if database exists
    if not st.session_state.db_path.exists():
        st.error(f"Database not found at: {st.session_state.db_path}")
        st.info("Please run `build_datatable.py` to create the database.")
        return

    # Render filters and get search parameters
    table_type, search_id, filters, filter_spec, selected_ids = render_search_filters()

    if table_type is None:
        return

    try:
        conn = db_manager.connect_db(st.session_state.db_path)

        # Perform search
        if search_id:
            # Direct ID search
            row = db_manager.query_entry(conn, table_type, search_id)
            if row:
                # Convert single row to DataFrame
                results = pd.DataFrame([dict(row)])
                st.success(f"Found {table_type}: `{search_id}`")
            else:
                results = pd.DataFrame()
                st.warning(f"No {table_type} found with ID: `{search_id}`")
        elif selected_ids:
            # Quick selection search
            id_col = db_manager.TABLES[table_type][1]
            placeholders = ",".join("?" * len(selected_ids))
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {db_manager.TABLES[table_type][0]} WHERE {id_col} IN ({placeholders})", selected_ids)
            rows = cursor.fetchall()
            results = pd.DataFrame([dict(row) for row in rows])
            st.success(f"Found {len(results)} {table_type}(s) from Quick Selection")
        elif filter_spec:
            # Advanced filter search using FilterSpec
            try:
                rows = db_manager.apply_filters(conn, table_type, filter_spec)
                results = pd.DataFrame([dict(row) for row in rows])
                st.success(f"Applied advanced filters: {filter_spec}")
            except Exception as e:
                st.error(f"Error applying filters: {e}")
                results = pd.DataFrame()
        else:
            # Standard filter-based search
            if filters:
                rows = db_manager.search_entries(conn, table_type, filters)
                results = pd.DataFrame([dict(row) for row in rows])
            else:
                # Get all entries
                results = db_manager.get_all_entries(conn, table_type)

        conn.close()

        # Display results
        if not results.empty:
            display_search_results(table_type, results)
        elif not search_id:
            st.info(f"No {table_type}s found. Try adjusting filters.")

    except Exception as e:
        st.error(f"Error searching database: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    # For testing component standalone
    render()
