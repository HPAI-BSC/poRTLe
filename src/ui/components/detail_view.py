"""
Detail View Component for poRTLe UI

Displays detailed information about database entries.
Reuses logic from display_table_entry.py.
"""

import streamlit as st
from pathlib import Path
import sys
import json
import re
import os

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ui.utils import db_manager, json_manager, file_viewer

# Demo mode - disables all data modification buttons when True
DEMO_MODE = os.environ.get("PORTLE_DEMO_MODE", "").lower() == "true"


def ansi_to_html(text: str) -> str:
    """
    Convert ANSI escape sequences to HTML with inline styles.

    Args:
        text: Text containing ANSI escape sequences

    Returns:
        HTML string with ANSI codes converted to styled spans
    """
    # ANSI color code mappings (standard terminal colors)
    ansi_colors = {
        # Regular colors
        '30': '#2e3436',  # Black
        '31': '#cc0000',  # Red
        '32': '#4e9a06',  # Green
        '33': '#c4a000',  # Yellow
        '34': '#3465a4',  # Blue
        '35': '#75507b',  # Magenta
        '36': '#06989a',  # Cyan
        '37': '#d3d7cf',  # White

        # Bright colors
        '90': '#555753',  # Bright Black (Gray)
        '91': '#ef2929',  # Bright Red
        '92': '#8ae234',  # Bright Green
        '93': '#fce94f',  # Bright Yellow
        '94': '#729fcf',  # Bright Blue
        '95': '#ad7fa8',  # Bright Magenta
        '96': '#34e2e2',  # Bright Cyan
        '97': '#eeeeec',  # Bright White
    }

    # Track current styles
    current_styles = {
        'color': None,
        'bold': False,
        'dim': False,
        'italic': False,
        'underline': False,
    }

    def get_style_string(styles: dict) -> str:
        """Generate inline CSS style string from current styles."""
        style_parts = []
        if styles['color']:
            style_parts.append(f"color: {styles['color']}")
        if styles['bold']:
            style_parts.append("font-weight: bold")
        if styles['dim']:
            style_parts.append("opacity: 0.6")
        if styles['italic']:
            style_parts.append("font-style: italic")
        if styles['underline']:
            style_parts.append("text-decoration: underline")
        return "; ".join(style_parts) if style_parts else ""

    # Split text into segments at ANSI escape codes
    ansi_escape_pattern = re.compile(r'\x1b\[([0-9;]+)m')

    result = []
    last_pos = 0
    spans_open = 0

    for match in ansi_escape_pattern.finditer(text):
        # Add text before this escape code
        if match.start() > last_pos:
            text_segment = text[last_pos:match.start()]
            # HTML escape the text
            text_segment = (text_segment
                           .replace('&', '&amp;')
                           .replace('<', '&lt;')
                           .replace('>', '&gt;'))
            result.append(text_segment)

        # Process the ANSI code
        codes = match.group(1).split(';')

        for code in codes:
            if code == '0':  # Reset
                # Close any open span
                if spans_open > 0:
                    result.append('</span>')
                    spans_open -= 1
                current_styles = {
                    'color': None,
                    'bold': False,
                    'dim': False,
                    'italic': False,
                    'underline': False,
                }
            elif code == '1':  # Bold
                current_styles['bold'] = True
            elif code == '2':  # Dim
                current_styles['dim'] = True
            elif code == '3':  # Italic
                current_styles['italic'] = True
            elif code == '4':  # Underline
                current_styles['underline'] = True
            elif code in ansi_colors:  # Color codes
                current_styles['color'] = ansi_colors[code]

        # Open a new span with updated styles if we have any styles
        style_string = get_style_string(current_styles)
        if style_string:
            # Close previous span if open
            if spans_open > 0:
                result.append('</span>')
                spans_open -= 1
            result.append(f'<span style="{style_string}">')
            spans_open += 1

        last_pos = match.end()

    # Add remaining text
    if last_pos < len(text):
        text_segment = text[last_pos:]
        text_segment = (text_segment
                       .replace('&', '&amp;')
                       .replace('<', '&lt;')
                       .replace('>', '&gt;'))
        result.append(text_segment)

    # Close any remaining open spans
    while spans_open > 0:
        result.append('</span>')
        spans_open -= 1

    return ''.join(result)


def render_logs_and_reports(row: dict, datapoint_id: str):
    """
    Render logs and reports section for a datapoint.

    Args:
        row: Database row for the datapoint
        datapoint_id: Datapoint ID
    """
    st.markdown("### 📁 Logs and Reports")

    # Get necessary IDs from the row
    benchmark_id = row["benchmark_id"] if "benchmark_id" in row.keys() else None
    dataset_id = row["dataset_id"] if "dataset_id" in row.keys() else None
    run_id = row["run_id"] if "run_id" in row.keys() else None
    agent_id = row["agent_id"] if "agent_id" in row.keys() else None

    if not all([benchmark_id, dataset_id, run_id, agent_id]):
        st.warning("Missing required IDs to locate log files")
        return

    # Get run_directory from runs table if available
    run_directory = None
    try:
        conn = db_manager.connect_db(st.session_state.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT run_directory FROM runs WHERE run_id = ?", (run_id,))
        result = cursor.fetchone()
        if result and result["run_directory"]:
            run_directory = result["run_directory"]
        conn.close()
    except Exception:
        # If there's any error, just use None and fall back to constructed path
        pass

    # Extract datapoint_directory from datapoint metadata
    datapoint_directory = None
    if "metadata" in row:
        metadata = json_manager.parse_metadata(row["metadata"])
        datapoint_directory = metadata.get("custom", {}).get("datapoint_directory")

    # Construct file paths
    repo_root = Path(__file__).parent.parent.parent.parent
    agent_log_paths, test_report_paths, reports_dir = file_viewer.construct_log_paths(
        repo_root=repo_root,
        benchmark_id=benchmark_id,
        dataset_id=dataset_id,
        run_id=run_id,
        agent_id=agent_id,
        datapoint_id=datapoint_id,
        run_directory=run_directory,
        datapoint_directory=datapoint_directory
    )

    # Display file paths always
    st.markdown("#### File Paths")

    # Show the reports directory being searched
    if reports_dir:
        dir_exists_marker = "✅" if reports_dir.exists() else "❌"
        st.code(f"{dir_exists_marker} Reports Directory: {reports_dir}", language="text")

    if agent_log_paths:
        st.markdown(f"**Found {len(agent_log_paths)} Agent Log(s):**")
        for log_path in agent_log_paths:
            st.code(f"✅ {log_path}", language="text")
    else:
        st.code("❌ Agent Log: No files found", language="text")

    if test_report_paths:
        st.markdown(f"**Found {len(test_report_paths)} Test Report(s):**")
        for report_path in test_report_paths:
            st.code(f"✅ {report_path}", language="text")
    else:
        st.code("❌ Test Report: No files found", language="text")

    st.markdown("---")

    # Initialize session state for loaded files if not exists
    if "loaded_log_files" not in st.session_state:
        st.session_state.loaded_log_files = set()

    # Create two columns for the file viewers
    file_col1, file_col2 = st.columns(2)

    with file_col1:
        st.markdown("#### 📄 Agent Logs")
        if agent_log_paths:
            for i, agent_log_path in enumerate(agent_log_paths):
                with st.expander(f"📄 {agent_log_path.name}", expanded=False):
                    file_info = file_viewer.get_file_info(agent_log_path)
                    st.caption(file_info)

                    # Download button
                    try:
                        with open(agent_log_path, 'rb') as f:
                            st.download_button(
                                label=f"⬇️ Download {agent_log_path.name}",
                                data=f,
                                file_name=agent_log_path.name,
                                mime="text/plain",
                                key=f"download_agent_{i}_{datapoint_id}"
                            )
                    except Exception as e:
                        st.error(f"Error creating download button: {e}")

                    # Lazy loading: only load content when user clicks "Load Content"
                    file_key = f"agent_log_{i}_{datapoint_id}"
                    if file_key in st.session_state.loaded_log_files:
                        # Content already loaded, show it
                        success, content = file_viewer.read_file_content(agent_log_path)
                        if success:
                            # Convert ANSI to HTML for colored display
                            html_content = ansi_to_html(content)
                            st.markdown(
                                f"""
                                <div style="
                                    background-color: #0e1117;
                                    color: #fafafa;
                                    font-family: 'Source Code Pro', monospace;
                                    white-space: pre-wrap;
                                    padding: 1rem;
                                    border-radius: 0.5rem;
                                    line-height: 1.5;
                                    font-size: 14px;
                                    overflow-x: auto;
                                ">
                                    {html_content}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.error(content)
                    else:
                        # Show load button
                        st.info("💡 Log content not loaded. Click below to load (may be slow for large files).")
                        if st.button(f"📂 Load Log Content", key=f"load_agent_{i}_{datapoint_id}"):
                            st.session_state.loaded_log_files.add(file_key)
                            st.rerun()
        else:
            st.warning("No agent log files found")

    with file_col2:
        st.markdown("#### 📊 Test Reports")
        if test_report_paths:
            for i, test_report_path in enumerate(test_report_paths):
                with st.expander(f"📊 {test_report_path.name}", expanded=False):
                    file_info = file_viewer.get_file_info(test_report_path)
                    st.caption(file_info)

                    # Download button
                    try:
                        with open(test_report_path, 'rb') as f:
                            st.download_button(
                                label=f"⬇️ Download {test_report_path.name}",
                                data=f,
                                file_name=test_report_path.name,
                                mime="text/plain",
                                key=f"download_report_{i}_{datapoint_id}"
                            )
                    except Exception as e:
                        st.error(f"Error creating download button: {e}")

                    # Lazy loading: only load content when user clicks "Load Content"
                    file_key = f"test_report_{i}_{datapoint_id}"
                    if file_key in st.session_state.loaded_log_files:
                        # Content already loaded, show it
                        success, content = file_viewer.read_file_content(test_report_path)
                        if success:
                            st.code(content, language="text", line_numbers=False)
                        else:
                            st.error(content)
                    else:
                        # Show load button
                        st.info("💡 Report content not loaded. Click below to load (may be slow for large files).")
                        if st.button(f"📂 Load Report Content", key=f"load_report_{i}_{datapoint_id}"):
                            st.session_state.loaded_log_files.add(file_key)
                            st.rerun()
        else:
            st.warning("No test report files found")


def render_metadata_section(metadata_dict: dict):
    """
    Render metadata in an expandable section.

    Args:
        metadata_dict: Parsed metadata dictionary
    """
    with st.expander("📋 Metadata", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Keys**")
            if metadata_dict.get("keys"):
                keys_display = " ".join(f"`{key}`" for key in metadata_dict["keys"])
                st.markdown(keys_display)
            else:
                st.caption("No keys")

        with col2:
            st.markdown("**Custom Fields**")
            if metadata_dict.get("custom"):
                for k, v in metadata_dict["custom"].items():
                    st.text(f"{k}: {v}")
            else:
                st.caption("No custom fields")

        st.markdown("**Notes**")
        if metadata_dict.get("notes"):
            for i, note in enumerate(metadata_dict["notes"], 1):
                date = note.get("date_added", "N/A")
                author = note.get("author", "Unknown")
                text = note.get("text", "")

                st.markdown(f"""
                **[{i}]** *{date}* - {author}
                > {text}
                """)
        else:
            st.caption("No notes")


def render_agent_config(agent_config_str: str):
    """
    Render agent configuration.

    Args:
        agent_config_str: JSON string of agent config
    """
    try:
        config = json.loads(agent_config_str) if agent_config_str else {}

        with st.expander("⚙️ Agent Configuration", expanded=False):
            st.json(config)

    except json.JSONDecodeError:
        st.error("Invalid agent configuration JSON")


def render_related_entries(relations: list, table_type: str):
    """
    Render related entries as clickable links.

    Args:
        relations: List of (relation_type, relation_id) tuples
        table_type: Current table type
    """
    if not relations:
        return

    with st.expander("🔗 Related Entries", expanded=False):
        for rel_type, rel_id in relations:
            col1, col2 = st.columns([1, 3])

            with col1:
                st.caption(rel_type.capitalize())

            with col2:
                # Create clickable button to view related entry
                if st.button(f"View {rel_id}", key=f"rel_{rel_type}_{rel_id}"):
                    st.session_state.selected_entry = rel_id
                    st.session_state.selected_table = rel_type
                    st.session_state.viewing_detail = True
                    # Clear datapoint viewing mode if coming from heatmap
                    st.session_state.viewing_datapoints = False
                    st.rerun()


def render_field_value(key: str, value: any):
    """
    Render a field value with appropriate formatting.

    Args:
        key: Field name
        value: Field value
    """
    if value is None:
        st.text(f"{key}: (null)")
    elif isinstance(value, (int, float)):
        if key == "time" and value != -1:
            st.text(f"{key}: {value:.2f}s")
        elif key == "score" and value != -1:
            st.text(f"{key}: {value:.2f}")
        else:
            st.text(f"{key}: {value}")
    else:
        # Truncate long values
        value_str = str(value)
        if len(value_str) > 200:
            st.text(f"{key}: {value_str[:200]}...")
        else:
            st.text(f"{key}: {value_str}")


def render_inline_metadata_editor(row, table_type: str, item_id: str, edit_mode_key: str):
    """
    Render inline metadata editor with enhanced capabilities.

    Args:
        row: Database row with current data
        table_type: Type of entry
        item_id: Entry ID
        edit_mode_key: Session state key for edit mode
    """
    from datetime import datetime
    import subprocess

    st.markdown("### ✏️ Edit Metadata")

    # Parse current metadata
    current_metadata = json_manager.parse_metadata(row["metadata"])

    # Get user name from session state
    default_author = st.session_state.get("user_name", "User")

    # Initialize session state for draft editing if not exists
    draft_notes_key = f"draft_notes_{item_id}"
    draft_custom_key = f"draft_custom_{item_id}"

    if draft_notes_key not in st.session_state:
        st.session_state[draft_notes_key] = list(current_metadata.get("notes", []))

    if draft_custom_key not in st.session_state:
        st.session_state[draft_custom_key] = dict(current_metadata.get("custom", {}))

    # Keys editor
    st.markdown("**🏷️ Keys**")
    current_keys_list = current_metadata.get("keys", [])

    # Get database connection to fetch all existing keys across this table type
    existing_keys = []
    try:
        repo_root = Path(__file__).parent.parent.parent.parent
        conn = db_manager.connect_db(repo_root / "results" / "portle.db")
        existing_keys = db_manager.get_existing_keys(conn, table_type)
        conn.close()
    except Exception as e:
        pass  # Silently fail if can't load existing keys

    # Multiselect for existing keys
    if existing_keys:
        st.caption(f"Select from existing {table_type} keys:")
        selected_existing_keys = st.multiselect(
            f"Existing {table_type} keys",
            options=existing_keys,
            default=[k for k in current_keys_list if k in existing_keys],
            key=f"keys_multiselect_{item_id}",
            label_visibility="collapsed",
            help=f"Select one or more existing keys used by other {table_type}s"
        )
    else:
        selected_existing_keys = []

    # Custom key input for new keys
    st.caption("Add custom keys (comma-separated):")
    custom_keys_list = [k for k in current_keys_list if k not in existing_keys]
    custom_keys_str = ", ".join(custom_keys_list)

    custom_keys_input = st.text_input(
        "Custom keys",
        value=custom_keys_str,
        placeholder="e.g., new-key, experimental",
        key=f"keys_custom_input_{item_id}",
        label_visibility="collapsed",
        help="Enter new custom keys separated by commas"
    )

    # Parse custom keys
    custom_keys_parsed = [k.strip() for k in custom_keys_input.split(",") if k.strip()]

    # Combine selected and custom keys
    all_keys = list(selected_existing_keys) + custom_keys_parsed

    # Show combined result
    if all_keys:
        st.success(f"✓ {len(all_keys)} key(s): {', '.join(all_keys)}")

    # Store combined keys for saving
    keys_input = ", ".join(all_keys)

    st.markdown("---")

    # Notes editor - display from draft state
    st.markdown("**📝 Notes**")

    # Display notes from draft state with delete buttons
    draft_notes = st.session_state[draft_notes_key]
    if draft_notes:
        for idx, note in enumerate(draft_notes):
            col1, col2 = st.columns([10, 1])
            with col1:
                st.text(f"📝 {note.get('date_added', 'N/A')} - {note.get('author', 'Unknown')}: {note.get('text', '')}")
            with col2:
                if st.button("❌", key=f"delete_note_{idx}_{item_id}", help="Delete this note", disabled=DEMO_MODE):
                    st.session_state[draft_notes_key].pop(idx)
                    st.rerun()
    else:
        st.caption("No notes")

    # Add new note section
    st.markdown("**Add New Note**")
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        new_note_author = st.text_input("Author", value=default_author, key=f"new_note_author_{item_id}")
    with col2:
        new_note_date = st.text_input("Date", value=datetime.now().strftime("%m-%d-%y"), key=f"new_note_date_{item_id}")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)

    new_note_text = st.text_area("Note text", placeholder="Enter note text...", height=80, key=f"new_note_text_{item_id}")

    if st.button("➕ Add Note", key=f"add_note_btn_{item_id}", type="secondary", disabled=DEMO_MODE):
        if new_note_text.strip():
            new_note = {
                "date_added": new_note_date,
                "author": new_note_author,
                "text": new_note_text.strip()
            }
            st.session_state[draft_notes_key].append(new_note)
            st.rerun()

    st.markdown("---")

    # Custom fields editor - display from draft state
    st.markdown("**⚙️ Custom Fields**")

    # Display custom fields from draft state with delete buttons
    draft_custom = st.session_state[draft_custom_key]
    if draft_custom:
        for field_key, field_value in draft_custom.items():
            col1, col2 = st.columns([10, 1])
            with col1:
                st.text(f"⚙️ **{field_key}**: {field_value}")
            with col2:
                if st.button("❌", key=f"delete_custom_{field_key}_{item_id}", help="Delete this field", disabled=DEMO_MODE):
                    del st.session_state[draft_custom_key][field_key]
                    st.rerun()
    else:
        st.caption("No custom fields")

    # Add new custom field section
    st.markdown("**Add New Custom Field**")
    col1, col2, col3 = st.columns([4, 4, 1])
    with col1:
        new_field_key = st.text_input("Field name", placeholder="e.g., difficulty", key=f"new_field_key_{item_id}")
    with col2:
        new_field_value = st.text_input("Field value", placeholder="e.g., hard", key=f"new_field_value_{item_id}")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)

    if st.button("➕ Add Field", key=f"add_field_btn_{item_id}", type="secondary", disabled=DEMO_MODE):
        if new_field_key.strip() and new_field_value.strip():
            st.session_state[draft_custom_key][new_field_key.strip()] = new_field_value.strip()
            st.rerun()

    st.markdown("---")

    # Save form
    with st.form(key=f"save_metadata_form_{item_id}"):
        # Options
        rebuild_db = st.checkbox(
            "🔨 Rebuild database after saving (recommended)",
            value=True,
            help="Automatically rebuild the database to sync changes"
        )

        # Submit buttons
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True, disabled=DEMO_MODE)
        with col2:
            cancel = st.form_submit_button("Cancel", use_container_width=True)

    # Handle form submission
    if cancel:
        # Clear draft state
        if draft_notes_key in st.session_state:
            del st.session_state[draft_notes_key]
        if draft_custom_key in st.session_state:
            del st.session_state[draft_custom_key]
        st.session_state[edit_mode_key] = False
        st.rerun()

    if submitted:
        # Parse keys
        new_keys = [k.strip() for k in keys_input.split(",") if k.strip()]

        # Use draft state for notes and custom fields
        final_notes = st.session_state.get(draft_notes_key, list(current_metadata.get("notes", [])))
        final_custom = st.session_state.get(draft_custom_key, dict(current_metadata.get("custom", {})))

        # Create updated metadata
        updated_metadata = {
            "keys": new_keys,
            "notes": final_notes,
            "custom": final_custom
        }

        # Save to JSON
        repo_root = Path(__file__).parent.parent.parent.parent

        # Get additional IDs from row if needed
        success = False

        try:
            if table_type == "agent":
                success = json_manager.update_agent_metadata(repo_root, item_id, updated_metadata)

            elif table_type == "benchmark":
                success = json_manager.update_benchmark_metadata(repo_root, item_id, updated_metadata)

            elif table_type == "dataset":
                benchmark_id = row["benchmark_id"]
                success = json_manager.update_dataset_metadata(
                    repo_root, benchmark_id, item_id, updated_metadata
                )

            elif table_type == "task":
                benchmark_id = row["benchmark_id"]
                success = json_manager.update_task_metadata(
                    repo_root, benchmark_id, item_id, updated_metadata
                )

            elif table_type == "run":
                benchmark_id = row["benchmark_id"]
                dataset_id = row["dataset_id"]
                success = json_manager.update_run_metadata(
                    repo_root, benchmark_id, dataset_id, item_id, updated_metadata
                )

            elif table_type == "datapoint":
                benchmark_id = row["benchmark_id"]
                dataset_id = row["dataset_id"]
                run_id = row["run_id"]
                success = json_manager.update_datapoint_metadata(
                    repo_root, benchmark_id, dataset_id, run_id, item_id, updated_metadata
                )

            if success:
                st.success("✅ Metadata saved successfully!")

                # Clear draft state
                if draft_notes_key in st.session_state:
                    del st.session_state[draft_notes_key]
                if draft_custom_key in st.session_state:
                    del st.session_state[draft_custom_key]

                # Rebuild database if requested
                if rebuild_db:
                    st.info("🔨 Rebuilding database...")
                    script_path = repo_root / "src" / "build_datatable.py"

                    try:
                        result = subprocess.run(
                            [sys.executable, str(script_path)],
                            cwd=str(repo_root),
                            capture_output=True,
                            text=True,
                            timeout=60
                        )

                        if result.returncode == 0:
                            st.success("✅ Database rebuilt successfully!")
                            st.session_state.db_sync_needed = False
                        else:
                            st.warning("⚠️ Database rebuild completed with warnings")
                            st.session_state.db_sync_needed = True
                            with st.expander("Build output"):
                                st.code(result.stdout + result.stderr)
                    except subprocess.TimeoutExpired:
                        st.error("❌ Database rebuild timed out")
                        st.session_state.db_sync_needed = True
                    except Exception as e:
                        st.error(f"❌ Database rebuild failed: {e}")
                        st.session_state.db_sync_needed = True
                else:
                    st.session_state.db_sync_needed = True

                st.balloons()
                # Exit edit mode
                st.session_state[edit_mode_key] = False
                st.rerun()
            else:
                st.error("❌ Failed to save metadata")

        except Exception as e:
            st.error(f"Error saving metadata: {e}")


def render_detail_panel(table_type: str, item_id: str):
    """
    Render detailed view of a database entry.

    Args:
        table_type: Type of entry (agent, benchmark, etc.)
        item_id: ID of the entry
    """
    try:
        conn = db_manager.connect_db(st.session_state.db_path)
        row = db_manager.query_entry(conn, table_type, item_id)

        if row is None:
            st.error(f"No {table_type} found with ID: {item_id}")
            conn.close()
            return

        # Get table info
        table_name, id_column = db_manager.TABLES[table_type]

        # Header
        st.subheader(f"{table_type.upper()}: {item_id}")

        # Edit button - toggle edit mode
        col1, col2 = st.columns([3, 1])

        # Initialize edit mode state
        edit_mode_key = f"edit_mode_{table_type}_{item_id}"
        if edit_mode_key not in st.session_state:
            st.session_state[edit_mode_key] = False

        with col2:
            if st.button(
                "📝 Edit Metadata" if not st.session_state[edit_mode_key] else "👁️ View Mode",
                key=f"edit_toggle_{item_id}",
                type="primary" if st.session_state[edit_mode_key] else "secondary",
                disabled=DEMO_MODE
            ):
                st.session_state[edit_mode_key] = not st.session_state[edit_mode_key]
                st.rerun()

        # Inline metadata editor (if in edit mode) - show at top for visibility
        if st.session_state[edit_mode_key]:
            render_inline_metadata_editor(row, table_type, item_id, edit_mode_key)
            st.markdown("---")

        # Display all fields
        with st.container(border=True):
            st.markdown("### Fields")

            for key in row.keys():
                if key == id_column:
                    continue  # Already displayed in header

                value = row[key]

                # Special handling for metadata
                if key == "metadata" and isinstance(value, str):
                    metadata = json_manager.parse_metadata(value)
                    render_metadata_section(metadata)

                # Special handling for agent_config
                elif key == "agent_config" and isinstance(value, str):
                    render_agent_config(value)

                # Regular fields
                else:
                    render_field_value(key, value)

        # Related entries
        relations = db_manager.get_related_entries(row, table_type)
        render_related_entries(relations, table_type)

        # Logs and Reports section (only for datapoints)
        if table_type == "datapoint":
            st.markdown("---")
            render_logs_and_reports(row, item_id)

        conn.close()

    except FileNotFoundError:
        st.error("Database not found")
    except Exception as e:
        st.error(f"Error loading entry: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())


def render():
    """Main render function for standalone detail view page."""
    st.title("📄 Entry Details")

    # Check if an entry is selected
    if st.session_state.selected_entry and st.session_state.selected_table:
        render_detail_panel(
            st.session_state.selected_table,
            st.session_state.selected_entry
        )
    else:
        st.info("No entry selected. Use the Search page to find and select an entry.")

        # Quick access form
        st.markdown("---")
        st.subheader("Quick Access")

        col1, col2 = st.columns(2)

        with col1:
            table_type = st.selectbox(
                "Entry Type",
                options=list(db_manager.TABLES.keys()),
                format_func=lambda x: x.capitalize()
            )

        with col2:
            item_id = st.text_input("Entry ID")

        if st.button("View Entry"):
            if item_id:
                st.session_state.selected_entry = item_id
                st.session_state.selected_table = table_type
                st.rerun()
            else:
                st.warning("Please enter an entry ID")


if __name__ == "__main__":
    # For testing component standalone
    render()
