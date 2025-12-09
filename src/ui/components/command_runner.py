"""
Command Runner Component for poRTLe UI

Provides interface for executing Python scripts and commands from the UI.
"""

import streamlit as st
from pathlib import Path
import sys
import subprocess
import json
import tempfile
import os
import re
import platform
import socket

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ui.utils import db_manager
from ui.utils.filter_engine import FilterSpec
from ui.components.heatmap import apply_task_filters
from ui.utils.process_manager import get_manager

# Demo mode - disables all data modification buttons when True
DEMO_MODE = os.environ.get("PORTLE_DEMO_MODE", "").lower() == "true"


def validate_keys(keys: list) -> tuple[bool, str]:
    """
    Validate that keys only contain alphanumeric characters, hyphens, and underscores.

    Args:
        keys: List of keys to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    for key in keys:
        if not re.match(r'^[a-zA-Z0-9_-]+$', key):
            return False, f"Invalid key '{key}': Keys can only contain letters, numbers, hyphens, and underscores (no spaces or special characters)"
    return True, ""


def load_agents_for_templates(repo_root: Path) -> list:
    """
    Load all agents from agents.json for use as templates.

    Args:
        repo_root: Path to repository root

    Returns:
        List of agent dictionaries with agent_id as key for easy lookup
    """
    agents_path = repo_root / "results" / "json" / "agents.json"
    agents = []

    try:
        if agents_path.exists():
            with open(agents_path, 'r') as f:
                agents = json.load(f)
    except Exception:
        pass  # Silently fail if agents.json doesn't exist or can't be parsed

    return agents


def get_agent_by_id(agents: list, agent_id: str) -> dict:
    """
    Get an agent by its agent_id.

    Args:
        agents: List of agent dictionaries
        agent_id: The agent_id to search for

    Returns:
        Agent dictionary or None if not found
    """
    for agent in agents:
        if agent.get("agent_id") == agent_id:
            return agent
    return None


def parse_existing_agent_ids(repo_root: Path) -> dict:
    """
    Parse existing agent_ids to extract unique agent names and backend models.

    Args:
        repo_root: Path to repository root

    Returns:
        Dictionary with 'agent_names' and 'backend_models' lists
    """
    agents_path = repo_root / "results" / "json" / "agents.json"
    agent_names = set()
    backend_models = set()

    try:
        if agents_path.exists():
            with open(agents_path, 'r') as f:
                agents_list = json.load(f)

            for agent_entry in agents_list:
                agent_id = agent_entry.get("agent_id", "")

                # Try to parse agent_id with __ separator
                if "__" in agent_id:
                    parts = agent_id.split("__", 1)
                    if len(parts) == 2:
                        agent_names.add(parts[0])
                        backend_models.add(parts[1])
                else:
                    # Legacy format - add as-is
                    agent_names.add(agent_id)

                # Also collect backend_model from config
                backend_model = agent_entry.get("agent_config", {}).get("backend_model")
                if backend_model and backend_model != "none":
                    backend_models.add(backend_model)

    except Exception:
        pass  # Silently fail if agents.json doesn't exist or can't be parsed

    return {
        "agent_names": sorted(list(agent_names)),
        "backend_models": sorted(list(backend_models))
    }


def _capture_command_output(command):
    """Return stripped stdout for a command or None if it cannot run."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def sanitize_hardware_info(hardware_info: str) -> str:
    """
    Sanitize hardware info for use in run IDs.

    - Replaces spaces and special characters with hyphens
    - Keeps only alphanumeric, hyphens, and underscores
    - Collapses multiple hyphens
    - Converts to lowercase

    Example: "Dakota's Kitchen" -> "dakota-s-kitchen"
    """
    import re
    # Replace spaces and special characters with hyphens
    sanitized = re.sub(r'[^a-zA-Z0-9_-]+', '-', hardware_info)
    # Remove leading/trailing hyphens and collapse multiple hyphens
    sanitized = re.sub(r'-+', '-', sanitized).strip('-')
    # Use lowercase for consistency
    sanitized = sanitized.lower()
    return sanitized


def detect_default_hardware_info() -> str:
    """Derive a friendly hardware description from the local machine."""
    override = os.environ.get("PORTLE_HARDWARE_INFO_DEFAULT")
    if override and override.strip():
        return override.strip()

    system = platform.system()
    friendly_name = None

    if system == "Darwin":
        friendly_name = _capture_command_output(["scutil", "--get", "ComputerName"])
        if not friendly_name:
            friendly_name = _capture_command_output(["scutil", "--get", "HostName"])
    elif system == "Linux":
        friendly_name = _capture_command_output(["hostnamectl", "--pretty"])
        if not friendly_name:
            friendly_name = _capture_command_output(["hostnamectl", "--static"])
    elif system == "Windows":
        friendly_name = os.environ.get("COMPUTERNAME")

    if not friendly_name:
        friendly_name = platform.node()

    if not friendly_name:
        friendly_name = socket.gethostname()

    friendly_name = (friendly_name or "").strip()
    return friendly_name or "local-machine"


DEFAULT_HARDWARE_INFO = detect_default_hardware_info()
# Sanitized version for display in forms
DEFAULT_HARDWARE_INFO_SANITIZED = sanitize_hardware_info(DEFAULT_HARDWARE_INFO)


def run_command_with_output(
    command: list,
    cwd: Path = None,
    description: str = "Running command..."
) -> tuple:
    """
    Run a command and stream output to Streamlit.

    Args:
        command: Command list (e.g., ['python', 'script.py'])
        cwd: Working directory
        description: Description for progress indicator

    Returns:
        Tuple of (success: bool, output: str)
    """
    output_placeholder = st.empty()
    output_lines = []

    with st.spinner(description):
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output
            for line in process.stdout:
                output_lines.append(line)
                output_placeholder.code('\n'.join(output_lines[-50:]))  # Show last 50 lines

            process.wait()

            full_output = '\n'.join(output_lines)

            if process.returncode == 0:
                return True, full_output
            else:
                return False, full_output

        except Exception as e:
            return False, f"Error: {str(e)}"


def run_command_background(
    command: list,
    cwd: Path = None,
    description: str = "Background command",
    metadata: dict = None
) -> str:
    """
    Run a command in the background (persistent execution).

    The command will continue running even if the UI closes.
    Use the Process Monitor to view status and logs.

    Args:
        command: Command list (e.g., ['python', 'script.py'])
        cwd: Working directory
        description: Description for the process
        metadata: Additional metadata to store with the process

    Returns:
        process_id: Unique identifier for tracking this process
    """
    manager = get_manager()
    process_id = manager.start_process(
        command=command,
        cwd=cwd,
        description=description,
        metadata=metadata
    )
    return process_id


def render_build_database_tab():
    """Render the Build Database tab."""
    st.markdown("### 🔨 Build Database from JSON Files")
    st.markdown("""
    Rebuild the SQLite database from all JSON files.
    This is required after editing metadata or adding new runs.
    """)

    repo_root = Path(__file__).parent.parent.parent.parent
    script_path = repo_root / "src" / "build_datatable.py"

    if not script_path.exists():
        st.error(f"Script not found: {script_path}")
        return

    st.info(f"Script: `{script_path.relative_to(repo_root)}`")

    # Button to trigger build
    if st.button("🔨 Build Database", type="primary", disabled=DEMO_MODE):
        st.markdown("---")
        st.subheader("Output:")

        command = [sys.executable, str(script_path)]
        success, output = run_command_with_output(
            command,
            cwd=repo_root,
            description="Building database..."
        )

        if success:
            st.success("✓ Database built successfully!")
            st.session_state.db_sync_needed = False
            st.balloons()
        else:
            st.error("✗ Database build failed")

        with st.expander("Full Output", expanded=True):
            st.code(output)


def render_add_agent_tab():
    """Render the Add LLM/Agent tab."""
    from datetime import datetime

    st.markdown("### 🤖 Add New LLM/Agent")
    st.markdown("""
    Add a new LLM or agent to the registry.
    Fill out the form below and it will be added to `results/json/agents.json`.
    """)

    repo_root = Path(__file__).parent.parent.parent.parent
    script_path = repo_root / "src" / "add_agent.py"

    if not script_path.exists():
        st.error(f"Script not found: {script_path}")
        return

    # Initialize session state for draft metadata
    draft_notes_key = "draft_notes_add_agent"
    draft_custom_key = "draft_custom_add_agent"

    if draft_notes_key not in st.session_state:
        st.session_state[draft_notes_key] = []

    if draft_custom_key not in st.session_state:
        st.session_state[draft_custom_key] = {}

    # Parse existing agent IDs for suggestions
    existing = parse_existing_agent_ids(repo_root)

    # Load all agents for template functionality
    all_agents = load_agents_for_templates(repo_root)
    agent_ids_for_template = ["[None - Start Fresh]"] + [a.get("agent_id", "") for a in all_agents if a.get("agent_id")]

    # Template selector
    st.markdown("---")
    st.subheader("📋 Use Existing Agent as Template")

    col_template, col_clear = st.columns([4, 1])

    with col_template:
        template_selection = st.selectbox(
            "Select an existing agent to use as a template",
            options=agent_ids_for_template,
            key="add_agent_template_selector",
            help="Choose an existing agent to pre-fill all fields. You can then modify the values as needed."
        )

    with col_clear:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Clear", key="clear_template_btn", help="Clear template and reset all fields"):
            # Clear all session state related to the form
            st.session_state["last_template_applied"] = None
            st.session_state["add_agent_template_selector"] = "[None - Start Fresh]"
            st.session_state["agent_name_selectbox"] = "[Create New...]"
            st.session_state["agent_name_custom"] = ""
            st.session_state["backend_model_selectbox"] = "[Create New...]"
            st.session_state["backend_model_custom"] = ""
            st.session_state["about_input"] = ""
            st.session_state["agent_folder_input"] = ""
            st.session_state["custom_config_input"] = ""
            st.session_state["agent_metadata_keys"] = ""
            st.session_state[draft_notes_key] = []
            st.session_state[draft_custom_key] = {}
            st.session_state["llm_mode_initialized"] = False
            st.rerun()

    # Handle template selection
    if template_selection != "[None - Start Fresh]":
        template_agent = get_agent_by_id(all_agents, template_selection)
        if template_agent:
            # Check if we need to apply the template (only when selection changes)
            if st.session_state.get("last_template_applied") != template_selection:
                # Parse agent_id to extract agent name and backend model
                agent_id = template_agent.get("agent_id", "")
                if "__" in agent_id:
                    parts = agent_id.split("__", 1)
                    template_agent_name = parts[0]
                    template_backend_model = parts[1]
                else:
                    template_agent_name = agent_id
                    template_backend_model = template_agent.get("agent_config", {}).get("backend_model", "")

                # Get template values
                template_about = template_agent.get("about", "")
                template_agent_folder = template_agent.get("agent_config", {}).get("agent_folder_path", "")
                template_custom_config = template_agent.get("agent_config", {}).get("custom", {})
                template_metadata = template_agent.get("metadata", {})
                template_keys = template_metadata.get("keys", [])
                template_notes = template_metadata.get("notes", [])
                template_custom_metadata = template_metadata.get("custom", {})

                # Apply template values to session state
                st.session_state["agent_name_selectbox"] = "[Create New...]"
                st.session_state["agent_name_custom"] = template_agent_name
                st.session_state["backend_model_selectbox"] = "[Create New...]"
                st.session_state["backend_model_custom"] = template_backend_model
                st.session_state["about_input"] = template_about
                st.session_state["agent_folder_input"] = template_agent_folder
                st.session_state["custom_config_input"] = json.dumps(template_custom_config, indent=2) if template_custom_config else ""
                st.session_state["agent_metadata_keys"] = ", ".join(template_keys) if template_keys else ""
                st.session_state[draft_notes_key] = template_notes.copy() if template_notes else []
                st.session_state[draft_custom_key] = template_custom_metadata.copy() if template_custom_metadata else {}

                # Mark template as applied
                st.session_state["last_template_applied"] = template_selection
                st.session_state["llm_mode_initialized"] = False  # Reset LLM mode when using template

                st.success(f"✅ Template applied from `{template_selection}`. Modify the fields below as needed.")
                st.rerun()
            else:
                st.info(f"📋 Using template from `{template_selection}`. Modify the fields below as needed.")
    else:
        # Clear template tracking when "None" is selected
        if st.session_state.get("last_template_applied"):
            st.session_state["last_template_applied"] = None

    # Setup mode toggle - LLM or Agentic
    st.markdown("---")
    st.subheader("Setup Mode")

    setup_mode = st.radio(
        "Choose setup type",
        options=["🧠 LLM Setup", "🤖 Agent Setup"],
        help="LLM Setup: Configure an LLM-only agent (no Docker container). Agentic Setup: Configure a full agent with custom folder.",
        horizontal=True,
        key="add_agent_setup_mode"
    )

    is_llm_mode = setup_mode == "🧠 LLM Setup"

    # Apply LLM setup defaults when mode is selected (but not when a template is being used)
    if is_llm_mode:
        # Set defaults for LLM mode if not already set and no template is actively applied
        template_active = st.session_state.get("last_template_applied") is not None
        if not template_active and ("llm_mode_initialized" not in st.session_state or not st.session_state["llm_mode_initialized"]):
            st.session_state["agent_name_selectbox"] = "[Create New...]"
            st.session_state["agent_name_custom"] = "llm"
            st.session_state["backend_model_selectbox"] = "[Create New...]"
            st.session_state["backend_model_custom"] = ""
            st.session_state["agent_folder_input"] = "none"
            st.session_state["agent_metadata_keys"] = "llm"
            st.session_state["custom_config_input"] = '{\n  "cvdp_llm_name": "<enter_model_name>"\n}'
            st.session_state["llm_mode_initialized"] = True
            st.rerun()

        st.info("💡 **LLM Mode:** Agent name is fixed to 'llm'. Agent folder is set to 'none'. Configure the backend model and custom config below.")
    else:
        # Clear LLM mode initialization when switching to Agentic mode (but preserve template values)
        if st.session_state.get("llm_mode_initialized", False):
            st.session_state["llm_mode_initialized"] = False
            # Only clear the pre-filled values if no template is active
            if not st.session_state.get("last_template_applied"):
                st.session_state["agent_name_custom"] = ""
                st.session_state["backend_model_custom"] = ""
                st.session_state["agent_folder_input"] = ""
                st.session_state["agent_metadata_keys"] = ""
                st.session_state["custom_config_input"] = ""
            st.rerun()

    st.markdown("---")

    # Agent Configuration (outside form)
    st.subheader("Agent Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Agent name with autocomplete
        st.markdown("**Agent Name** *")

        if is_llm_mode:
            # In LLM mode, agent name is fixed to "llm" and uneditable
            agent_name = st.text_input(
                "Agent Name (fixed for LLM mode)",
                value="llm",
                disabled=True,
                key="agent_name_display",
                label_visibility="collapsed",
                help="Agent name is fixed to 'llm' in LLM Setup mode"
            )
        else:
            # In Agentic mode, allow selection/creation
            agent_name_options = ["[Create New...]"] + existing["agent_names"]
            agent_name_selection = st.selectbox(
                "Select existing or create new",
                options=agent_name_options,
                key="agent_name_selectbox",
                label_visibility="collapsed",
                help="Select an existing agent name or create a new one"
            )

            if agent_name_selection == "[Create New...]":
                agent_name = st.text_input(
                    "Enter new agent name",
                    placeholder="e.g., opencode, react-agent, codex",
                    key="agent_name_custom",
                    label_visibility="collapsed"
                )
            else:
                agent_name = agent_name_selection
                st.caption(f"Using existing: {agent_name}")

    with col2:
        # Backend model with autocomplete
        st.markdown("**Backend Model** *")
        backend_model_options = ["[Create New...]"] + existing["backend_models"]
        backend_model_selection = st.selectbox(
            "Select existing or create new",
            options=backend_model_options,
            key="backend_model_selectbox",
            label_visibility="collapsed",
            help="Select an existing model or enter a new one"
        )

        if backend_model_selection == "[Create New...]":
            backend_model = st.text_input(
                "Enter new backend model",
                placeholder="e.g., gpt-4, claude-3, local-model",
                key="backend_model_custom",
                label_visibility="collapsed"
            )
        else:
            backend_model = backend_model_selection
            st.caption(f"Using existing: {backend_model}")

    # Validate that neither field contains __
    has_error = False
    if agent_name and "__" in agent_name:
        st.error("❌ Agent Name cannot contain double underscore (__)")
        has_error = True
    if backend_model and "__" in backend_model:
        st.error("❌ Backend Model cannot contain double underscore (__)")
        has_error = True

    # Show computed agent_id if both fields are valid
    if agent_name and backend_model and not has_error:
        computed_agent_id = f"{agent_name}__{backend_model}"
        st.info(f"🆔 **Generated Agent ID:** `{computed_agent_id}`")

        # Check if this combination already exists
        if existing["agent_names"] and existing["backend_models"]:
            agents_path = repo_root / "results" / "json" / "agents.json"
            try:
                if agents_path.exists():
                    with open(agents_path, 'r') as f:
                        agents_list = json.load(f)
                    existing_ids = [a.get("agent_id", "") for a in agents_list]
                    if computed_agent_id in existing_ids:
                        st.warning(f"⚠️ Agent ID `{computed_agent_id}` already exists!")
            except Exception:
                pass

    about = st.text_area(
        "Description *",
        placeholder="Brief description of the agent",
        help="What does this agent do?",
        key="about_input"
    )

    agent_folder = st.text_input(
        "Agent Folder Path *",
        placeholder="e.g., agents/example-agent, or 'none' for LLM-only",
        help="Path to agent code (relative to repo root), or 'none' for LLM-only agents",
        key="agent_folder_input"
    )

    st.markdown("**Custom Configuration (optional)**")
    st.caption("Add custom key-value pairs for agent-specific config")

    custom_config = st.text_area(
        "Custom Config (JSON format)",
        placeholder='{\n  "temperature": 0.7,\n  "max_tokens": 1000\n}',
        help="Additional configuration as JSON. In LLM mode, make sure to update the 'cvdp_llm_name' field with your model name.",
        height=100,
        key="custom_config_input"
    )

    if is_llm_mode:
        st.success("⚡ **LLM Setup Mode Active:** Agent name set to 'llm', folder set to 'none'. Update 'cvdp_llm_name' in Custom Config above with your backend model name.")
    else:
        st.info("**Note:** To register an LLM for CVDP (LLM Mode/force-copilot), use **LLM Setup** mode above.")

    st.markdown("---")

    # Metadata section OUTSIDE form for interactive editing
    with st.expander("📋 Metadata (optional)", expanded=False):
        st.markdown("Add metadata to categorize and document this agent.")

        # Get user name from session state
        default_author = st.session_state.get("user_name", "User")

        # Keys editor
        st.markdown("**🏷️ Keys**")
        metadata_keys = st.text_input(
            "Keys (comma-separated)",
            placeholder="e.g., production, experimental, high-performance, llm",
            help="Add keys to categorize this agent",
            key="agent_metadata_keys"
        )

        if is_llm_mode and "llm" not in metadata_keys:
            st.info("💡 'llm' key has been added automatically in LLM Setup mode")

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
                    if st.button("❌", key=f"delete_agent_note_{idx}", help="Delete this note", disabled=DEMO_MODE):
                        st.session_state[draft_notes_key].pop(idx)
                        st.rerun()
        else:
            st.caption("No notes")

        # Add new note section
        st.markdown("**Add New Note**")
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            new_note_author = st.text_input("Author", value=default_author, key="agent_new_note_author")
        with col2:
            new_note_date = st.text_input("Date", value=datetime.now().strftime("%m-%d-%y"), key="agent_new_note_date")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)

        new_note_text = st.text_area("Note text", placeholder="Enter note text...", height=80, key="agent_new_note_text")

        if st.button("➕ Add Note", key="agent_add_note_btn", type="secondary", disabled=DEMO_MODE):
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
        st.markdown("**⚙️ Custom Metadata Fields**")

        # Display custom fields from draft state with delete buttons
        draft_custom = st.session_state[draft_custom_key]
        if draft_custom:
            for field_key, field_value in draft_custom.items():
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.text(f"⚙️ **{field_key}**: {field_value}")
                with col2:
                    if st.button("❌", key=f"delete_agent_custom_{field_key}", help="Delete this field", disabled=DEMO_MODE):
                        del st.session_state[draft_custom_key][field_key]
                        st.rerun()
        else:
            st.caption("No custom metadata fields")

        # Add new custom field section
        st.markdown("**Add New Custom Metadata Field**")
        col1, col2, col3 = st.columns([4, 4, 1])
        with col1:
            new_field_key = st.text_input("Field name", placeholder="e.g., version", key="agent_new_field_key")
        with col2:
            new_field_value = st.text_input("Field value", placeholder="e.g., 1.0.0", key="agent_new_field_value")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)

        if st.button("➕ Add Field", key="agent_add_field_btn", type="secondary", disabled=DEMO_MODE):
            if new_field_key.strip() and new_field_value.strip():
                st.session_state[draft_custom_key][new_field_key.strip()] = new_field_value.strip()
                st.rerun()

    st.markdown("---")

    # Form with submit button
    with st.form("add_agent_form"):
        submitted = st.form_submit_button("🤖 Add Agent", type="primary", disabled=DEMO_MODE)

        if submitted:
            # Validate required fields
            if not all([agent_name, about, backend_model, agent_folder]):
                st.error("Please fill all required fields (*)")
            elif "__" in agent_name or "__" in backend_model:
                st.error("Agent Name and Backend Model cannot contain double underscore (__)")
            else:
                # Construct final agent_id
                agent_id = f"{agent_name}__{backend_model}"
                st.markdown("---")
                st.subheader("Adding Agent")

                # Parse custom config
                try:
                    custom = json.loads(custom_config) if custom_config.strip() else {}
                except json.JSONDecodeError:
                    st.error("Invalid JSON in custom config")
                    return

                # Parse metadata from session state
                keys = [k.strip() for k in st.session_state.get("agent_metadata_keys", "").split(",") if k.strip()]

                # Validate keys
                is_valid, error_msg = validate_keys(keys)
                if not is_valid:
                    st.error(error_msg)
                    return

                notes = st.session_state.get(draft_notes_key, [])
                custom_metadata = st.session_state.get(draft_custom_key, {})

                # Build command arguments
                command = [
                    sys.executable,
                    str(script_path),
                    "--agent-id", agent_id,
                    "--about", about,
                    "--backend-model", backend_model,
                    "--agent-folder", agent_folder,
                    "--custom-config", json.dumps(custom),
                    "--metadata-keys", json.dumps(keys),
                    "--metadata-notes", json.dumps(notes),
                    "--metadata-custom", json.dumps(custom_metadata)
                ]

                # Run add_agent.py script
                success, output = run_command_with_output(
                    command,
                    cwd=repo_root,
                    description=f"Adding agent '{agent_id}'..."
                )

                if success:
                    st.success(f"✓ Agent '{agent_id}' added successfully!")
                    st.info(f"🆔 **Agent ID:** `{agent_id}` (composed from `{agent_name}` + `{backend_model}`)")

                    # Clear draft metadata, mode initialization, and template tracking after successful add
                    st.session_state[draft_notes_key] = []
                    st.session_state[draft_custom_key] = {}
                    st.session_state["llm_mode_initialized"] = False
                    st.session_state["last_template_applied"] = None

                    # Rebuild database to sync new agent data
                    st.info("🔨 Rebuilding database...")
                    db_script_path = repo_root / "src" / "build_datatable.py"

                    database_rebuilt = False
                    try:
                        db_result = subprocess.run(
                            [sys.executable, str(db_script_path)],
                            cwd=str(repo_root),
                            capture_output=True,
                            text=True,
                            timeout=60
                        )

                        if db_result.returncode == 0:
                            st.success("✅ Database rebuilt successfully!")
                            st.session_state.db_sync_needed = False
                            database_rebuilt = True
                        else:
                            st.warning("⚠️ Database rebuild completed with warnings")
                            st.session_state.db_sync_needed = True
                            with st.expander("Database build output"):
                                st.code(db_result.stdout + db_result.stderr)
                    except subprocess.TimeoutExpired:
                        st.error("❌ Database rebuild timed out")
                        st.session_state.db_sync_needed = True
                    except Exception as e:
                        st.error(f"❌ Database rebuild failed: {e}")
                        st.session_state.db_sync_needed = True

                    # Only celebrate if database rebuild succeeded
                    if database_rebuilt:
                        st.balloons()
                    else:
                        st.warning("⚠️ Agent added but database rebuild incomplete.")
                else:
                    st.error("✗ Failed to add agent")

                with st.expander("Full Output", expanded=True):
                    st.code(output)


def render_task_advanced_filters(conn, benchmark_id: str, dataset_id: str):
    """
    Render advanced filters for task selection.

    Args:
        conn: Database connection
        benchmark_id: Benchmark ID for context
        dataset_id: Dataset ID for context

    Returns:
        FilterSpec object if filters are defined, None otherwise
    """
    with st.expander("🎯 Advanced Filters", expanded=False):
        st.markdown(
            "Build filters for task metadata and custom fields. "
            "Filter by difficulty, keys, custom metadata, etc."
        )

        # Initialize session state for task filters
        if "task_field_filters" not in st.session_state:
            st.session_state.task_field_filters = []
        if "task_metadata_filters" not in st.session_state:
            st.session_state.task_metadata_filters = []
        if "task_metadata_keys" not in st.session_state:
            st.session_state.task_metadata_keys = []

        # Get available columns for tasks
        try:
            tbl_columns = db_manager.get_table_columns(conn, "task", exclude_json=True)
            column_names = [col['name'] for col in tbl_columns]

            # Add common metadata paths for tasks
            metadata_options = [
                "metadata.keys",
                "metadata.custom.source_task_id",
                "metadata.custom.cvdp_difficulty",  # CVDP-specific
                "metadata.custom.cvdp_cid",  # CVDP-specific
                "metadata.custom.<field>",  # Placeholder for any custom field
            ]
            column_names.extend(metadata_options)
        except Exception:
            column_names = []

        # Display current filters
        st.markdown("#### Active Filters")
        all_filters = []
        for f in st.session_state.task_field_filters:
            all_filters.append(("field", f))
        for m in st.session_state.task_metadata_filters:
            all_filters.append(("metadata", m))

        if all_filters:
            filters_to_remove = []
            tag_filters_removed = []
            cols = st.columns(4)
            for i, (filter_type, f) in enumerate(all_filters):
                display_value = ", ".join(f["value"]) if isinstance(f["value"], list) else str(f["value"])
                field_name = f["field"] if filter_type == "field" else f"metadata.{f['path']}"
                filter_text = f"{field_name} {f['op']} {display_value}"

                with cols[i % 4]:
                    if st.button(filter_text, key=f"remove_task_filter_{i}", type="secondary"):
                        filters_to_remove.append((filter_type, f))

            if filters_to_remove:
                for filter_type, f in filters_to_remove:
                    if filter_type == "field":
                        st.session_state.task_field_filters.remove(f)
                    else:
                        st.session_state.task_metadata_filters.remove(f)
                        if f.get("path") == "keys" and f.get("op") in ("contains", "has_key"):
                            tag_filters_removed.append(f.get("value"))
                if tag_filters_removed:
                    updated_tags = [
                        t for t in st.session_state.task_metadata_keys
                        if t not in tag_filters_removed
                    ]
                    st.session_state.task_metadata_keys = updated_tags
                    if "task_tag_multiselect" in st.session_state:
                        st.session_state["task_tag_multiselect"] = updated_tags
                st.rerun()
        else:
            st.caption("No filters applied")

        # Add new filter
        st.markdown("**Add Filter:**")
        if column_names:
            st.caption(f"📋 Available: {', '.join(column_names[:10])}{'...' if len(column_names) > 10 else ''}")

        col1, col2, col3, col4 = st.columns([2, 1, 2, 0.5])

        with col1:
            new_field = st.text_input(
                "Field or metadata path",
                placeholder="e.g., metadata.custom.source_task_id",
                key="new_task_field_name",
                label_visibility="collapsed"
            )
        with col2:
            new_op = st.selectbox(
                "Operator",
                options=["==", "!=", ">", "<", ">=", "<=", "contains", "in"],
                key="new_task_field_op",
                label_visibility="collapsed"
            )
        with col3:
            new_value = st.text_input(
                "Value",
                placeholder="e.g., easy or a,b,c for 'in'",
                key="new_task_field_value",
                label_visibility="collapsed"
            )
        with col4:
            if st.button("Add", key="add_task_field_filter", disabled=not (new_field and new_value)):
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

                # Check if metadata path
                if new_field.startswith("metadata."):
                    metadata_path = new_field[len("metadata."):]
                    st.session_state.task_metadata_filters.append({
                        "path": metadata_path,
                        "op": new_op,
                        "value": parsed_value
                    })
                else:
                    st.session_state.task_field_filters.append({
                        "field": new_field,
                        "op": new_op,
                        "value": parsed_value
                    })
                st.rerun()

        # Quick Tags for metadata.keys
        try:
            available_task_tags = db_manager.get_existing_keys(conn, "task")
        except Exception:
            available_task_tags = []

        st.markdown("#### Quick Tags")
        st.caption("Tap tags to add `metadata.keys contains <tag>` filters for tasks")

        existing_tag_filters = [
            f.get("value")
            for f in st.session_state.task_metadata_filters
            if f.get("path") == "keys" and f.get("op") in ("contains", "has_key")
        ]
        current_tag_selection = list(st.session_state.task_metadata_keys)
        for tag in existing_tag_filters:
            if isinstance(tag, str) and tag not in current_tag_selection:
                current_tag_selection.append(tag)

        if available_task_tags:
            current_tag_selection = [t for t in current_tag_selection if t in available_task_tags]
            st.session_state.task_metadata_keys = current_tag_selection
            previous_tags = set(current_tag_selection)

            selected_tags = st.multiselect(
                "Quick Tags for Tasks",
                options=available_task_tags,
                default=current_tag_selection,
                key="task_tag_multiselect",
                help="Adds a metadata.keys contains <tag> filter for each selected tag",
                label_visibility="collapsed"
            )
            st.session_state.task_metadata_keys = selected_tags

            added_tags = set(selected_tags) - previous_tags
            removed_tags = previous_tags - set(selected_tags)

            if added_tags:
                for tag in added_tags:
                    filter_dict = {"path": "keys", "op": "contains", "value": tag}
                    if filter_dict not in st.session_state.task_metadata_filters:
                        st.session_state.task_metadata_filters.append(filter_dict)

            if removed_tags:
                st.session_state.task_metadata_filters = [
                    f for f in st.session_state.task_metadata_filters
                    if not (
                        f.get("path") == "keys"
                        and f.get("op") in ("contains", "has_key")
                        and f.get("value") in removed_tags
                    )
                ]

            if added_tags or removed_tags:
                st.rerun()
        else:
            st.session_state.task_metadata_keys = []
            st.caption("No tags found for tasks yet.")

        # Clear all button
        if all_filters:
            if st.button("🗑️ Clear All Filters", key="clear_task_filters"):
                st.session_state.task_field_filters = []
                st.session_state.task_metadata_filters = []
                st.session_state.task_metadata_keys = []
                st.rerun()

        # Build FilterSpec if filters exist
        if st.session_state.task_field_filters or st.session_state.task_metadata_filters:
            return FilterSpec(
                field_filters=st.session_state.task_field_filters,
                metadata_filters=st.session_state.task_metadata_filters,
                logic="AND"  # All filters must match
            )

        return None


def render_run_benchmark_tab():
    """Render the Run Benchmark tab."""
    from datetime import datetime

    st.markdown("### 🏃 Run Benchmark")
    st.markdown("""
    Execute a benchmark run using `run_dataset.py`.
    Configure the run parameters below.
    """)

    repo_root = Path(__file__).parent.parent.parent.parent
    script_path = repo_root / "src" / "run_dataset.py"

    if not script_path.exists():
        st.error(f"Script not found: {script_path}")
        return

    # Initialize session state for draft metadata
    draft_notes_key = "draft_notes_run_benchmark"
    draft_custom_key = "draft_custom_run_benchmark"

    if draft_notes_key not in st.session_state:
        st.session_state[draft_notes_key] = []

    if draft_custom_key not in st.session_state:
        st.session_state[draft_custom_key] = {}

    try:
        conn = db_manager.connect_db(st.session_state.db_path)

        # Benchmark and Dataset selectors OUTSIDE form so they can update dynamically
        st.subheader("Run Configuration")

        col1, col2 = st.columns(2)

        with col1:
            # Benchmark selector
            benchmarks = db_manager.get_benchmarks(conn)
            if benchmarks:
                benchmark = st.selectbox("Benchmark *", benchmarks, key="run_benchmark_selector")
            else:
                st.error("No benchmarks found")
                benchmark = None

        with col2:
            # Dataset selector - updates when benchmark changes
            if benchmark:
                datasets = db_manager.get_datasets(conn, benchmark)
                if datasets:
                    dataset = st.selectbox("Dataset *", datasets, key="run_dataset_selector")
                else:
                    st.error(f"No datasets for {benchmark}")
                    dataset = None
            else:
                dataset = None
                st.info("Select a benchmark first")

        # Detect if this benchmark uses CVDP adapter
        uses_cvdp = False
        from benchmarks import BenchmarkRegistry
        if benchmark:
            try:
                adapter = BenchmarkRegistry.get_adapter(benchmark.lower())
                uses_cvdp = adapter.get_benchmark_runner_dir() == "cvdp_benchmark"
            except KeyError:
                uses_cvdp = False
            # # Check if benchmark name suggests CVDP adapter
            # benchmark_lower = benchmark.lower().replace("_", "").replace("-", "")
            # if "cvdp" in benchmark_lower:
            #     uses_cvdp = True

        # Task selector - optional task filtering
        st.subheader("Task Selection (optional)")
        if benchmark and dataset:
            # Get all tasks for this dataset
            all_tasks = db_manager.get_tasks(conn, benchmark_id=benchmark, dataset_id=dataset)

            if all_tasks:
                # Quick Selection - Manual multiselect
                st.markdown("#### Quick Selection")
                st.caption("Manually select specific tasks to run")

                quick_selection_key = "run_benchmark_quick_selection"
                if quick_selection_key not in st.session_state:
                    st.session_state[quick_selection_key] = []

                # Filter session state to only include valid tasks from current dataset
                valid_defaults = [t for t in st.session_state[quick_selection_key] if t in all_tasks]

                quick_selected_tasks = st.multiselect(
                    "Select tasks",
                    options=all_tasks,
                    default=valid_defaults,
                    key="run_benchmark_task_multiselect",
                    help="Directly select specific tasks to run",
                    label_visibility="collapsed"
                )
                st.session_state[quick_selection_key] = quick_selected_tasks

                if quick_selected_tasks:
                    st.success(f"✓ {len(quick_selected_tasks)} task(s) selected via Quick Selection")

                # Advanced Filters
                task_filter_spec = render_task_advanced_filters(conn, benchmark, dataset)

                # Determine final task selection
                selected_tasks = None

                if quick_selected_tasks:
                    # Quick Selection takes precedence
                    selected_tasks = quick_selected_tasks
                    st.info(f"💡 Using Quick Selection: {len(selected_tasks)} tasks")
                elif task_filter_spec:
                    # Apply advanced filters
                    filtered_task_ids = apply_task_filters(conn, task_filter_spec)
                    selected_tasks = [tid for tid in all_tasks if tid in filtered_task_ids]
                    st.info(f"🎯 Using Advanced Filters: {len(selected_tasks)}/{len(all_tasks)} tasks match")
                else:
                    # No selection - will run all tasks
                    selected_tasks = None
                    st.info(f"ℹ️ No filters applied - will run all {len(all_tasks)} tasks")

            else:
                st.info(f"No tasks found for dataset {dataset}")
                selected_tasks = None
        else:
            st.info("Select a benchmark and dataset to choose specific tasks")
            selected_tasks = None

        st.markdown("---")

        # Metadata section OUTSIDE form for interactive editing
        with st.expander("📋 Metadata (optional)", expanded=False):
            st.markdown("Add metadata to categorize and document this benchmark run.")

            # Get user name from session state
            default_author = st.session_state.get("user_name", "User")

            # Keys editor
            st.markdown("**🏷️ Keys**")
            metadata_keys = st.text_input(
                "Keys (comma-separated)",
                value="ui-generated",
                placeholder="e.g., experiment-1, production, verified",
                help="Add keys to categorize this run",
                key="run_metadata_keys"
            )

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
                        if st.button("❌", key=f"delete_run_note_{idx}", help="Delete this note", disabled=DEMO_MODE):
                            st.session_state[draft_notes_key].pop(idx)
                            st.rerun()
            else:
                st.caption("No notes")

            # Add new note section
            st.markdown("**Add New Note**")
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                new_note_author = st.text_input("Author", value=default_author, key="run_new_note_author")
            with col2:
                new_note_date = st.text_input("Date", value=datetime.now().strftime("%m-%d-%y"), key="run_new_note_date")
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)

            new_note_text = st.text_area("Note text", placeholder="Enter note text...", height=80, key="run_new_note_text")

            if st.button("➕ Add Note", key="run_add_note_btn", type="secondary", disabled=DEMO_MODE):
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
                        if st.button("❌", key=f"delete_run_custom_{field_key}", help="Delete this field", disabled=DEMO_MODE):
                            del st.session_state[draft_custom_key][field_key]
                            st.rerun()
            else:
                st.caption("No custom fields")

            # Add new custom field section
            st.markdown("**Add New Custom Field**")
            col1, col2, col3 = st.columns([4, 4, 1])
            with col1:
                new_field_key = st.text_input("Field name", placeholder="e.g., difficulty", key="run_new_field_key")
            with col2:
                new_field_value = st.text_input("Field value", placeholder="e.g., hard", key="run_new_field_value")
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)

            if st.button("➕ Add Field", key="run_add_field_btn", type="secondary", disabled=DEMO_MODE):
                if new_field_key.strip() and new_field_value.strip():
                    st.session_state[draft_custom_key][new_field_key.strip()] = new_field_value.strip()
                    st.rerun()

        add_debug_key = st.checkbox(
            "Add Debug Key to Run",
            value=st.session_state.get("run_add_debug_key", False),
            help="Adds a 'debug' tag to this run's metadata.keys for easier filtering",
            key="run_add_debug_key"
        )

        st.markdown("---")

        # Mode selector - only show for CVDP benchmarks
        if uses_cvdp:
            st.subheader("Execution Mode")
            execution_mode = st.radio(
                "Choose execution mode",
                options=["Agent Mode", "LLM Mode (force-copilot)"],
                help="Agent Mode: Run with Docker agent. LLM Mode: Run model in copilot mode (uses cvdp_llm_name from agent custom config)",
                horizontal=True,
                key="run_benchmark_execution_mode"
            )
            is_llm_mode = execution_mode == "LLM Mode (force-copilot)"
        else:
            # Non-CVDP benchmarks always use agent mode
            is_llm_mode = False
            st.info("ℹ️ This benchmark uses standard agent execution")

        st.markdown("---")

        # Agent selector OUTSIDE form so it updates dynamically
        agents = db_manager.get_agents(conn)
        if agents:
            help_text = "Select agent to run"
            if is_llm_mode:
                help_text += " (must have cvdp_llm_name in custom config for LLM mode)"

            # Set default to first LLM agent if in LLM mode, otherwise first agent
            default_index = 0
            if is_llm_mode:
                # Find first LLM agent (has cvdp_llm_name)
                try:
                    agents_path = repo_root / "results" / "json" / "agents.json"
                    with open(agents_path, 'r') as f:
                        agents_list = json.load(f)
                    for i, agent_id in enumerate(agents):
                        agent_config = next((a for a in agents_list if a["agent_id"] == agent_id), None)
                        if agent_config and agent_config.get("agent_config", {}).get("custom", {}).get("cvdp_llm_name"):
                            default_index = i
                            break
                except Exception:
                    pass

            selected_agent = st.selectbox(
                "Agent *",
                agents,
                index=default_index,
                help=help_text,
                key="run_benchmark_agent_selector"
            )
        else:
            st.error("No agents found")
            selected_agent = None

        # Validate agent requirements and show feedback
        if selected_agent:
            agents_path = repo_root / "results" / "json" / "agents.json"
            try:
                with open(agents_path, 'r') as f:
                    agents_list = json.load(f)
                agent_config = next((a for a in agents_list if a["agent_id"] == selected_agent), None)

                if agent_config:
                    if is_llm_mode:
                        # LLM Mode validation - check for cvdp_llm_name
                        cvdp_llm_name = agent_config.get("agent_config", {}).get("custom", {}).get("cvdp_llm_name")
                        if not cvdp_llm_name:
                            st.error(f"⚠️ Agent '{selected_agent}' does not have 'cvdp_llm_name' in agent_config.custom. Required for LLM Mode.")
                        else:
                            st.success(f"✓ Will use model: **{cvdp_llm_name}**")
                    else:
                        # Agent Mode validation - check for valid agent_folder_path
                        agent_folder_path = agent_config.get("agent_config", {}).get("agent_folder_path")

                        if not agent_folder_path or agent_folder_path == "none":
                            st.error(f"⚠️ Agent '{selected_agent}' does not have a valid agent folder. This agent is for LLM-only mode.")
                        else:
                            # Check if the folder exists
                            agent_dir = repo_root / agent_folder_path
                            if not agent_dir.exists():
                                st.error(f"⚠️ Agent folder does not exist: `{agent_folder_path}`")
                            else:
                                st.success(f"✓ Agent folder: `{agent_folder_path}`")
            except Exception as e:
                st.warning(f"Could not validate agent configuration: {e}")

        # Rest of the form
        with st.form("run_benchmark_form"):

            col1, col2 = st.columns(2)

            with col1:
                # Use the selected agent from above
                agent = selected_agent
                st.info(f"Selected agent: **{agent}**" if agent else "No agent selected")

            with col2:
                # Hardware info
                hardware = st.text_input(
                    "Hardware Info *",
                    value=DEFAULT_HARDWARE_INFO_SANITIZED,
                    help="Identifier for the hardware running this benchmark (sanitized)"
                )

            col3, col4 = st.columns(2)

            with col3:
                n_runs = st.number_input("Number of runs", min_value=1, value=1)

            with col4:
                threads = st.number_input("Threads", min_value=1, value=1)

            st.markdown("---")

            # Background execution option
            run_in_background = st.checkbox(
                "🔄 Run in background (persistent execution)",
                value=True,
                help="Run the benchmark in the background. It will continue even if you close this window. "
                     "You can monitor progress in the Process Monitor tab."
            )

            if run_in_background:
                st.info("💡 The benchmark will run in the background. Use the **Process Monitor** tab to view progress and logs.")

            submitted = st.form_submit_button("🏃 Run Benchmark", type="primary", disabled=DEMO_MODE)

            if submitted:
                # Validate required fields
                if not all([benchmark, dataset, agent, hardware]):
                    st.error("Please fill all required fields (benchmark, dataset, agent, hardware)")
                    return

                st.markdown("---")
                st.subheader("Benchmark Execution")

                # Parse metadata from session state
                keys = [k.strip() for k in st.session_state.get("run_metadata_keys", "ui-generated").split(",") if k.strip()]
                if st.session_state.get("run_add_debug_key"):
                    if "debug" not in keys:
                        keys.append("debug")

                # Validate keys
                is_valid, error_msg = validate_keys(keys)
                if not is_valid:
                    st.error(error_msg)
                    return

                notes = st.session_state.get(draft_notes_key, [])
                custom = st.session_state.get(draft_custom_key, {})

                # Automatically add force_copilot to metadata when in LLM mode
                if is_llm_mode:
                    custom["force_copilot"] = True

                # Create run.yaml dynamically
                run_yaml_path = repo_root / "src" / "run.yaml"

                import yaml
                run_config = {
                    "benchmark_id": benchmark,
                    "dataset_id": dataset,
                    "agent_id": agent,  # Always use agent_id
                    "hardware_info": hardware,
                    "n": n_runs,
                    "threads": threads,
                    "metadata": {
                        "keys": keys,
                        "notes": notes,
                        "custom": custom
                    }
                }

                # Add force_copilot flag for LLM mode
                if is_llm_mode:
                    run_config["metadata"]["custom"]["force_copilot"] = True

                # Add task_ids if specific tasks were selected
                if selected_tasks and len(selected_tasks) > 0:
                    run_config["task_ids"] = selected_tasks
                    st.info(f"✓ Running {len(selected_tasks)} selected task(s)")
                else:
                    st.info(f"✓ Running all tasks in dataset")

                with open(run_yaml_path, 'w') as f:
                    yaml.dump(run_config, f)

                st.info("Created run.yaml configuration")

                # Run the benchmark
                command = [sys.executable, str(script_path)]

                if run_in_background:
                    # Run in background
                    process_metadata = {
                        "benchmark": benchmark,
                        "dataset": dataset,
                        "agent": agent,
                        "hardware": hardware,
                        "n_runs": n_runs,
                        "threads": threads,
                        "keys": keys,
                        "task_count": len(selected_tasks) if selected_tasks else "all"
                    }

                    # Add force_copilot flag to metadata
                    if is_llm_mode:
                        process_metadata["force_copilot"] = True

                    # Create description based on mode
                    if is_llm_mode:
                        description = f"Benchmark: {agent} on {benchmark}/{dataset} (LLM mode)"
                    else:
                        description = f"Benchmark: {agent} on {benchmark}/{dataset}"

                    process_id = run_command_background(
                        command=command,
                        cwd=repo_root / "src",
                        description=description,
                        metadata=process_metadata
                    )

                    st.success(f"✓ Benchmark started in background!")
                    st.info(f"📊 Process ID: `{process_id}`")
                    st.info("🔍 Go to the **Process Monitor** tab to view progress and logs")

                    # Clear draft metadata after successful start
                    st.session_state[draft_notes_key] = []
                    st.session_state[draft_custom_key] = {}

                    # Note about database rebuild
                    st.warning("⚠️ Remember to rebuild the database after the benchmark completes!")

                else:
                    # Run in foreground (old behavior)
                    if is_llm_mode:
                        run_description = f"Running {agent} on {benchmark}/{dataset} (LLM mode)..."
                    else:
                        run_description = f"Running {agent} on {benchmark}/{dataset}..."

                    success, output = run_command_with_output(
                        command,
                        cwd=repo_root / "src",
                        description=run_description
                    )

                    if success:
                        st.success("✓ Benchmark completed successfully!")

                        # Clear draft metadata after successful run
                        st.session_state[draft_notes_key] = []
                        st.session_state[draft_custom_key] = {}

                        # Rebuild database to sync new run data
                        st.info("🔨 Rebuilding database...")
                        db_script_path = repo_root / "src" / "build_datatable.py"

                        try:
                            db_result = subprocess.run(
                                [sys.executable, str(db_script_path)],
                                cwd=str(repo_root),
                                capture_output=True,
                                text=True,
                                timeout=60
                            )

                            if db_result.returncode == 0:
                                st.success("✅ Database rebuilt successfully!")
                                st.session_state.db_sync_needed = False
                            else:
                                st.warning("⚠️ Database rebuild completed with warnings")
                                st.session_state.db_sync_needed = True
                                with st.expander("Database build output"):
                                    st.code(db_result.stdout + db_result.stderr)
                        except subprocess.TimeoutExpired:
                            st.error("❌ Database rebuild timed out")
                            st.session_state.db_sync_needed = True
                        except Exception as e:
                            st.error(f"❌ Database rebuild failed: {e}")
                            st.session_state.db_sync_needed = True

                        st.balloons()
                    else:
                        st.error("✗ Benchmark failed")

                    with st.expander("Full Output", expanded=True):
                        st.code(output)

        conn.close()

    except FileNotFoundError:
        st.error("Database not found")


def render_add_existing_run_tab():
    """Render the Add Existing Run tab."""
    from datetime import datetime

    st.markdown("### 📁 Add Existing Run")
    st.markdown(
        """
        Convert an existing run directory (from `results/tmp` or anywhere on disk) into a finalized
        `run.json` entry using `convert_run_results.py`.
        Provide the run configuration details below and poRTLe will generate the JSON and update the database.
        """
    )

    repo_root = Path(__file__).parent.parent.parent.parent
    script_path = repo_root / "src" / "convert_run_results.py"

    if not script_path.exists():
        st.error(f"Script not found: {script_path}")
        return

    draft_notes_key = "draft_notes_add_existing_run"
    draft_custom_key = "draft_custom_add_existing_run"

    if draft_notes_key not in st.session_state:
        st.session_state[draft_notes_key] = []

    if draft_custom_key not in st.session_state:
        st.session_state[draft_custom_key] = {}

    try:
        conn = db_manager.connect_db(st.session_state.db_path)

        st.subheader("Benchmark Selection")

        col1, col2 = st.columns(2)

        with col1:
            benchmarks = db_manager.get_benchmarks(conn)
            if benchmarks:
                benchmark = st.selectbox(
                    "Benchmark *",
                    benchmarks,
                    key="add_existing_run_benchmark_selector"
                )
            else:
                st.error("No benchmarks found in database")
                benchmark = None

        with col2:
            if benchmark:
                datasets = db_manager.get_datasets(conn, benchmark)
                if datasets:
                    dataset = st.selectbox(
                        "Dataset *",
                        datasets,
                        key="add_existing_run_dataset_selector"
                    )
                else:
                    st.error(f"No datasets available for {benchmark}")
                    dataset = None
            else:
                dataset = None
                st.info("Select a benchmark to see its datasets")

        st.markdown("---")

        # Metadata editor
        with st.expander("📋 Metadata (optional)", expanded=False):
            st.markdown("Add metadata to categorize and document this run.")

            default_author = st.session_state.get("user_name", "User")

            st.markdown("**🏷️ Keys**")
            metadata_keys = st.text_input(
                "Keys (comma-separated)",
                value="ui-generated",
                placeholder="e.g., imported, regression, verified",
                help="Add keys to categorize this run",
                key="add_existing_run_metadata_keys"
            )

            st.markdown("---")

            # Notes
            st.markdown("**📝 Notes**")
            draft_notes = st.session_state[draft_notes_key]
            if draft_notes:
                for idx, note in enumerate(draft_notes):
                    col1, col2 = st.columns([10, 1])
                    with col1:
                        st.text(
                            f"📝 {note.get('date_added', 'N/A')} - {note.get('author', 'Unknown')}: {note.get('text', '')}"
                        )
                    with col2:
                        if st.button(
                            "❌",
                            key=f"delete_add_existing_run_note_{idx}",
                            help="Delete this note",
                            disabled=DEMO_MODE
                        ):
                            st.session_state[draft_notes_key].pop(idx)
                            st.rerun()
            else:
                st.caption("No notes yet")

            st.markdown("**Add New Note**")
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                new_note_author = st.text_input(
                    "Author",
                    value=default_author,
                    key="add_existing_run_new_note_author"
                )
            with col2:
                new_note_date = st.text_input(
                    "Date",
                    value=datetime.now().strftime("%m-%d-%y"),
                    key="add_existing_run_new_note_date"
                )
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)

            new_note_text = st.text_area(
                "Note text",
                placeholder="Enter note text...",
                height=80,
                key="add_existing_run_new_note_text"
            )

            if st.button("➕ Add Note", key="add_existing_run_add_note_btn", type="secondary", disabled=DEMO_MODE):
                if new_note_text.strip():
                    new_note = {
                        "date_added": new_note_date,
                        "author": new_note_author,
                        "text": new_note_text.strip()
                    }
                    st.session_state[draft_notes_key].append(new_note)
                    st.rerun()

            st.markdown("---")

            # Custom metadata
            st.markdown("**⚙️ Custom Fields**")
            draft_custom = st.session_state[draft_custom_key]
            if draft_custom:
                for field_key, field_value in draft_custom.items():
                    col1, col2 = st.columns([10, 1])
                    with col1:
                        st.text(f"⚙️ **{field_key}**: {field_value}")
                    with col2:
                        if st.button(
                            "❌",
                            key=f"delete_add_existing_run_custom_{field_key}",
                            help="Delete this field",
                            disabled=DEMO_MODE
                        ):
                            del st.session_state[draft_custom_key][field_key]
                            st.rerun()
            else:
                st.caption("No custom fields")

            st.markdown("**Add New Custom Field**")
            col1, col2, col3 = st.columns([4, 4, 1])
            with col1:
                new_field_key = st.text_input(
                    "Field name",
                    placeholder="e.g., workflow",
                    key="add_existing_run_new_field_key"
                )
            with col2:
                new_field_value = st.text_input(
                    "Field value",
                    placeholder="e.g., agentic",
                    key="add_existing_run_new_field_value"
                )
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)

            if st.button("➕ Add Field", key="add_existing_run_add_field_btn", type="secondary", disabled=DEMO_MODE):
                if new_field_key.strip() and new_field_value.strip():
                    st.session_state[draft_custom_key][new_field_key.strip()] = new_field_value.strip()
                    st.rerun()

        st.markdown("---")

        agents = db_manager.get_agents(conn)

        with st.form("add_existing_run_form"):
            run_dir_input = st.text_input(
                "Run Directory *",
                placeholder="/path/to/results/tmp/.../run_id",
                help="Path containing raw_result.json and agent logs"
            )

            output_path_input = st.text_input(
                "Custom Output Path",
                value="",
                placeholder=str(
                    repo_root / "results" / "json" / "<benchmark>" / "<dataset>" / "<run_id>.json"
                ),
                help="Optional override for the generated run JSON location"
            )

            st.caption("Run IDs are generated automatically using the hardware info and current time.")

            col1, col2 = st.columns(2)

            with col1:
                if agents:
                    agent = st.selectbox(
                        "Agent *",
                        agents,
                        key="add_existing_run_agent_selector"
                    )
                else:
                    st.error("No agents found in database")
                    agent = None

            with col2:
                hardware = st.text_input(
                    "Hardware Info *",
                    value=DEFAULT_HARDWARE_INFO_SANITIZED,
                    help="Identifier describing the hardware (sanitized)"
                )

            col3, col4 = st.columns(2)

            with col3:
                run_start_input = st.text_input(
                    "Run Start (ISO-8601) *",
                    value=datetime.now().isoformat(timespec="seconds"),
                    help="Example: 2025-01-01T12:34:56"
                )

            with col4:
                run_end_input = st.text_input(
                    "Run End (ISO-8601) *",
                    value=datetime.now().isoformat(timespec="seconds"),
                    help="Must be >= Run Start"
                )

            col5, col6 = st.columns(2)

            with col5:
                n_runs = st.number_input("Runs per Task", min_value=1, value=1)

            with col6:
                threads = st.number_input("Threads", min_value=1, value=1)

            submitted = st.form_submit_button("➕ Add Existing Run", type="primary", disabled=DEMO_MODE)

        if submitted:
            errors = []

            if not run_dir_input:
                errors.append("Run directory is required")
            if not benchmark:
                errors.append("Benchmark selection is required")
            if not dataset:
                errors.append("Dataset selection is required")
            if not agent:
                errors.append("Agent selection is required")
            if not hardware:
                errors.append("Hardware info is required")

            try:
                run_start_dt = datetime.fromisoformat(run_start_input)
            except ValueError:
                errors.append("Run start must be a valid ISO-8601 timestamp")
                run_start_dt = None

            try:
                run_end_dt = datetime.fromisoformat(run_end_input)
            except ValueError:
                errors.append("Run end must be a valid ISO-8601 timestamp")
                run_end_dt = None

            if run_start_dt and run_end_dt and run_end_dt < run_start_dt:
                errors.append("Run end cannot be earlier than run start")

            run_dir_path = Path(run_dir_input).expanduser()
            if run_dir_input and not run_dir_path.exists():
                errors.append(f"Run directory not found: {run_dir_path}")

            output_path = None
            if output_path_input.strip():
                output_path = Path(output_path_input).expanduser()

            if errors:
                for err in errors:
                    st.error(err)
            else:
                st.markdown("---")
                st.subheader("Conversion Output")

                keys = [
                    k.strip()
                    for k in st.session_state.get("add_existing_run_metadata_keys", "ui-generated").split(",")
                    if k.strip()
                ]

                # Validate keys
                is_valid, error_msg = validate_keys(keys)
                if not is_valid:
                    st.error(error_msg)
                    return

                notes = st.session_state.get(draft_notes_key, [])
                custom = st.session_state.get(draft_custom_key, {})
                metadata = {
                    "keys": keys,
                    "notes": notes,
                    "custom": custom,
                }

                temp_metadata_path = None
                try:
                    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix="_metadata.json")
                    json.dump(metadata, tmp, indent=2)
                    tmp.flush()
                    temp_metadata_path = tmp.name
                    tmp.close()

                    command = [
                        sys.executable,
                        str(script_path),
                        "--run-dir", str(run_dir_path),
                        "--benchmark-id", benchmark,
                        "--dataset-id", dataset,
                        "--agent-id", agent,
                        "--hardware-info", hardware,
                        "--run-start", run_start_input,
                        "--run-end", run_end_input,
                        "--n", str(int(n_runs)),
                        "--threads", str(int(threads)),
                        "--metadata-file", temp_metadata_path,
                    ]

                    if output_path:
                        command.extend(["--output", str(output_path)])

                    success, output = run_command_with_output(
                        command,
                        cwd=repo_root,
                        description="Converting run directory..."
                    )

                finally:
                    if temp_metadata_path and os.path.exists(temp_metadata_path):
                        os.remove(temp_metadata_path)

                if success:
                    st.success("✓ Run JSON created successfully!")

                    run_id_match = re.search(r"Generated run_id:\s*([A-Za-z0-9\-]+)", output)
                    if not run_id_match:
                        run_id_match = re.search(r"run_id=([A-Za-z0-9\-]+)", output)
                    if run_id_match:
                        st.info(f"Generated Run ID: {run_id_match.group(1)}")

                    st.session_state[draft_notes_key] = []
                    st.session_state[draft_custom_key] = {}

                    st.info("🔨 Rebuilding database...")
                    db_script_path = repo_root / "src" / "build_datatable.py"

                    database_rebuilt = False
                    try:
                        db_result = subprocess.run(
                            [sys.executable, str(db_script_path)],
                            cwd=str(repo_root),
                            capture_output=True,
                            text=True,
                            timeout=60
                        )

                        if db_result.returncode == 0:
                            st.success("✅ Database rebuilt successfully!")
                            st.session_state.db_sync_needed = False
                            database_rebuilt = True
                        else:
                            st.warning("⚠️ Database rebuild completed with warnings")
                            st.session_state.db_sync_needed = True
                            with st.expander("Database build output"):
                                st.code(db_result.stdout + db_result.stderr)
                    except subprocess.TimeoutExpired:
                        st.error("❌ Database rebuild timed out")
                        st.session_state.db_sync_needed = True
                    except Exception as e:
                        st.error(f"❌ Database rebuild failed: {e}")
                        st.session_state.db_sync_needed = True

                    # Only celebrate if database rebuild succeeded
                    if database_rebuilt:
                        st.balloons()
                    else:
                        st.warning("⚠️ Run added but database rebuild incomplete.")
                else:
                    st.error("✗ Conversion failed")

                with st.expander("Full Output", expanded=True):
                    st.code(output)

        conn.close()

    except FileNotFoundError:
        st.error("Database not found")


def render_create_diy_task_tab():
    """Render the Create DIY Task tab."""
    from datetime import datetime

    st.markdown("### 🎨 Create DIY Task")
    st.markdown("""
    Create a custom task from your own RTL code and test infrastructure.
    Upload or specify a directory containing your task files.
    """)

    repo_root = Path(__file__).parent.parent.parent.parent
    script_path = repo_root / "src" / "create_diy_task.py"

    if not script_path.exists():
        st.error(f"Script not found: {script_path}")
        return

    # Instructions
    with st.expander("📖 Instructions", expanded=False):
        st.markdown("""
        ### Task Directory Structure

        Your task directory should contain:

        - **`rtl/`** - Your RTL code (SystemVerilog, Verilog)
          - By default: included in `context` (agent can access it)
          - Optional: can be placed in `patch` as expected solution (for design tasks)
          - ⚠️ **Only include source code** - no Git repos, binaries, or PDFs

        - **`docs/`** (optional) - Documentation and specifications
          - Will be provided to the agent as context
          - Example: specification.md, requirements.txt
          - ⚠️ **Text only** - no PDFs or images

        - **`verif/`** (optional) - Verification files
          - Testbenches, verification scripts
          - Will be provided to the agent as context

        - **`src/`** (optional) - Test harness
          - Test runners, helper scripts
          - Will be included in the test harness
          - Example: test_runner.py, harness_library.py

        - **`docker-compose.yml`** (optional) - Docker configuration
          - Will be included in the test harness

        ### Example Structure

        ```
        my_task/
        ├── rtl/
        │   └── my_module.sv
        ├── docs/
        │   └── specification.md
        ├── verif/
        │   └── my_module_tb.sv
        ├── src/
        │   ├── test_runner.py
        │   └── .env
        └── docker-compose.yml
        ```

        ### Best Practices

        ✅ **Include**: `.sv`, `.v`, `.md`, `.txt`, `.sh`, `.py` source files
        ❌ **Avoid**: PDFs, images, Git repos, compiled binaries, archives

        💡 **Tip**: Enable "Clean task directory before creating" to automatically
        remove problematic files like Git repositories, PDFs, images, and compiled
        binaries that cause JSONL bloat and harness failures.

        ### What Happens Next

        1. (Optional) Task directory is cleaned of binary files
        2. Your task will be converted to JSONL format
        3. It will be added to the specified benchmark/dataset
        4. You can run benchmarks on this task just like any other task
        """)

    st.markdown("---")

    # Task Configuration
    st.subheader("Task Configuration")

    try:
        conn = db_manager.connect_db(st.session_state.db_path)

        col1, col2 = st.columns(2)

        with col1:
            # Benchmark selector with option to create new
            st.markdown("**Benchmark**")
            benchmarks = db_manager.get_benchmarks(conn)
            benchmark_options = ["[Create New Benchmark]"] + (benchmarks if benchmarks else [])
            selected_benchmark_option = st.selectbox(
                "Select or create benchmark",
                benchmark_options,
                key="diy_benchmark_selector",
                label_visibility="collapsed"
            )

            if selected_benchmark_option == "[Create New Benchmark]":
                benchmark_id = st.text_input(
                    "New Benchmark ID *",
                    placeholder="e.g., my_benchmark",
                    help="Unique identifier for the new benchmark",
                    key="diy_new_benchmark_id"
                )
            else:
                benchmark_id = selected_benchmark_option

        with col2:
            # Dataset selector with option to create new
            st.markdown("**Dataset**")
            if benchmark_id and benchmark_id != "[Create New Benchmark]":
                datasets = db_manager.get_datasets(conn, benchmark_id)
                dataset_options = ["[Create New Dataset]"] + (datasets if datasets else [])
            else:
                dataset_options = ["[Create New Dataset]"]

            selected_dataset_option = st.selectbox(
                "Select or create dataset",
                dataset_options,
                key="diy_dataset_selector",
                label_visibility="collapsed"
            )

            if selected_dataset_option == "[Create New Dataset]":
                dataset_id = st.text_input(
                    "New Dataset ID *",
                    placeholder="e.g., custom_tasks",
                    help="Unique identifier for the new dataset",
                    key="diy_new_dataset_id"
                )
            else:
                # Strip benchmark prefix from dataset_id if present (e.g., "b4__b4" -> "b4")
                dataset_id = selected_dataset_option
                if benchmark_id and dataset_id.startswith(f"{benchmark_id}__"):
                    dataset_id = dataset_id[len(benchmark_id) + 2:]  # Remove "benchmark__" prefix

        conn.close()

    except Exception as e:
        st.warning(f"Could not load benchmarks/datasets: {e}")
        st.info("You can still create a new benchmark/dataset below")

        col1, col2 = st.columns(2)
        with col1:
            benchmark_id = st.text_input(
                "Benchmark ID *",
                placeholder="e.g., my_benchmark",
                key="diy_benchmark_id_fallback"
            )
        with col2:
            dataset_id = st.text_input(
                "Dataset ID *",
                placeholder="e.g., custom_tasks",
                key="diy_dataset_id_fallback"
            )

    st.markdown("---")

    # Task Details
    st.subheader("Task Details")

    # Task directory input
    col1, col2 = st.columns([3, 1])
    with col1:
        task_dir = st.text_input(
            "Task Directory Path *",
            placeholder="/path/to/my_task or relative/path/to/my_task",
            help="Path to directory containing rtl/, docs/, verif/, src/, etc. Can include input.jsonl.",
            key="diy_task_dir"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📂 Load from input.jsonl", help="Load configuration from input.jsonl in task directory"):
            if task_dir:
                task_dir_path = Path(task_dir).expanduser().resolve()
                input_jsonl = task_dir_path / "input.jsonl"
                if input_jsonl.exists():
                    try:
                        with input_jsonl.open("r") as f:
                            config = json.loads(f.readline().strip())
                            # Update session state directly for each widget
                            st.session_state["diy_loaded_config"] = config
                            st.session_state["diy_prompt"] = config.get("prompt", "")
                            st.session_state["diy_task_id"] = config.get("id", "")
                            # Convert categories list to comma-separated string
                            if "categories" in config:
                                cats = config["categories"]
                                if isinstance(cats, list):
                                    st.session_state["diy_categories"] = ", ".join(cats)
                                else:
                                    st.session_state["diy_categories"] = str(cats)
                            else:
                                st.session_state["diy_categories"] = ""
                            st.success("✓ Loaded configuration from input.jsonl")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load input.jsonl: {e}")
                else:
                    st.warning("input.jsonl not found in task directory")
            else:
                st.warning("Please enter a task directory path first")

    # Initialize session state keys with CVDP-compliant defaults if they don't exist
    if "diy_prompt" not in st.session_state:
        st.session_state["diy_prompt"] = ""
    if "diy_task_id" not in st.session_state:
        st.session_state["diy_task_id"] = "cvdp_custom_task_0001"
    if "diy_categories" not in st.session_state:
        st.session_state["diy_categories"] = "cid004, easy"
    if "diy_metadata_keys" not in st.session_state:
        st.session_state["diy_metadata_keys"] = "diy-task"
    if "diy_metadata_notes" not in st.session_state:
        st.session_state["diy_metadata_notes"] = []
    if "diy_metadata_custom" not in st.session_state:
        st.session_state["diy_metadata_custom"] = {}

    prompt = st.text_area(
        "Task Prompt *",
        placeholder="Example: Design a FIFO module with configurable depth and width...",
        help="Description of what the agent should do",
        height=150,
        key="diy_prompt"
    )

    col1, col2 = st.columns(2)

    with col1:
        task_id = st.text_input(
            "Task ID (optional)",
            placeholder="cvdp_<task_name>_####",
            help="⚠️ CVDP format is finicky! Recommend to use 'cvdp_' prefix followed by unique name and 4-digit number (e.g., cvdp_custom_task_0001).",
            key="diy_task_id"
        )

    with col2:
        categories = st.text_input(
            "Categories (optional)",
            placeholder="cidXXX, easy",
            help="⚠️ CVDP format is finicky! First category should be 'cidXXX' where XXX is a number, and difficulty level (easy, medium, hard) should be included.",
            key="diy_categories"
        )

    with st.expander("📋 Metadata (optional)", expanded=False):
        st.markdown("Add metadata to help categorize and document this DIY task.")

        default_author = st.session_state.get("user_name", "User")

        st.markdown("**🏷️ Keys**")
        st.session_state["diy_metadata_keys"] = st.text_input(
            "Keys (comma-separated)",
            value=st.session_state["diy_metadata_keys"],
            placeholder="e.g., diy, cvdp, experimental",
            help="Add keys/tags to categorize this task",
            key="diy_metadata_keys_input",
        )

        st.markdown("---")
        st.markdown("**📝 Notes**")
        draft_notes = st.session_state["diy_metadata_notes"]
        if draft_notes:
            for idx, note in enumerate(draft_notes):
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.text(f"📝 {note.get('date_added', 'N/A')} - {note.get('author', 'Unknown')}: {note.get('text', '')}")
                with col2:
                    if st.button("❌", key=f"delete_diy_note_{idx}", help="Delete this note", disabled=DEMO_MODE):
                        st.session_state["diy_metadata_notes"].pop(idx)
                        st.rerun()
        else:
            st.caption("No notes")

        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            new_note_author = st.text_input("Author", value=default_author, key="diy_new_note_author")
        with col2:
            new_note_date = st.text_input("Date", value=datetime.now().strftime("%m-%d-%y"), key="diy_new_note_date")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)

        new_note_text = st.text_area("Note text", placeholder="Enter note text...", height=80, key="diy_new_note_text")

        if st.button("➕ Add Note", key="diy_add_note_btn", type="secondary", disabled=DEMO_MODE):
            if new_note_text.strip():
                new_note = {
                    "date_added": new_note_date,
                    "author": new_note_author,
                    "text": new_note_text.strip()
                }
                st.session_state["diy_metadata_notes"].append(new_note)
                st.rerun()

        st.markdown("---")
        st.markdown("**⚙️ Custom Fields**")
        draft_custom = st.session_state["diy_metadata_custom"]
        if draft_custom:
            for field_key, field_value in draft_custom.items():
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.text(f"⚙️ **{field_key}**: {field_value}")
                with col2:
                    if st.button("❌", key=f"delete_diy_custom_{field_key}", help="Delete this field", disabled=DEMO_MODE):
                        del st.session_state["diy_metadata_custom"][field_key]
                        st.rerun()
        else:
            st.caption("No custom fields")

        col1, col2, col3 = st.columns([4, 4, 1])
        with col1:
            new_field_key = st.text_input("Field name", placeholder="e.g., difficulty", key="diy_new_field_key")
        with col2:
            new_field_value = st.text_input("Field value", placeholder="e.g., hard", key="diy_new_field_value")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)

        if st.button("➕ Add Field", key="diy_add_field_btn", type="secondary", disabled=DEMO_MODE):
            if new_field_key.strip() and new_field_value.strip():
                st.session_state["diy_metadata_custom"][new_field_key.strip()] = new_field_value.strip()
                st.rerun()

    st.markdown("---")

    # Form with submit button
    with st.form("create_diy_task_form"):
        # Checkbox to clean task directory
        clean_task = st.checkbox(
            "🧹 Clean task directory before creating",
            value=False,
            help="Removes binary files (PDFs, images, Git repos, compiled binaries) that cause JSONL bloat and failures. Keeps only source code and text files."
        )

        # Checkbox to auto-build benchmark
        auto_build_benchmark = st.checkbox(
            "🔨 Automatically build benchmark after creating task",
            value=True,
            help="Runs build_benchmark_json.py to rebuild the benchmark JSON file after task creation"
        )

        submitted = st.form_submit_button("🎨 Create Task", type="primary", disabled=DEMO_MODE)

        if submitted:
            errors = []

            if not task_dir:
                errors.append("Task directory path is required")
            if not prompt:
                errors.append("Task prompt is required")
            if not benchmark_id or benchmark_id == "[Create New Benchmark]":
                errors.append("Benchmark ID is required")
            if not dataset_id or dataset_id == "[Create New Dataset]":
                errors.append("Dataset ID is required")

            # Validate task directory exists
            task_dir_path = Path(task_dir).expanduser().resolve()
            if task_dir and not task_dir_path.exists():
                errors.append(f"Task directory not found: {task_dir_path}")

            if errors:
                for err in errors:
                    st.error(err)
            else:
                st.markdown("---")
                st.subheader("Creating DIY Task")

                # Clean task directory if requested
                if clean_task:
                    st.info("🧹 Cleaning task directory...")
                    clean_script_path = repo_root / "diy_tasks" / "clean_task.py"

                    if clean_script_path.exists():
                        try:
                            clean_command = [
                                sys.executable,
                                str(clean_script_path),
                                str(task_dir_path)
                            ]

                            clean_result = subprocess.run(
                                clean_command,
                                cwd=str(repo_root),
                                capture_output=True,
                                text=True,
                                timeout=60
                            )

                            if clean_result.returncode == 0:
                                # Parse the output to show stats
                                output_lines = clean_result.stdout.strip().split('\n')
                                # Look for summary lines
                                for line in output_lines:
                                    if "Files removed:" in line or "Space freed:" in line:
                                        st.success(f"✓ {line.strip()}")

                                with st.expander("🧹 Cleaning Details", expanded=False):
                                    st.code(clean_result.stdout)
                            else:
                                st.warning("⚠️ Cleaning completed with warnings")
                                with st.expander("Cleaning output"):
                                    st.code(clean_result.stdout + clean_result.stderr)
                        except subprocess.TimeoutExpired:
                            st.error("❌ Cleaning timed out")
                        except Exception as e:
                            st.error(f"❌ Cleaning failed: {e}")
                    else:
                        st.warning(f"⚠️ Cleaning script not found at {clean_script_path}")

                # Build command
                command = [
                    sys.executable,
                    str(script_path),
                    "--task-dir", str(task_dir_path),
                    "--prompt", prompt,
                    "--benchmark-id", benchmark_id,
                    "--dataset-id", dataset_id
                ]

                # Add optional arguments
                if task_id:
                    command.extend(["--task-id", task_id])
                if categories:
                    command.extend(["--categories", categories])
                # Metadata (optional)
                metadata_keys = [
                    k.strip() for k in st.session_state.get("diy_metadata_keys", "").split(",") if k.strip()
                ]
                metadata_notes = st.session_state.get("diy_metadata_notes", [])
                metadata_custom = st.session_state.get("diy_metadata_custom", {})

                if metadata_keys:
                    command.extend(["--metadata-keys", json.dumps(metadata_keys)])
                if metadata_notes:
                    command.extend(["--metadata-notes", json.dumps(metadata_notes)])
                if metadata_custom:
                    command.extend(["--metadata-custom", json.dumps(metadata_custom)])

                # Run the script
                success, output = run_command_with_output(
                    command,
                    cwd=repo_root,
                    description=f"Creating DIY task..."
                )

                if success:
                    st.success("✓ DIY task created successfully!")

                    # Build benchmark and rebuild database if checkbox is checked
                    if auto_build_benchmark:
                        st.info("🔨 Building benchmark and rebuilding database...")
                        db_script_path = repo_root / "src" / "build_datatable.py"

                        benchmark_built = False
                        database_rebuilt = False

                        try:
                            # Step 1: Build benchmark JSON
                            build_benchmark_cmd = [
                                sys.executable,
                                str(repo_root / "src" / "build_benchmark_json.py"),
                                benchmark_id
                            ]

                            build_result = subprocess.run(
                                build_benchmark_cmd,
                                cwd=str(repo_root),
                                capture_output=True,
                                text=True,
                                timeout=60
                            )

                            if build_result.returncode == 0:
                                st.success(f"✅ Benchmark JSON built for {benchmark_id}")
                                benchmark_built = True
                            else:
                                st.warning(f"⚠️ Benchmark JSON build had warnings")
                                with st.expander("Build benchmark output"):
                                    st.code(build_result.stdout + build_result.stderr)

                            # Step 2: Rebuild database (always run, even if benchmark build had warnings)
                            st.info("🗄️ Rebuilding database...")
                            db_result = subprocess.run(
                                [sys.executable, str(db_script_path)],
                                cwd=str(repo_root),
                                capture_output=True,
                                text=True,
                                timeout=60
                            )

                            if db_result.returncode == 0:
                                st.success("✅ Database rebuilt successfully!")
                                st.session_state.db_sync_needed = False
                                database_rebuilt = True
                            else:
                                st.warning("⚠️ Database rebuild completed with warnings")
                                st.session_state.db_sync_needed = True
                                with st.expander("Database build output"):
                                    st.code(db_result.stdout + db_result.stderr)

                        except subprocess.TimeoutExpired:
                            st.error("❌ Build process timed out")
                            st.session_state.db_sync_needed = True
                        except Exception as e:
                            st.error(f"❌ Build process failed: {e}")
                            st.session_state.db_sync_needed = True

                        # Step 3: Show success and refresh UI (only after all builds complete)
                        if database_rebuilt:
                            st.balloons()
                            st.info("✓ Your task is now ready to use in benchmarks!")

                            # Force UI refresh to pick up new benchmark
                            st.info("🔄 Refreshing UI to load new benchmark...")
                            import time
                            time.sleep(1)  # Brief pause to ensure database writes are complete
                            st.rerun()
                        else:
                            st.warning("⚠️ Task created but database rebuild incomplete. Refresh may be needed.")
                    else:
                        st.warning("⚠️ Remember to build the benchmark and rebuild the database:")
                        st.code(f"python src/build_benchmark_json.py {benchmark_id}\npython src/build_datatable.py")
                        st.info("✓ Task created! Build benchmark to make it available.")
                else:
                    st.error("✗ Task creation failed")

                with st.expander("Full Output", expanded=True):
                    st.code(output)


def render_build_benchmark_tab():
    """Render the Build Benchmark tab."""
    st.markdown("### 📋 Build Benchmark JSON")
    st.markdown("""
    Create or rebuild benchmark poRTLe JSON files from JSONL datasets.
    """)

    repo_root = Path(__file__).parent.parent.parent.parent
    script_path = repo_root / "src" / "build_benchmark_json.py"

    if not script_path.exists():
        st.error(f"Script not found: {script_path}")
        return

    # Initialize session state for draft metadata
    draft_notes_key = "draft_notes_build_benchmark"
    draft_custom_key = "draft_custom_build_benchmark"

    if draft_notes_key not in st.session_state:
        st.session_state[draft_notes_key] = []

    if draft_custom_key not in st.session_state:
        st.session_state[draft_custom_key] = {}

    # Metadata section OUTSIDE form for interactive editing
    with st.expander("📋 Metadata (optional)", expanded=False):
        st.markdown("Add metadata to categorize and document this benchmark.")

        # Get user name from session state
        default_author = st.session_state.get("user_name", "User")

        # Keys editor
        st.markdown("**🏷️ Keys**")
        metadata_keys = st.text_input(
            "Keys (comma-separated)",
            value="",
            placeholder="e.g., production, experimental, verified",
            help="Add keys to categorize this benchmark",
            key="build_benchmark_metadata_keys"
        )

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
                    if st.button("❌", key=f"delete_build_benchmark_note_{idx}", help="Delete this note", disabled=DEMO_MODE):
                        st.session_state[draft_notes_key].pop(idx)
                        st.rerun()
        else:
            st.caption("No notes")

        # Add new note section
        st.markdown("**Add New Note**")
        from datetime import datetime
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            new_note_author = st.text_input("Author", value=default_author, key="build_benchmark_new_note_author")
        with col2:
            new_note_date = st.text_input("Date", value=datetime.now().strftime("%m-%d-%y"), key="build_benchmark_new_note_date")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)

        new_note_text = st.text_area("Note text", placeholder="Enter note text...", height=80, key="build_benchmark_new_note_text")

        if st.button("➕ Add Note", key="add_build_benchmark_note_btn", type="secondary", disabled=DEMO_MODE):
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
                    if st.button("❌", key=f"delete_build_benchmark_custom_{field_key}", help="Delete this field", disabled=DEMO_MODE):
                        del st.session_state[draft_custom_key][field_key]
                        st.rerun()
        else:
            st.caption("No custom fields")

        # Add new custom field section
        st.markdown("**Add New Custom Field**")
        col1, col2, col3 = st.columns([4, 4, 1])
        with col1:
            new_field_key = st.text_input("Field name", placeholder="e.g., difficulty", key="build_benchmark_new_field_key")
        with col2:
            new_field_value = st.text_input("Field value", placeholder="e.g., hard", key="build_benchmark_new_field_value")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)

        if st.button("➕ Add Field", key="add_build_benchmark_field_btn", type="secondary", disabled=DEMO_MODE):
            if new_field_key.strip() and new_field_value.strip():
                st.session_state[draft_custom_key][new_field_key.strip()] = new_field_value.strip()
                st.rerun()

    st.markdown("---")

    # Show available benchmarks
    st.subheader("Available Benchmarks")
    benchmark_datasets_dir = repo_root / "benchmark_datasets"

    if benchmark_datasets_dir.exists():
        available_benchmarks = []
        for item in benchmark_datasets_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Count datasets (JSONL files)
                dataset_count = sum(1 for f in item.iterdir() if f.is_file() and f.suffix == '.jsonl')
                available_benchmarks.append((item.name, dataset_count))

        if available_benchmarks:
            available_benchmarks.sort()  # Sort alphabetically
            st.markdown("**Found in `benchmark_datasets/`:**")

            # Display in columns for better layout
            cols_per_row = 3
            for i in range(0, len(available_benchmarks), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(available_benchmarks):
                        bench_name, dataset_count = available_benchmarks[idx]
                        with col:
                            st.markdown(f"📋 **{bench_name}**")
                            st.caption(f"{dataset_count} dataset(s)")
        else:
            st.info("No benchmarks found in `benchmark_datasets/` directory")
    else:
        st.warning(f"Directory not found: `{benchmark_datasets_dir.relative_to(repo_root)}`")

    st.markdown("---")

    with st.form("build_benchmark_form"):
        benchmark_name = st.text_input(
            "Benchmark Name *",
            placeholder="e.g., cvdp_example, cvdp, TuRTLe, MyBenchmark",
            help="Name for the benchmark (adapter will be auto-selected based on name)"
        )

        st.caption("Datasets are automatically read from `benchmark_datasets/<benchmark_name>`")

        submitted = st.form_submit_button("🔨 Build Benchmark", type="primary", disabled=DEMO_MODE)

        if submitted:
            if not benchmark_name:
                st.error("Please fill all fields")
            else:
                st.markdown("---")
                st.subheader("Output:")

                # Get metadata from draft state
                keys = [k.strip() for k in st.session_state.get("build_benchmark_metadata_keys", "").split(",") if k.strip()]

                # Validate keys
                is_valid, error_msg = validate_keys(keys)
                if not is_valid:
                    st.error(error_msg)
                    return

                notes = st.session_state.get(draft_notes_key, [])
                custom = st.session_state.get(draft_custom_key, {})

                # Build command with metadata arguments
                command = [
                    sys.executable,
                    str(script_path),
                    benchmark_name,
                    "--metadata-keys", json.dumps(keys),
                    "--metadata-notes", json.dumps(notes),
                    "--metadata-custom", json.dumps(custom)
                ]

                # Run the script
                success, output = run_command_with_output(
                    command,
                    cwd=repo_root,
                    description=f"Building {benchmark_name} benchmark..."
                )

                if success:
                    st.success("✓ Benchmark JSON created successfully!")

                    # Clear draft metadata after successful build
                    st.session_state[draft_notes_key] = []
                    st.session_state[draft_custom_key] = {}

                    # Rebuild database to sync new benchmark data
                    st.info("🔨 Rebuilding database...")
                    db_script_path = repo_root / "src" / "build_datatable.py"

                    database_rebuilt = False
                    try:
                        db_result = subprocess.run(
                            [sys.executable, str(db_script_path)],
                            cwd=str(repo_root),
                            capture_output=True,
                            text=True,
                            timeout=60
                        )

                        if db_result.returncode == 0:
                            st.success("✅ Database rebuilt successfully!")
                            st.session_state.db_sync_needed = False
                            database_rebuilt = True
                        else:
                            st.warning("⚠️ Database rebuild completed with warnings")
                            st.session_state.db_sync_needed = True
                            with st.expander("Database build output"):
                                st.code(db_result.stdout + db_result.stderr)
                    except subprocess.TimeoutExpired:
                        st.error("❌ Database rebuild timed out")
                        st.session_state.db_sync_needed = True
                    except Exception as e:
                        st.error(f"❌ Database rebuild failed: {e}")
                        st.session_state.db_sync_needed = True

                    # Only celebrate if database rebuild succeeded
                    if database_rebuilt:
                        st.balloons()
                    else:
                        st.warning("⚠️ Benchmark built but database rebuild incomplete.")
                else:
                    st.error("✗ Build failed")

                with st.expander("Full Output", expanded=True):
                    st.code(output)


def render():
    """Main render function for commands page."""
    st.title("⚙️ Commands")
    st.markdown("Execute Python scripts and commands from the UI")

    # Tabs for different commands
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Build Benchmark",
        "🤖 Add LLM/Agent",
        "🏃 Run Benchmark",
        "📁 Add Existing Run",
        "🎨 Create DIY Task",
        "🔨 Build Database"
    ])

    with tab1:
        render_build_benchmark_tab()

    with tab2:
        render_add_agent_tab()

    with tab3:
        render_run_benchmark_tab()

    with tab4:
        render_add_existing_run_tab()

    with tab5:
        render_create_diy_task_tab()

    with tab6:
        render_build_database_tab()


if __name__ == "__main__":
    # For testing component standalone
    render()
