#!/usr/bin/env python3
"""
Description: File that adds an agent defined by the user to agents.json

Can be used in two ways:
1. Edit the configuration variables at the top and run the script
2. Pass arguments via command line (used by the UI)

When the file is run, it first checks to make sure that the agent defined at the top of
the file is formatted correctly. Then it checks to make sure the agent_name does not
exist inside the agents.json file yet. When everything looks good it adds the agent
to the agents.json file.
"""

# ===== USER CONFIGURATION - EDIT THESE VARIABLES =====
AGENT_ID = "opencode-agent-kimi-k2-thinking"
ABOUT = "First example agent for CVDP benchmark testing"
BACKEND_MODEL = "kimi-k2-thinking"
AGENT_FOLDER_PATH = "agents/opencode-agent"
CUSTOM_CONFIG = {
    "fed": "Bread",
    "sleep": 0.2
}
METADATA_KEYS = ["initial-development"]
METADATA_NOTES = [
    {
        "date_added": "11-15-25",
        "author": "Dakota Barnes",
        "text": "Initial kimi-k2-thinking test"
    }
]
METADATA_CUSTOM = {}
# ===== END USER CONFIGURATION =====

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple


def ensure_dir(p: Path):
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def create_metadata(keys: List[str] = None, notes: List[Dict] = None,
                   custom: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a metadata dictionary with standard structure.

    Args:
        keys: List of keys for sorting/filtering
        notes: List of note dictionaries {date_added, author, text}
        custom: Custom metadata dictionary

    Returns:
        Metadata dictionary
    """
    return {
        "keys": keys or [],
        "notes": notes or [],
        "custom": custom or {}
    }


def validate_agent_config(agent_id: str, about: str, backend_model: str,
                         agent_folder_path: str, custom_config: dict,
                         metadata_keys: list, metadata_notes: list,
                         metadata_custom: dict) -> Tuple[bool, str]:
    """
    Validate the agent configuration.

    Args:
        agent_id: Agent ID
        about: Agent description
        backend_model: Backend model name
        agent_folder_path: Path to agent folder
        custom_config: Custom configuration dict
        metadata_keys: Metadata keys list
        metadata_notes: Metadata notes list
        metadata_custom: Metadata custom dict

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check AGENT_ID
    if not agent_id or not isinstance(agent_id, str) or not agent_id.strip():
        return False, "AGENT_ID must be a non-empty string"

    # Check AGENT_ID format (alphanumeric, hyphens, underscores only)
    if not re.match(r'^[a-zA-Z0-9_-]+$', agent_id):
        return False, f"AGENT_ID '{agent_id}' contains invalid characters. Use only letters, numbers, hyphens, and underscores."

    # Check ABOUT
    if not about or not isinstance(about, str) or not about.strip():
        return False, "ABOUT must be a non-empty string"

    # Check BACKEND_MODEL
    if not backend_model or not isinstance(backend_model, str) or not backend_model.strip():
        return False, "BACKEND_MODEL must be a non-empty string"

    # Check AGENT_FOLDER_PATH
    if not agent_folder_path or not isinstance(agent_folder_path, str) or not agent_folder_path.strip():
        return False, "AGENT_FOLDER_PATH must be a non-empty string"

    # Check CUSTOM_CONFIG
    if not isinstance(custom_config, dict):
        return False, "CUSTOM_CONFIG must be a dictionary"

    # Check METADATA_KEYS
    if not isinstance(metadata_keys, list):
        return False, "METADATA_KEYS must be a list"

    # Check METADATA_NOTES
    if not isinstance(metadata_notes, list):
        return False, "METADATA_NOTES must be a list"

    for i, note in enumerate(metadata_notes):
        if not isinstance(note, dict):
            return False, f"METADATA_NOTES[{i}] must be a dictionary"

        required_fields = ["date_added", "author", "text"]
        for field in required_fields:
            if field not in note:
                return False, f"METADATA_NOTES[{i}] missing required field: {field}"
            if not isinstance(note[field], str) or not note[field].strip():
                return False, f"METADATA_NOTES[{i}][{field}] must be a non-empty string"

    # Check METADATA_CUSTOM
    if not isinstance(metadata_custom, dict):
        return False, "METADATA_CUSTOM must be a dictionary"

    return True, ""


def load_agents_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    Load existing agents.json or return empty list.

    Args:
        json_path: Path to agents.json file

    Returns:
        List of agent dictionaries
    """
    if not json_path.exists():
        return []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Handle both list format and dict with "agents" key
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "agents" in data:
                return data["agents"]
            else:
                return []
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing agents.json: {e}")


def agent_exists(agents: List[Dict[str, Any]], agent_id: str) -> bool:
    """
    Check if agent_id already exists in agents list.

    Args:
        agents: List of agent dictionaries
        agent_id: Agent ID to check

    Returns:
        True if agent exists, False otherwise
    """
    for agent in agents:
        if agent.get("agent_id") == agent_id:
            return True
    return False


def create_agent_entry(agent_id: str, about: str, backend_model: str,
                       agent_folder_path: str, custom_config: dict,
                       metadata_keys: list, metadata_notes: list,
                       metadata_custom: dict) -> Dict[str, Any]:
    """
    Create agent entry from configuration.

    Args:
        agent_id: Agent ID
        about: Agent description
        backend_model: Backend model name
        agent_folder_path: Path to agent folder
        custom_config: Custom configuration dict
        metadata_keys: Metadata keys list
        metadata_notes: Metadata notes list
        metadata_custom: Metadata custom dict

    Returns:
        Agent dictionary following schema
    """
    return {
        "agent_id": agent_id,
        "about": about,
        "agent_config": {
            "backend_model": backend_model,
            "agent_folder_path": agent_folder_path,
            "custom": custom_config
        },
        "metadata": create_metadata(
            keys=metadata_keys,
            notes=metadata_notes,
            custom=metadata_custom
        )
    }


def save_agents_json(json_path: Path, agents: List[Dict[str, Any]]):
    """
    Save agents list to agents.json.

    Args:
        json_path: Path to agents.json file
        agents: List of agent dictionaries
    """
    # Ensure directory exists
    ensure_dir(json_path.parent)

    # Save with pretty formatting
    with open(json_path, 'w') as f:
        json.dump(agents, f, indent=2)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Add an agent to agents.json",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--agent-id", type=str, help="Agent ID")
    parser.add_argument("--about", type=str, help="Agent description")
    parser.add_argument("--backend-model", type=str, help="Backend model name")
    parser.add_argument("--agent-folder", type=str, help="Path to agent folder")
    parser.add_argument("--custom-config", type=str, default="{}", help="Custom config as JSON string")
    parser.add_argument("--metadata-keys", type=str, default="[]", help="Metadata keys as JSON array string")
    parser.add_argument("--metadata-notes", type=str, default="[]", help="Metadata notes as JSON array string")
    parser.add_argument("--metadata-custom", type=str, default="{}", help="Metadata custom as JSON string")

    return parser.parse_args()


def main():
    """Main function to add agent to agents.json."""
    print("\n=== Adding Agent to agents.json ===\n")

    # Check if using CLI args or hardcoded variables
    args = parse_args()

    # Use CLI args if provided, otherwise use hardcoded variables
    if args.agent_id:
        agent_id = args.agent_id
        about = args.about
        backend_model = args.backend_model
        agent_folder_path = args.agent_folder
        custom_config = json.loads(args.custom_config)
        metadata_keys = json.loads(args.metadata_keys)
        metadata_notes = json.loads(args.metadata_notes)
        metadata_custom = json.loads(args.metadata_custom)
        print("Using command-line arguments...")
    else:
        agent_id = AGENT_ID
        about = ABOUT
        backend_model = BACKEND_MODEL
        agent_folder_path = AGENT_FOLDER_PATH
        custom_config = CUSTOM_CONFIG
        metadata_keys = METADATA_KEYS
        metadata_notes = METADATA_NOTES
        metadata_custom = METADATA_CUSTOM
        print("Using configuration variables from file...")

    # Step 1: Validate configuration
    print("\nStep 1: Validating agent configuration...")
    is_valid, error_msg = validate_agent_config(
        agent_id, about, backend_model, agent_folder_path,
        custom_config, metadata_keys, metadata_notes, metadata_custom
    )
    if not is_valid:
        print(f"ERROR: {error_msg}")
        print("\nPlease fix the configuration and try again.")
        return 1
    print("  ✓ Configuration is valid")

    # Step 2: Get paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    json_path = repo_root / "results" / "json" / "agents.json"

    print(f"\nStep 2: Loading agents.json from {json_path}...")
    agents = load_agents_json(json_path)
    print(f"  ✓ Found {len(agents)} existing agent(s)")

    # Step 3: Check for duplicates
    print(f"\nStep 3: Checking for duplicate agent_id '{agent_id}'...")
    if agent_exists(agents, agent_id):
        print(f"ERROR: Agent with id '{agent_id}' already exists in agents.json")
        print("\nPlease use a different AGENT_ID or remove the existing agent first.")
        return 1
    print("  ✓ Agent ID is unique")

    # Step 4: Create agent entry
    print("\nStep 4: Creating agent entry...")
    new_agent = create_agent_entry(
        agent_id, about, backend_model, agent_folder_path,
        custom_config, metadata_keys, metadata_notes, metadata_custom
    )
    print(f"  ✓ Created agent entry for '{agent_id}'")
    print(f"     Backend model: {backend_model}")
    print(f"     Agent folder: {agent_folder_path}")
    print(f"     About: {about}")

    # Step 5: Add and save
    print("\nStep 5: Adding agent to agents.json...")
    agents.append(new_agent)
    save_agents_json(json_path, agents)
    print(f"  ✓ Successfully saved to {json_path}")

    # Success summary
    print("\n=== Agent Added Successfully ===")
    print(f"Agent ID: {agent_id}")
    print(f"Total agents in registry: {len(agents)}")
    print(f"Location: {json_path}")

    return 0


if __name__ == "__main__":
    exit(main())
