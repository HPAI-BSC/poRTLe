"""
JSON Manager for poRTLe UI

Handles reading and writing JSON files (agents.json, benchmark JSONs, run JSONs).
Provides utilities for metadata parsing and manipulation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def parse_metadata(metadata_str: str) -> Dict[str, Any]:
    """
    Parse metadata JSON string into structured data.

    Args:
        metadata_str: JSON string containing metadata

    Returns:
        Dictionary with keys, notes, and custom fields
    """
    try:
        if not metadata_str:
            return {"keys": [], "notes": [], "custom": {}}
        metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
        return {
            "keys": metadata.get("keys", []),
            "notes": metadata.get("notes", []),
            "custom": metadata.get("custom", {})
        }
    except (json.JSONDecodeError, TypeError):
        return {"keys": [], "notes": [], "custom": {}}


def format_metadata(keys: List[str], notes: List[Dict], custom: Dict) -> Dict[str, Any]:
    """
    Format metadata components into standardized structure.

    Args:
        keys: List of key strings
        notes: List of note dictionaries
        custom: Dictionary of custom fields

    Returns:
        Formatted metadata dictionary
    """
    return {
        "keys": keys if keys else [],
        "notes": notes if notes else [],
        "custom": custom if custom else {}
    }


def add_note_to_metadata(
    metadata: Dict[str, Any],
    note_text: str,
    author: str = "Unknown"
) -> Dict[str, Any]:
    """
    Add a new note to metadata.

    Args:
        metadata: Existing metadata dictionary
        note_text: Text of the note
        author: Author name

    Returns:
        Updated metadata dictionary
    """
    if "notes" not in metadata:
        metadata["notes"] = []

    new_note = {
        "date_added": datetime.now().strftime("%m-%d-%y"),
        "author": author,
        "text": note_text
    }

    metadata["notes"].append(new_note)
    return metadata


def load_json_file(file_path: Path) -> Optional[Any]:
    """
    Load a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data or None if error
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None


def save_json_file(file_path: Path, data: Any, indent: int = 2) -> bool:
    """
    Save data to a JSON file.

    Args:
        file_path: Path to JSON file
        data: Data to save
        indent: JSON indentation level

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False


def load_agents_json(repo_root: Path) -> List[Dict[str, Any]]:
    """
    Load agents.json file.

    Args:
        repo_root: Repository root directory

    Returns:
        List of agent dictionaries
    """
    agents_path = repo_root / "results" / "json" / "agents.json"
    data = load_json_file(agents_path)
    return data if data else []


def save_agents_json(repo_root: Path, agents: List[Dict[str, Any]]) -> bool:
    """
    Save agents.json file.

    Args:
        repo_root: Repository root directory
        agents: List of agent dictionaries

    Returns:
        True if successful
    """
    agents_path = repo_root / "results" / "json" / "agents.json"
    return save_json_file(agents_path, agents)


def update_agent_metadata(
    repo_root: Path,
    agent_id: str,
    metadata: Dict[str, Any]
) -> bool:
    """
    Update metadata for a specific agent.

    Args:
        repo_root: Repository root directory
        agent_id: Agent ID to update
        metadata: New metadata dictionary

    Returns:
        True if successful
    """
    agents = load_agents_json(repo_root)

    for agent in agents:
        if agent["agent_id"] == agent_id:
            agent["metadata"] = metadata
            return save_agents_json(repo_root, agents)

    return False


def load_benchmark_json(
    repo_root: Path,
    benchmark_id: str
) -> Optional[Dict[str, Any]]:
    """
    Load a benchmark JSON file.

    Args:
        repo_root: Repository root directory
        benchmark_id: Benchmark ID

    Returns:
        Benchmark data or None if not found
    """
    benchmark_path = (
        repo_root / "results" / "json" / benchmark_id /
        f"{benchmark_id}_benchmark.json"
    )
    return load_json_file(benchmark_path)


def save_benchmark_json(
    repo_root: Path,
    benchmark_id: str,
    data: Dict[str, Any]
) -> bool:
    """
    Save a benchmark JSON file.

    Args:
        repo_root: Repository root directory
        benchmark_id: Benchmark ID
        data: Benchmark data

    Returns:
        True if successful
    """
    benchmark_path = (
        repo_root / "results" / "json" / benchmark_id /
        f"{benchmark_id}_benchmark.json"
    )
    return save_json_file(benchmark_path, data)


def update_benchmark_metadata(
    repo_root: Path,
    benchmark_id: str,
    metadata: Dict[str, Any]
) -> bool:
    """
    Update metadata for a benchmark.

    Args:
        repo_root: Repository root directory
        benchmark_id: Benchmark ID
        metadata: New metadata

    Returns:
        True if successful
    """
    data = load_benchmark_json(repo_root, benchmark_id)
    if data:
        data["metadata"] = metadata
        return save_benchmark_json(repo_root, benchmark_id, data)
    return False


def update_dataset_metadata(
    repo_root: Path,
    benchmark_id: str,
    dataset_id: str,
    metadata: Dict[str, Any]
) -> bool:
    """
    Update metadata for a dataset within a benchmark.

    Args:
        repo_root: Repository root directory
        benchmark_id: Benchmark ID
        dataset_id: Dataset ID
        metadata: New metadata

    Returns:
        True if successful
    """
    data = load_benchmark_json(repo_root, benchmark_id)
    if data:
        for dataset in data.get("datasets", []):
            if dataset["dataset_id"] == dataset_id:
                dataset["metadata"] = metadata
                return save_benchmark_json(repo_root, benchmark_id, data)
    return False


def update_task_metadata(
    repo_root: Path,
    benchmark_id: str,
    task_id: str,
    metadata: Dict[str, Any]
) -> bool:
    """
    Update metadata for a task within a benchmark.

    Args:
        repo_root: Repository root directory
        benchmark_id: Benchmark ID
        task_id: Task ID
        metadata: New metadata

    Returns:
        True if successful
    """
    data = load_benchmark_json(repo_root, benchmark_id)
    if data:
        for dataset in data.get("datasets", []):
            for task in dataset.get("tasks", []):
                if task["task_id"] == task_id:
                    task["metadata"] = metadata
                    return save_benchmark_json(repo_root, benchmark_id, data)
    return False


def load_run_json(
    repo_root: Path,
    benchmark_id: str,
    dataset_id: str,
    run_id: str
) -> Optional[Dict[str, Any]]:
    """
    Load a run JSON file.

    Args:
        repo_root: Repository root directory
        benchmark_id: Benchmark ID
        dataset_id: Dataset ID
        run_id: Run ID

    Returns:
        Run data or None if not found
    """
    run_path = (
        repo_root / "results" / "json" / benchmark_id /
        dataset_id / f"{run_id}.json"
    )
    return load_json_file(run_path)


def save_run_json(
    repo_root: Path,
    benchmark_id: str,
    dataset_id: str,
    run_id: str,
    data: Dict[str, Any]
) -> bool:
    """
    Save a run JSON file.

    Args:
        repo_root: Repository root directory
        benchmark_id: Benchmark ID
        dataset_id: Dataset ID
        run_id: Run ID
        data: Run data

    Returns:
        True if successful
    """
    run_path = (
        repo_root / "results" / "json" / benchmark_id /
        dataset_id / f"{run_id}.json"
    )
    return save_json_file(run_path, data)


def update_run_metadata(
    repo_root: Path,
    benchmark_id: str,
    dataset_id: str,
    run_id: str,
    metadata: Dict[str, Any]
) -> bool:
    """
    Update metadata for a run.

    Args:
        repo_root: Repository root directory
        benchmark_id: Benchmark ID
        dataset_id: Dataset ID
        run_id: Run ID
        metadata: New metadata

    Returns:
        True if successful
    """
    data = load_run_json(repo_root, benchmark_id, dataset_id, run_id)
    if data:
        data["metadata"] = metadata
        return save_run_json(repo_root, benchmark_id, dataset_id, run_id, data)
    return False


def update_datapoint_metadata(
    repo_root: Path,
    benchmark_id: str,
    dataset_id: str,
    run_id: str,
    datapoint_id: str,
    metadata: Dict[str, Any]
) -> bool:
    """
    Update metadata for a datapoint within a run.

    Args:
        repo_root: Repository root directory
        benchmark_id: Benchmark ID
        dataset_id: Dataset ID
        run_id: Run ID
        datapoint_id: Datapoint ID
        metadata: New metadata

    Returns:
        True if successful
    """
    data = load_run_json(repo_root, benchmark_id, dataset_id, run_id)
    if data:
        for datapoint in data.get("datapoints", []):
            if datapoint["datapoint_id"] == datapoint_id:
                datapoint["metadata"] = metadata
                return save_run_json(repo_root, benchmark_id, dataset_id, run_id, data)
    return False


def load_run_jsons_for_dataset(
    repo_root: Path,
    benchmark_id: str,
    dataset_id: str
) -> List[Dict[str, Any]]:
    """
    Load all run JSON files for a dataset.

    Args:
        repo_root: Repository root directory
        benchmark_id: Benchmark ID
        dataset_id: Dataset ID

    Returns:
        List of run data dictionaries
    """
    runs_dir = repo_root / "results" / "json" / benchmark_id / dataset_id
    runs = []

    if runs_dir.exists():
        for json_file in runs_dir.glob("*.json"):
            data = load_json_file(json_file)
            if data:
                runs.append(data)

    return runs
