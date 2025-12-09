"""
File Viewer Utilities for poRTLe UI

Handles construction of file paths for logs and test reports,
and provides utilities for reading and displaying these files.
"""

import re
from pathlib import Path
from typing import Optional, Tuple, List


def construct_log_paths(
    repo_root: Path,
    benchmark_id: str,
    dataset_id: str,
    run_id: str,
    agent_id: str,
    datapoint_id: str,
    run_directory: Optional[str] = None,
    datapoint_directory: Optional[str] = None
) -> Tuple[List[Path], List[Path], Optional[Path]]:
    """
    Construct file paths for agent log and test report by searching the reports directory.

    The actual path structure is:
    results/tmp/{benchmark_id}/{dataset_id}/{run_id}/{task_dir_name}/reports/{number}_agent.txt
    results/tmp/{benchmark_id}/{dataset_id}/{run_id}/{task_dir_name}/reports/{number}.txt

    Datapoint ID format: {run_id}-{task_id}-{run_num:03d}
    Example: local-machine2-11-17-25-130824-benchmark_dataset__task_id_0001-000

    Args:
        repo_root: Repository root directory
        benchmark_id: Benchmark ID
        dataset_id: Dataset ID
        run_id: Run ID (e.g., "local-machine2-11-17-25-130824")
        agent_id: Agent ID (not used in path, kept for compatibility)
        datapoint_id: Datapoint ID
        run_directory: Optional run directory path from database (if available, used instead of constructing path)
        datapoint_directory: Optional datapoint directory path (e.g., "sample_1") for runs with n > 1

    Returns:
        Tuple of (agent_log_paths, test_report_paths, reports_dir)
        - agent_log_paths: List of agent log files found
        - test_report_paths: List of test report files found
        - reports_dir: The reports directory that was searched
    """
    # Parse datapoint_id to extract task_id and run_num
    # Format: {run_id}-{task_id}-{run_num:03d}
    # Example: local-machine2-11-17-25-130824-benchmark_dataset__task_id_0001-000

    # Split by run_id first to get the rest
    if not datapoint_id.startswith(run_id):
        return [], [], None

    remainder = datapoint_id[len(run_id):].lstrip("-")

    # The last 3 characters should be the run number (000, 001, etc.)
    if len(remainder) < 4:  # At least -{run_num}
        return [], [], None

    # Extract run_num from the end (last -XXX)
    parts = remainder.rsplit("-", 1)
    if len(parts) != 2:
        return [], [], None

    task_id = parts[0]
    run_num_str = parts[1]

    try:
        run_num = int(run_num_str)
        # Convert to 1-based index (datapoint uses 0-based, files use 1-based)
        file_number = run_num + 1
    except ValueError:
        return [], [], None

    # Extract source_task_id from task_id
    # task_id format: {dataset_id}__{source_task_id}
    # Example: benchmark_name__dataset_name__source_task_id_0001
    # We need to extract the source_task_id portion
    expected_prefix = f"{dataset_id}__"
    if task_id.startswith(expected_prefix):
        source_task_id = task_id[len(expected_prefix):]
    else:
        # Fallback: use task_id as-is
        source_task_id = task_id

    # Remove the _#### suffix from source_task_id to get task directory name
    # Note: This pattern is used by CVDP and may vary for other benchmarks.
    # For benchmarks without this suffix, the regex simply returns the original string.
    task_dir_name = re.sub(r'_\d{4}$', '', source_task_id)

    # Construct the reports directory path
    # Use datapoint_directory from metadata if available, otherwise construct the path
    if datapoint_directory:
        run_dir = Path(datapoint_directory)
    elif run_directory:
        run_dir = Path(run_directory)
    else:
        run_dir = repo_root / "results" / "tmp" / benchmark_id / dataset_id / run_id

    # Search for task directories matching the pattern (may have slight variations)
    # First try exact match
    reports_dir = run_dir / task_dir_name / "reports"

    if not reports_dir.exists():
        # Try to find a matching task directory
        if run_dir.exists():
            # Look for directories that start with the task name
            matching_dirs = list(run_dir.glob(f"{task_dir_name}*"))
            if matching_dirs:
                # Use the first match
                reports_dir = matching_dirs[0] / "reports"

    # If reports directory doesn't exist, return empty lists
    if not reports_dir.exists():
        return [], [], reports_dir

    # Find all agent log files (pattern: *_agent.txt)
    agent_log_paths = sorted(reports_dir.glob("*_agent.txt"))

    # Find all test report files (pattern: *.txt but not *_agent.txt)
    test_report_paths = sorted([
        f for f in reports_dir.glob("*.txt")
        if not f.name.endswith("_agent.txt")
    ])

    return agent_log_paths, test_report_paths, reports_dir


def read_file_content(file_path: Path) -> Tuple[bool, str]:
    """
    Read content from a file.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (success, content)
        - success: True if file was read successfully, False otherwise
        - content: File content if successful, error message otherwise
    """
    try:
        if not file_path.exists():
            return False, f"File not found at: {file_path}"

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        if not content.strip():
            return True, "(File is empty)"

        return True, content

    except PermissionError:
        return False, f"Permission denied: {file_path}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def get_file_info(file_path: Path) -> str:
    """
    Get information about a file (size, last modified, etc.).

    Args:
        file_path: Path to the file

    Returns:
        Formatted file information string
    """
    if not file_path.exists():
        return "File does not exist"

    try:
        stat = file_path.stat()
        size_kb = stat.st_size / 1024

        if size_kb < 1:
            size_str = f"{stat.st_size} bytes"
        elif size_kb < 1024:
            size_str = f"{size_kb:.1f} KB"
        else:
            size_str = f"{size_kb/1024:.1f} MB"

        return f"Size: {size_str}"
    except Exception as e:
        return f"Error getting file info: {str(e)}"
