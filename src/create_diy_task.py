#!/usr/bin/env python3
"""
create_diy_task.py

Create a DIY (Do-It-Yourself) task for poRTLe from a directory or code repository.

This script takes a directory containing RTL code and supporting files, along with
a prompt for the agent, and converts it into a JSONL task entry compatible with
poRTLe benchmarks.

Directory Structure (user provides):
  task_dir/
    rtl/           - RTL code (will be included in context)
    docs/          - Documentation (included in context)
    verif/         - Verification files (included in context)
    src/           - Test harness (included in harness, not given to agent)
    docker-compose.yml - Used to build Test harness docker (included in harness)

The script generates a JSONL entry with:
  - id: unique task identifier
  - categories: task categories
  - system_message: instructions for the agent
  - prompt: the task description
  - context: docs/, verif/, and other supporting files
  - harness: test infrastructure (src/, docker-compose.yml)
  - patch: empty (will be filled by agent solution)

Usage:
    python src/create_diy_task.py --task-dir <path> --prompt <prompt> \
        --benchmark-id <benchmark> --dataset-id <dataset> \
        [--task-id <id>] [--categories <cat1,cat2>] [--output <file>]
"""

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Dict, Any


# Default system message for DIY tasks
DEFAULT_SYSTEM_MESSAGE = """  You are a language model that has the following file operations available at your disposal:
  - **List files in a directory** by running one of the following commands:
    - `ls`
    - `tree`
  - **Read files** by using:
    - `cat <filename>`
  - **Write files** by using:
    - `echo <content> > <filename>`
  - **Compile Verilog** by using `iverilog` such as:
    - `iverilog -o <output_filename>.out -g2012 <verilog_code_file> <verilog_testbench_file>`
  - **Run Simulation** by using:
    - `vvp <output_filename>.out`
 - **Update the file content** by using:
    - `sed -i '3s/old/new/' file.txt`
  - **Find current working directory** by using:
    - `pwd`

You will be given a prompt and your task is to understand it and solve the given issue by using the above-mentioned commands as needed. In the final step, you should create a Linux patch to highlight the necessary file updates to achieve the targeted goal.

  You will solve the problem step by step using the following approach of
  - thought (thinking process of the step you're going to take)
  - action (the command you will be running to get more details/context that's helpful to solve the problem)
  - observation (the output from the action you will observe based on which you will take your next step)

  The last step will be the final output summary and the patch itself in the following format
  - thought (the summary of what you did and some introduction of the patch file itself)
  - patch (a Linux-based patch that needs to be applied to reach the relevant solution)

  The patch file should only be applied to a single file to reach the required solution."""


def sanitize_filename(name: str) -> str:
    """
    Sanitize a filename to be safe for filesystem and JSON.
    Replace spaces and special characters with underscores or hyphens.
    """
    # Replace spaces with hyphens
    name = re.sub(r'\s+', '-', name)
    # Replace problematic characters with underscores
    name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
    return name


def read_text_or_b64(p: Path) -> Any:
    """
    Read a file as text if possible, otherwise as base64.

    Args:
        p: Path to file

    Returns:
        String content or dict with __b64__ key
    """
    try:
        return p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        try:
            return {"__b64__": base64.b64encode(p.read_bytes()).decode("ascii")}
        except Exception:
            return ""


def is_harness_path(rel: Path) -> bool:
    """
    Determine if a path should be classified as part of the test harness.

    Harness files include:
      - docker-compose.yml
      - src/** (test runners, helpers, etc.)
      - .env files in src/

    Args:
        rel: Relative path

    Returns:
        True if this is a harness file
    """
    s = rel.as_posix()
    return (
        s.endswith("docker-compose.yml")
        or s.startswith("src/")
        or s.endswith("src/.env")
        or s.endswith(".env")
    )


def collect_files_from_directory(
    task_dir: Path,
    rtl_as_solution: bool = False
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Collect files from a task directory and organize them into context, harness, and patch.

    Args:
        task_dir: Root directory containing task files
        rtl_as_solution: If True, RTL files go in patch (as expected solution)
                        If False, RTL files go in context (as existing code to work with)

    Returns:
        Tuple of (context, harness, patch) dictionaries
    """
    context = {}
    harness = {}
    patch = {}

    # Track RTL files for patch field
    rtl_files = []

    # Process all files in the directory
    for p in task_dir.rglob("*"):
        if p.is_dir():
            continue

        rel = p.relative_to(task_dir)
        rel_str = rel.as_posix()

        # Skip hidden files (except .env), common ignore patterns, and input.jsonl (metadata file)
        if (rel.name.startswith('.') and rel.name != '.env') or rel.name in ['__pycache__', '.DS_Store', 'input.jsonl']:
            continue

        # Read file content
        content = read_text_or_b64(p)

        # Classify the file
        if is_harness_path(rel):
            # Test harness files
            harness[rel_str] = content
        elif rel_str.startswith("rtl/"):
            # RTL files go in context and we track them for patch
            context[rel_str] = content
            rtl_files.append(rel_str)
        else:
            # Other files (docs, verif, etc.) always go in context
            context[rel_str] = content

    # Create patch entries for RTL files (empty strings to indicate patchable files)
    for rtl_file in rtl_files:
        patch[rtl_file] = ""

    return context, harness, patch


def generate_task_id(benchmark_id: str, dataset_id: str, base_id: str = None) -> str:
    """
    Generate a unique task ID.

    Args:
        benchmark_id: Benchmark identifier
        dataset_id: Dataset identifier
        base_id: Optional base identifier

    Returns:
        Unique task ID
    """
    if base_id:
        # Sanitize the base_id
        base_id = sanitize_filename(base_id)
        return f"{benchmark_id}_{dataset_id}_{base_id}"
    else:
        # Generate a timestamp-based ID
        import time
        timestamp = int(time.time())
        return f"{benchmark_id}_{dataset_id}_diy_{timestamp}"


def create_diy_task(
    task_dir: Path,
    prompt: str,
    benchmark_id: str,
    dataset_id: str,
    task_id: str = None,
    categories: list = None,
    system_message: str = None,
    metadata: dict = None,
    rtl_as_solution: bool = False
) -> Dict[str, Any]:
    """
    Create a DIY task entry from a directory.

    Args:
        task_dir: Directory containing task files
        prompt: Task prompt for the agent
        benchmark_id: Benchmark identifier
        dataset_id: Dataset identifier
        task_id: Optional custom task ID
        categories: Optional list of category tags
        system_message: Optional custom system message
        metadata: Optional metadata dict
        rtl_as_solution: If True, RTL goes in patch (as expected solution)
                        If False (default), RTL goes in context (as existing code)

    Returns:
        Dictionary representing the task in JSONL format
    """
    # Generate or use provided task ID
    if not task_id:
        task_id = generate_task_id(benchmark_id, dataset_id)
    else:
        task_id = sanitize_filename(task_id)

    # Use default categories if none provided
    if not categories:
        categories = ["diy", "custom"]

    # Use default system message if none provided
    if not system_message:
        system_message = DEFAULT_SYSTEM_MESSAGE

    # Collect files from directory
    context, harness, patch = collect_files_from_directory(
        task_dir,
        rtl_as_solution=rtl_as_solution
    )

    # Build the task object
    task = {
        "id": task_id,
        "categories": categories,
        "system_message": system_message,
        "prompt": prompt,
        "context": context,
        "harness": harness,
        "patch": patch
    }

    # Add metadata if provided
    if metadata:
        task["metadata"] = metadata

    return task


def append_to_jsonl(task: Dict[str, Any], output_file: Path) -> None:
    """
    Append a task to a JSONL file (or create it if it doesn't exist).

    Args:
        task: Task dictionary
        output_file: Path to output JSONL file
    """
    # Create parent directories if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Append to file
    with output_file.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(task, ensure_ascii=False) + "\n")


def load_input_jsonl(task_dir: Path) -> dict:
    """
    Load configuration from input.jsonl if it exists in the task directory.

    Args:
        task_dir: Task directory to check for input.jsonl

    Returns:
        Dictionary with configuration or empty dict if file doesn't exist
    """
    input_file = task_dir / "input.jsonl"
    if not input_file.exists():
        return {}

    try:
        with input_file.open("r", encoding="utf-8") as f:
            # Read first line only (JSONL format)
            first_line = f.readline().strip()
            if first_line:
                config = json.loads(first_line)
                print(f"✓ Loaded configuration from input.jsonl")
                return config
    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: Could not parse input.jsonl: {e}")
        return {}

    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Create a DIY task for poRTLe from a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a task from a directory
  python src/create_diy_task.py \\
      --task-dir my_task/ \\
      --prompt "Design a FIFO module" \\
      --benchmark-id my_benchmark \\
      --dataset-id custom_tasks

  # Create with custom task ID and categories
  python src/create_diy_task.py \\
      --task-dir my_task/ \\
      --prompt "Design a FIFO module" \\
      --benchmark-id my_benchmark \\
      --dataset-id custom_tasks \\
      --task-id fifo_design_001 \\
      --categories easy,fifo,memory

  # Specify output location
  python src/create_diy_task.py \\
      --task-dir my_task/ \\
      --prompt "Design a FIFO module" \\
      --benchmark-id my_benchmark \\
      --dataset-id custom_tasks \\
      --output benchmark_datasets/my_benchmark/custom_tasks.jsonl
        """
    )

    # Required arguments
    parser.add_argument(
        "--task-dir",
        type=Path,
        required=True,
        help="Directory containing task files (rtl/, docs/, verif/, src/, etc.). "
             "Can include input.jsonl with task configuration."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        help="Task prompt/description for the agent (read from input.jsonl if not provided)"
    )
    parser.add_argument(
        "--benchmark-id",
        type=str,
        required=True,
        help="Benchmark identifier (e.g., 'my_benchmark', 'cvdp')"
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Dataset identifier (e.g., 'custom_tasks', 'easy')"
    )

    # Optional arguments
    parser.add_argument(
        "--task-id",
        type=str,
        help="Custom task ID (auto-generated if not provided)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        help="Comma-separated list of categories (e.g., 'easy,fifo,memory')"
    )
    parser.add_argument(
        "--system-message",
        type=str,
        help="Custom system message (uses default if not provided)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL file path (defaults to benchmark_datasets/<benchmark>/<dataset>.jsonl)"
    )
    parser.add_argument(
        "--rtl-as-solution",
        action="store_true",
        help="Put RTL files in 'patch' as expected solution (for design tasks). "
             "By default, RTL goes in 'context' as existing code to work with."
    )

    # Metadata arguments
    parser.add_argument(
        "--metadata-keys",
        type=str,
        help="JSON-encoded list of metadata keys"
    )
    parser.add_argument(
        "--metadata-notes",
        type=str,
        help="JSON-encoded list of metadata notes"
    )
    parser.add_argument(
        "--metadata-custom",
        type=str,
        help="JSON-encoded dict of custom metadata fields"
    )

    args = parser.parse_args()

    # Validate task directory
    if not args.task_dir.exists():
        raise SystemExit(f"Error: Task directory not found: {args.task_dir}")

    # Try to load configuration from input.jsonl
    input_config = load_input_jsonl(args.task_dir)

    # Merge command-line args with input.jsonl (CLI takes precedence)
    prompt = args.prompt or input_config.get("prompt")
    task_id = args.task_id or input_config.get("id")
    system_message = args.system_message or input_config.get("system_message")

    # Validate required fields
    if not prompt:
        raise SystemExit("Error: --prompt is required (or provide input.jsonl with 'prompt' field)")

    # Parse categories (CLI takes precedence over input.jsonl)
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    elif "categories" in input_config:
        categories = input_config["categories"] if isinstance(input_config["categories"], list) else None

    # Parse metadata
    metadata = {}
    if args.metadata_keys:
        try:
            metadata["keys"] = json.loads(args.metadata_keys)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Error: Invalid JSON in --metadata-keys: {e}")

    if args.metadata_notes:
        try:
            metadata["notes"] = json.loads(args.metadata_notes)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Error: Invalid JSON in --metadata-notes: {e}")

    if args.metadata_custom:
        try:
            metadata["custom"] = json.loads(args.metadata_custom)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Error: Invalid JSON in --metadata-custom: {e}")

    # Create the task
    task = create_diy_task(
        task_dir=args.task_dir,
        prompt=prompt,
        benchmark_id=args.benchmark_id,
        dataset_id=args.dataset_id,
        task_id=task_id,
        categories=categories,
        system_message=system_message,
        metadata=metadata if metadata else None,
        rtl_as_solution=args.rtl_as_solution
    )

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Default: benchmark_datasets/<benchmark>/<dataset>.jsonl
        repo_root = Path(__file__).parent.parent
        output_file = (
            repo_root / "benchmark_datasets" / args.benchmark_id / f"{args.dataset_id}.jsonl"
        )

    # Append to JSONL file
    append_to_jsonl(task, output_file)

    # Print summary
    print(f"✓ DIY task created successfully!")
    print(f"  Task ID: {task['id']}")
    print(f"  Categories: {', '.join(task['categories'])}")
    print(f"  Context files: {len(task['context'])}")
    print(f"  Harness files: {len(task['harness'])}")
    print(f"  Patch files: {len(task['patch'])}")
    print(f"  Output: {output_file}")
    print(f"\nTask appended to: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Review the task in: {output_file}")
    print(f"  2. Build benchmark JSON: python src/build_benchmark_json.py {args.benchmark_id}")
    print(f"  3. Rebuild database: python src/build_datatable.py")


if __name__ == "__main__":
    main()
