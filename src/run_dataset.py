#!/usr/bin/env python3
"""
Description: File that will read the configuration in the run.yaml file for one or
multiple runs, then run the specified agent on the specified dataset, and save the
result to the results/tmp/ folder. Then it will extract the necessary data and save
it to the results/json/ directory

DS: Benchmark specific. Means that different datasets will require different implementations

--Run Benchmark--
Reads the run.yaml config file to determine the benchmark, dataset, and agent to run on
Ensures that the jsons exist for the agent, benchmark, dataset
Creates a run_id for the run
DS: Ensures that the json keys is created for all the tasks in the benchmark json
DS: Runs the benchmark (CVDP has a run script) with the parameters.
Sends the full results to the results/tmp/benchmark/dataset/run_id folder

--Collect Data--
Collect the run data from run_id folder to make a run_id run Json Entry
For each datapoint, create a datapoint_id datapoint json Entry
Save the run_id.json, and each datapoint_id json to a run_id.json file in
results/json/benchmark_name/dataset_name/run_id.json
"""

import json
import re
import shutil
import subprocess
import sys
import time
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import benchmark adapter system
from benchmarks import BenchmarkRegistry


# ============================================================================
# Docker Network Management
# ============================================================================

def create_docker_network(network_name: str) -> bool:
    """
    Create a Docker bridge network for this poRTLe run.
    
    Args:
        network_name: Name of the network to create
        
    Returns:
        True if network was created or already exists, False on failure
    """
    try:
        # Check if network already exists
        result = subprocess.run(
            ["docker", "network", "ls", "--filter", f"name=^{network_name}$", "--format", "{{.Name}}"],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip() == network_name:
            print(f"  Docker network '{network_name}' already exists")
            return True
        
        # Create the network
        print(f"  Creating Docker network: {network_name}")
        result = subprocess.run(
            ["docker", "network", "create", network_name, "--driver", "bridge"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  ✓ Docker network created: {network_name}")
            return True
        else:
            print(f"  ✗ Failed to create network: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error creating Docker network: {e}")
        return False


def remove_docker_network(network_name: str) -> bool:
    """
    Remove a Docker bridge network.
    
    Args:
        network_name: Name of the network to remove
        
    Returns:
        True if network was removed or didn't exist, False on failure
    """
    try:
        result = subprocess.run(
            ["docker", "network", "rm", network_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  ✓ Docker network removed: {network_name}")
            return True
        elif "not found" in result.stderr.lower():
            # Network doesn't exist, that's fine
            return True
        else:
            print(f"  ⚠ Warning: Could not remove network {network_name}: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ⚠ Warning: Error removing Docker network: {e}")
        return False


def generate_network_name(run_id: str) -> str:
    """
    Generate a unique Docker network name for this poRTLe run.
    
    Args:
        run_id: The run identifier
        
    Returns:
        A unique network name
    """
    # Add UUID suffix for extra uniqueness (in case of rapid re-runs)
    unique_suffix = uuid.uuid4().hex[:8]
    network_name = f"portle-{run_id}-{unique_suffix}"
    
    # Ensure network name is valid for Docker (alphanumeric, dash, underscore)
    network_name = ''.join(c if c.isalnum() or c in '-_' else '-' for c in network_name)
    
    # Docker has a 64 character limit
    if len(network_name) > 64:
        network_name = network_name[:64]
    
    return network_name


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


def get_custom_field(entry: Dict[str, Any], field: str, default: Any = None) -> Any:
    """
    Safely extract a value from the entry's custom metadata.

    Args:
        entry: Dictionary that may contain metadata/custom
        field: Field name inside metadata.custom
        default: Value to return if field is missing

    Returns:
        Field value or default
    """
    metadata = entry.get("metadata") or {}
    custom = metadata.get("custom") or {}
    if not isinstance(custom, dict):
        return default
    return custom.get(field, default)


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and parse run.yaml configuration file.

    Args:
        config_path: Path to run.yaml

    Returns:
        Configuration dictionary

    Required fields:
        - benchmark_id: Benchmark identifier
        - dataset_id: Dataset identifier
        - agent_id: Agent identifier (always required)
        - hardware_info: Hardware identifier
        - n: Number of runs per task
        - threads: Number of parallel threads

    Optional fields:
        - task_ids: List of specific task IDs to run (if omitted, runs all tasks)
        - metadata: Additional metadata for the run
            - metadata.custom.force_copilot: Boolean to force copilot mode for agentic datasets (CVDP-specific)
                When force_copilot is True, the model name is extracted from agent_config.custom.cvdp_llm_name
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields - agent_id is always required
    required_fields = ["benchmark_id", "dataset_id", "agent_id", "hardware_info", "n", "threads"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in run.yaml: {field}")

    return config


def load_agents_json(agents_path: Path) -> List[Dict[str, Any]]:
    """
    Load agents.json file.

    Args:
        agents_path: Path to agents.json

    Returns:
        List of agent dictionaries
    """
    if not agents_path.exists():
        raise FileNotFoundError(f"agents.json not found at: {agents_path}")

    with open(agents_path, 'r') as f:
        agents = json.load(f)

    return agents if isinstance(agents, list) else []


def load_benchmark_json(benchmark_path: Path) -> Dict[str, Any]:
    """
    Load benchmark JSON file.

    Args:
        benchmark_path: Path to benchmark JSON

    Returns:
        Benchmark dictionary
    """
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark JSON not found at: {benchmark_path}")

    with open(benchmark_path, 'r') as f:
        benchmark_data = json.load(f)

    return benchmark_data


def validate_configuration(config: Dict[str, Any], repo_root: Path) -> tuple:
    """
    Validate that all required JSONs exist and contain the specified IDs.

    Args:
        config: Configuration dictionary from run.yaml
        repo_root: Repository root path

    Returns:
        Tuple of (agent_data, dataset_data, tasks_list)
    """
    benchmark_id = config["benchmark_id"]
    dataset_id = config["dataset_id"]
    agent_id = config["agent_id"]

    # Load and validate agents.json (always required)
    agents_path = repo_root / "results" / "json" / "agents.json"
    agents = load_agents_json(agents_path)
    agent = next((a for a in agents if a["agent_id"] == agent_id), None)
    if not agent:
        raise ValueError(f"Agent '{agent_id}' not found in agents.json")

    # Load and validate benchmark JSON
    benchmark_path = repo_root / "results" / "json" / benchmark_id / f"{benchmark_id}_benchmark.json"
    benchmark_data = load_benchmark_json(benchmark_path)

    # Find dataset in benchmark
    dataset = next((d for d in benchmark_data["datasets"] if d["dataset_id"] == dataset_id), None)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_id}' not found in benchmark '{benchmark_id}'")

    # Get tasks list
    tasks = dataset.get("tasks", [])
    if not tasks:
        raise ValueError(f"No tasks found in dataset '{dataset_id}'")

    # Filter tasks if task_ids specified
    task_ids = config.get("task_ids")
    if task_ids:
        if not isinstance(task_ids, list):
            raise ValueError(f"task_ids must be a list, got {type(task_ids).__name__}")

        # Create a set of all valid task IDs for quick lookup
        valid_task_ids = {task["task_id"] for task in tasks}

        # Validate that all specified task_ids exist
        invalid_task_ids = [tid for tid in task_ids if tid not in valid_task_ids]
        if invalid_task_ids:
            raise ValueError(
                f"The following task_ids were not found in dataset '{dataset_id}':\n" +
                "\n".join(f"  - {tid}" for tid in invalid_task_ids)
            )

        # Filter tasks to only include specified task_ids
        tasks = [task for task in tasks if task["task_id"] in task_ids]
        print(f"  ✓ Task filtering: {len(tasks)}/{len(valid_task_ids)} tasks selected")

    return agent, dataset, tasks


def generate_run_id(agent_id: Optional[str], model_name: Optional[str], hardware_info: str) -> str:
    """
    Generate unique run ID based on timestamp, agent/model, and hardware.

    Args:
        agent_id: Agent identifier (None if using model)
        model_name: Model identifier (None if using agent)
        hardware_info: Hardware identifier

    Returns:
        Unique run ID (e.g., "11-15-25-143052-example-agent-dakota-macbook")
    """
    import re
    # Use agent_id or model_name
    identifier = agent_id if agent_id else model_name
    if not identifier:
        identifier = "unknown"

    # Sanitize identifier: replace spaces and special characters with hyphens
    # Keep only alphanumeric characters, hyphens, and underscores
    sanitized_agent = re.sub(r'[^a-zA-Z0-9_-]+', '-', identifier)
    # Remove leading/trailing hyphens and collapse multiple hyphens
    sanitized_agent = re.sub(r'-+', '-', sanitized_agent).strip('-')
    # Use lowercase for consistency
    sanitized_agent = sanitized_agent.lower()

    # Sanitize hardware_info: replace spaces and special characters with hyphens
    # Keep only alphanumeric characters, hyphens, and underscores
    sanitized_hardware = re.sub(r'[^a-zA-Z0-9_-]+', '-', hardware_info)
    # Remove leading/trailing hyphens and collapse multiple hyphens
    sanitized_hardware = re.sub(r'-+', '-', sanitized_hardware).strip('-')
    # Use lowercase for consistency
    sanitized_hardware = sanitized_hardware.lower()

    now = datetime.now()
    date_str = now.strftime("%m-%d-%y")
    time_str = now.strftime("%H%M%S")
    return f"{date_str}-{time_str}-{sanitized_agent}-{sanitized_hardware}"


def ensure_central_images(repo_root: Path):
    """
    Ensure central poRTLe Docker images are built.
    
    Checks if portle-agent-base and portle-oss-cad-suite images exist,
    and builds them if missing.
    
    Args:
        repo_root: Path to poRTLe repository root
    """
    central_images = ["portle-agent-base:latest", "portle-oss-cad-suite:latest"]
    missing = []
    
    for image in central_images:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            missing.append(image)
    
    if missing:
        print(f"  Building missing central images: {', '.join(missing)}")
        build_script = repo_root / "docker" / "build.sh"
        if build_script.exists():
            subprocess.run(["bash", str(build_script)], cwd=str(repo_root / "docker"), check=True)
            print("  ✓ Central images built")
        else:
            print(f"  ⚠ Warning: Central build script not found at {build_script}")
            print(f"    Missing images: {', '.join(missing)}")
    else:
        print("  ✓ Central images already exist")


def build_agent(agent_dir: Path, repo_root: Path = None):
    """
    Build agent Docker image using build_agent.sh script.

    Args:
        agent_dir: Path to agent directory
        repo_root: Path to poRTLe repository root (for central images)
    """
    # Ensure central images exist first
    if repo_root:
        ensure_central_images(repo_root)
    
    build_script = agent_dir / "build_agent.sh"
    if not build_script.exists():
        raise FileNotFoundError(f"Build script not found at: {build_script}")

    print(f"\n$ bash {build_script.name} (cwd={agent_dir})")
    subprocess.run(["bash", build_script.name], cwd=str(agent_dir), check=True)


def copy_agent_code(src: Path, dest: Path):
    """
    Copy agent code files to run directory in agent/ subfolder.

    Args:
        src: Source agent directory
        dest: Destination run directory
    """
    if not src.is_dir():
        raise FileNotFoundError(f"Agent directory not found: {src}")

    agent_dest = dest / "agent"
    ensure_dir(agent_dest)

    for name in ("agent.py", "prompts.py", "portle_config.json", "opencode.json"):
        source = src / name
        if source.exists():
            shutil.copyfile(source, agent_dest / name)

    nested_agent_src = src / "agent"
    if nested_agent_src.is_dir():
        shutil.copytree(nested_agent_src, agent_dest / "agent", dirs_exist_ok=True)


def run_task(task_id: str, dataset_jsonl: Path, agent_name: Optional[str], run_dir: Path,
             benchmark_runner_dir: Path, adapter, repo_root: Path, model: Optional[str] = None,
             force_copilot: bool = False, network_name: Optional[str] = None) -> tuple:
    """
    Run a single task using the benchmark runner.

    Args:
        task_id: Task identifier
        dataset_jsonl: Path to dataset JSONL file
        agent_name: Agent name for Docker image (None if using model)
        run_dir: Run directory path
        benchmark_runner_dir: Benchmark runner directory
        adapter: BenchmarkAdapter instance
        repo_root: Repository root path
        model: Model name for LLM mode (None if using agent)
        force_copilot: Whether to force copilot mode for agentic datasets
        network_name: Docker network name managed by poRTLe (for thread safety)

    Returns:
        Tuple of (passed: bool, output: str)
    """
    # Use adapter to run the task
    return adapter.run_task(
        task_id=task_id,
        dataset_path=dataset_jsonl,
        agent_name=agent_name,
        run_dir=run_dir,
        benchmark_runner_dir=benchmark_runner_dir,
        repo_root=repo_root,
        model=model,
        force_copilot=force_copilot,
        network_name=network_name
    )


def parse_token_count(run_dir: Path, task_id: str, adapter, attempt_index: int = 0,
                     log_path: Optional[str] = None) -> int:
    """
    Parse token count from agent log files using benchmark adapter.

    Args:
        run_dir: Run directory path
        task_id: Task identifier
        adapter: BenchmarkAdapter instance
        attempt_index: Zero-based attempt index
        log_path: Optional log path hint from raw_result.json (deprecated, kept for compatibility)

    Returns:
        Token count, or -1 if not found
    """
    # Use adapter to parse token count
    return adapter.parse_token_count(run_dir, task_id, attempt_index)


def load_raw_results(run_dir: Path) -> Dict[str, Any]:
    """
    Load the raw_result.json contents for a run directory.

    Args:
        run_dir: Run directory path

    Returns:
        Parsed raw results dictionary (empty if missing/invalid)
    """
    raw_result_path = run_dir / "raw_result.json"
    if not raw_result_path.exists():
        print(f"  Warning: raw_result.json not found at {raw_result_path}")
        return {}

    try:
        with open(raw_result_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        print(f"  Warning: raw_result.json at {raw_result_path} is not a JSON object")
    except Exception as e:
        print(f"  Warning: Could not parse raw_result.json: {e}")
        import traceback
        traceback.print_exc()
    return {}


def parse_raw_results(run_dir: Path, task_id: str, adapter, attempt_index: int = 0,
                      raw_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parse time, error, and score from raw results using benchmark adapter.

    Args:
        run_dir: Run directory path
        task_id: Task identifier
        adapter: BenchmarkAdapter instance
        attempt_index: Zero-based attempt index
        raw_results: Optional pre-loaded raw results dictionary (deprecated, kept for compatibility)

    Returns:
        Dictionary with time, error, score (defaults to -1, -1, 0.0 if not found)
    """
    # Use adapter to parse raw results
    return adapter.parse_raw_results(run_dir, task_id, attempt_index)


def create_datapoint(datapoint_id: str, benchmark_id: str, dataset_id: str,
                     task_id: str, agent_id: str, run_id: str,
                     passed: bool = None, execution_time: float = -1.0,
                     tokens: int = -1, error_code: int = 0,
                     score: float = None) -> Dict[str, Any]:
    """
    Create a datapoint entry.

    Args:
        datapoint_id: Unique datapoint identifier
        benchmark_id: Benchmark ID
        dataset_id: Dataset ID
        task_id: Task ID
        agent_id: Agent ID
        run_id: Run ID
        passed: Whether the task passed (deprecated, use score instead)
        execution_time: Execution time in seconds
        tokens: Token count
        error_code: Error code (0 for no error)
        score: Score as a float (0.0 to 1.0). If not provided, uses passed boolean.

    Returns:
        Datapoint dictionary
    """
    # Determine score: prefer explicit score, fall back to passed boolean
    if score is not None:
        final_score = score
    elif passed is not None:
        final_score = 1.0 if passed else 0.0
    else:
        final_score = 0.0

    return {
        "datapoint_id": datapoint_id,
        "benchmark_id": benchmark_id,
        "dataset_id": dataset_id,
        "task_id": task_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "tokens": tokens,
        "time": execution_time,
        "error": error_code,
        "score": final_score,
        "metadata": create_metadata()
    }


def save_run_json(run_data: Dict[str, Any], output_path: Path):
    """
    Save run JSON to file.

    Args:
        run_data: Run data dictionary
        output_path: Output file path
    """
    ensure_dir(output_path.parent)
    with open(output_path, 'w') as f:
        json.dump(run_data, f, indent=2)


def main():
    """Main function to orchestrate benchmark run."""
    print("\n" + "=" * 60)
    print("poRTLe Dataset Runner")
    print("=" * 60)

    # Get paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    config_path = script_dir / "run.yaml"

    # ===== PHASE 1: SETUP & VALIDATION =====
    print("\n[PHASE 1] Setup & Validation")
    print("-" * 60)

    print("Loading configuration from run.yaml...")
    config = load_config(config_path)
    print(f"  Benchmark: {config['benchmark_id']}")
    print(f"  Dataset:   {config['dataset_id']}")
    print(f"  Agent:     {config['agent_id']}")

    # Get force_copilot from metadata.custom (CVDP-specific setting)
    metadata = config.get("metadata", {})
    custom = metadata.get("custom", {})
    force_copilot = custom.get("force_copilot", False)

    if force_copilot:
        print(f"  Mode:      LLM (force-copilot)")
    else:
        print(f"  Mode:      Agent")

    print(f"  Hardware:  {config['hardware_info']}")
    print(f"  Runs/task: {config['n']}")

    print("\nValidating JSONs exist...")
    agent, dataset, tasks = validate_configuration(config, repo_root)
    source_dataset_id = get_custom_field(dataset, "source_dataset_id", dataset["dataset_id"])

    print(f"  ✓ Agent found: {agent['about']}")
    print(f"  ✓ Dataset found: {dataset['task_count']} tasks")
    print(f"  ✓ Tasks loaded: {len(tasks)} tasks")

    # Extract agent info (always available)
    agent_folder_path = agent["agent_config"]["agent_folder_path"]
    agent_dir = repo_root / agent_folder_path if agent_folder_path != "none" else None
    # Extract agent name from path (e.g., "agents/example-agent" -> "example-agent")
    agent_name = Path(agent_folder_path).name if agent_folder_path != "none" else None

    # Write agent_config to portle_config.json in agent directory
    if agent_dir:
        agent_config = agent.get("agent_config", {})
        portle_config_path = agent_dir / "portle_config.json"
        with open(portle_config_path, 'w') as f:
            json.dump(agent_config, f, indent=2)
        print(f"  ✓ Written agent config to {portle_config_path}")

    # Generate run ID
    agent_id = config["agent_id"]
    run_id = generate_run_id(agent_id, None, config["hardware_info"])
    run_start = datetime.now()
    print(f"\nGenerated run_id: {run_id}")
    print(f"Run started: {run_start.isoformat()}")

    # ===== PHASE 2: BUILD AGENT =====
    print("\n[PHASE 2] Build Agent")
    print("-" * 60)

    if agent_dir and not force_copilot:
        print(f"Building agent from: {agent_dir}")
        build_agent(agent_dir, repo_root)
        print("  ✓ Agent built successfully")
    else:
        print("Skipping agent build (LLM mode or no agent folder)")

    # Setup run directory
    run_dir = repo_root / "results" / "tmp" / config["benchmark_id"] / config["dataset_id"] / run_id
    ensure_dir(run_dir)
    print(f"\nRun directory: {run_dir}")

    if agent_dir and not force_copilot:
        print("Copying agent code to run directory...")
        copy_agent_code(agent_dir, run_dir)
        print("  ✓ Agent code copied")

    # ===== PHASE 3: RUN BENCHMARK =====
    print("\n[PHASE 3] Run Benchmark")
    print("-" * 60)

    # Get dataset JSONL file
    dataset_jsonl = repo_root / "benchmark_datasets" / config["benchmark_id"] / f"{source_dataset_id}.jsonl"
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"Dataset JSONL not found: {dataset_jsonl}")

    # Get benchmark adapter (default to CVDP for backward compatibility)
    benchmark_name = config.get("benchmark_id", "CVDP")
    adapter_name = config.get("adapter", benchmark_name.lower())
    try:
        adapter = BenchmarkRegistry.get_adapter(adapter_name)
        print(f"Using benchmark adapter: {adapter_name}")
    except KeyError as e:
        print(f"ERROR: {e}")
        print(f"Defaulting to 'cvdp' adapter for backward compatibility")
        adapter = BenchmarkRegistry.get_adapter("cvdp")

    # Get benchmark runner directory from adapter
    benchmark_runner_dir = repo_root / "benchmark_runners" / adapter.get_benchmark_runner_dir()
    if not benchmark_runner_dir.exists():
        raise FileNotFoundError(f"Benchmark runner not found: {benchmark_runner_dir}")

    # Extract model name from agent metadata if using force_copilot
    model_name = None
    if force_copilot:
        model_name = adapter.get_model_for_agent(agent, force_copilot)
        print(f"  Using model: {model_name} (from agent_config.custom.cvdp_llm_name)")

    # Run all tasks
    datapoints = []
    n_runs = config["n"]
    num_threads = config.get("threads", 1)

    print(f"\nRunning {len(tasks)} tasks (n={n_runs} per task) with {num_threads} threads...")
    
    # Create a shared Docker network for all tasks in this run
    # This prevents race conditions where one task's subprocess cleans up
    # the network while other tasks are still using it
    network_name = None
    if num_threads > 1:
        network_name = generate_network_name(run_id)
        print(f"\nSetting up shared Docker network for parallel execution...")
        if not create_docker_network(network_name):
            print("  ⚠ Warning: Could not create shared network, tasks will create their own")
            network_name = None
    
    import concurrent.futures
    
    # Helper function for single task run execution
    def process_single_run(task_idx, task, run_num):
        task_id = task["task_id"]
        source_task_id = get_custom_field(task, "source_task_id", task_id)
        display_task_id = source_task_id if source_task_id != task_id else task_id
        
        # Thread-safe printing
        msg = f"\n[Task {task_idx+1}/{len(tasks)}] {display_task_id}"
        if n_runs > 1:
            msg += f" (Run {run_num + 1}/{n_runs})"
        print(msg)

        # Create sample-specific run directory
        # If n > 1, use subdirectories like sample_0, sample_1
        # If n = 1, use the base dir to maintain backward compatibility
        if n_runs > 1:
            sample_dir = run_dir / f"sample_{run_num}"
            ensure_dir(sample_dir)
        else:
            sample_dir = run_dir

        # Run the task with timing
        task_start_time = time.time()

        passed, output = run_task(
            task_id=source_task_id,
            dataset_jsonl=dataset_jsonl,
            agent_name=agent_name,
            run_dir=sample_dir,  # Use sample-specific dir or base dir
            benchmark_runner_dir=benchmark_runner_dir,
            adapter=adapter,
            repo_root=repo_root,
            model=model_name,
            force_copilot=force_copilot,
            network_name=network_name  # Pass poRTLe-managed network
        )

        task_end_time = time.time()
        agent_execution_time = task_end_time - task_start_time

        # Parse actual metrics from output files in the sample directory
        print(f"  Parsing metrics for {display_task_id}...")
        raw_results = parse_raw_results(sample_dir, source_task_id, adapter)
        tokens = parse_token_count(sample_dir, source_task_id, adapter,
                                 log_path=raw_results.get("log_path"))

        # Create datapoint entry with actual metrics
        # Use score from raw_results which handles multiple tests correctly
        datapoint_id = f"{run_id}-{task_id}-{run_num:03d}"
        datapoint = create_datapoint(
            datapoint_id=datapoint_id,
            benchmark_id=config["benchmark_id"],
            dataset_id=config["dataset_id"],
            task_id=task_id,
            agent_id=config.get("agent_id"),
            run_id=run_id,
            score=raw_results["score"],
            execution_time=agent_execution_time,
            tokens=tokens,
            error_code=raw_results["error"]
        )
        metadata = datapoint.get("metadata") or create_metadata()
        datapoint["metadata"] = metadata
        custom_meta = metadata.get("custom") or {}
        metadata["custom"] = custom_meta
        custom_meta["source_task_id"] = source_task_id
        custom_meta["source_dataset_id"] = source_dataset_id
        custom_meta["sample_index"] = run_num
        custom_meta["datapoint_directory"] = str(sample_dir)
        custom_meta["test_harness_time"] = raw_results.get("time", -1.0)
        # Surface per-metric scores (e.g., notsotiny syntax/equivalence/functionality) if provided
        if raw_results.get("component_scores"):
            custom_meta["component_scores"] = raw_results["component_scores"]
            custom_meta["syntax_score"] = raw_results["component_scores"].get("syntax")
            custom_meta["equivalence_score"] = raw_results["component_scores"].get("equivalence")
            custom_meta["functionality_score"] = raw_results["component_scores"].get("functionality")
        
        # Print result with metrics
        score = raw_results["score"]
        result_str = "PASS ✓" if score == 1.0 else f"PARTIAL ({score:.2f})" if score > 0 else "FAIL ✗"
        print(f"  {display_task_id} (Run {run_num}) Result: {result_str}")
        print(f"  {display_task_id} (Run {run_num}) Tokens: {tokens if tokens != -1 else 'N/A'}")
        print(f"  {display_task_id} (Run {run_num}) Agent Time: {agent_execution_time:.2f}s")
        print(f"  {display_task_id} (Run {run_num}) Score: {raw_results['score']}")
        
        return datapoint

    # Create list of all run configurations
    run_configs = []
    for i, task in enumerate(tasks):
        for run_num in range(n_runs):
            run_configs.append((i, task, run_num))

    # Execute in parallel
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_single_run, i, task, run_num) 
                      for i, task, run_num in run_configs]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    datapoint = future.result()
                    datapoints.append(datapoint)
                except Exception as exc:
                    print(f"Task generated an exception: {exc}")
                    import traceback
                    traceback.print_exc()
    finally:
        # Clean up the shared Docker network after all tasks complete
        if network_name:
            print(f"\nCleaning up shared Docker network...")
            remove_docker_network(network_name)

    # ===== PHASE 4: COLLECT RESULTS =====
    print("\n[PHASE 4] Collect Results")
    print("-" * 60)

    run_end = datetime.now()
    total_time = (run_end - run_start).total_seconds()

    print(f"Run completed: {run_end.isoformat()}")
    print(f"Total time: {total_time:.2f}s")

    # Calculate summary statistics
    total_datapoints = len(datapoints)
    passed_datapoints = sum(1 for dp in datapoints if dp["score"] == 1.0)
    print(f"\nResults: {passed_datapoints}/{total_datapoints} passed")

    # Determine if this is a partial run (subset of tasks)
    total_tasks_in_dataset = dataset.get("task_count", len(tasks))
    is_partial_run = len(tasks) < total_tasks_in_dataset

    # Get metadata from config (force_copilot is already in metadata.custom if it was set)
    run_metadata = config.get("metadata", create_metadata())

    # Create run JSON structure
    run_data = {
        "run_id": run_id,
        "benchmark_id": config["benchmark_id"],
        "dataset_id": config["dataset_id"],
        "agent_id": config["agent_id"],  # Always present (never None)
        "hardware_info": config["hardware_info"],
        "n": config["n"],
        "threads": config.get("threads", 1),
        "is_partial_run": is_partial_run,
        "run_start": run_start.isoformat(),
        "run_end": run_end.isoformat(),
        "total_time": f"{total_time:.2f}s",
        "run_directory": str(run_dir),
        "metadata": run_metadata,
        "datapoints": datapoints
    }

    # Save run JSON
    output_path = repo_root / "results" / "json" / config["benchmark_id"] / config["dataset_id"] / f"{run_id}.json"
    save_run_json(run_data, output_path)
    print(f"\nRun JSON saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Run completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit(1)
