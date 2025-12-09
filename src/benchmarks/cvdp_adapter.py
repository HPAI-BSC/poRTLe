"""
CVDP Benchmark Adapter.

This module provides CVDP-specific implementation of the BenchmarkAdapter interface.
It handles all CVDP-specific logic including:
- JSONL parsing with CVDP's categories format
- Task execution with CVDP's Docker runner
- Token counting from CVDP agent logs
- Result parsing from CVDP's raw_result.json format
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from .base import BenchmarkAdapter


class CVDPAdapter(BenchmarkAdapter):
    """
    CVDP-specific benchmark adapter.

    CVDP (Code Verification and Debugging Platform) specific details:
    - Uses JSONL files with 'categories' field: [cid, difficulty]
    - Task IDs contain workflow type (agentic/non-agentic)
    - Token counts logged as "Total Final Tokens: N"
    - Task directories named without _NNNN suffix
    - Results in raw_result.json with specific structure
    """

    # CVDP-specific patterns and constants
    BENCHMARK_RUNNER_DIR = "cvdp_benchmark"
    TOKEN_PATTERN = r'Total\s+Final\s+Tokens:\s+(\d+)'
    TASK_DIR_PATTERN = r'_\d{4}$'

    def build_benchmark_json(
        self,
        dataset_dir: Path,
        output_dir: Path,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Build CVDP benchmark JSON from JSONL dataset files.

        Args:
            dataset_dir: Directory containing .jsonl files
            output_dir: Directory for output JSON
            metadata: Benchmark-level metadata

        Returns:
            Path to generated benchmark JSON file
        """
        benchmark_name = metadata.get("benchmark_id", "CVDP")

        # Initialize benchmark structure
        benchmark_data = {
            "benchmark_id": benchmark_name,
            "metadata": self._create_metadata(
                keys=metadata.get("keys", []),
                notes=metadata.get("notes", []),
                custom=metadata.get("custom", {})
            ),
            "datasets": []
        }

        # Find all JSONL files
        jsonl_files = sorted(dataset_dir.glob("*.jsonl"))
        if not jsonl_files:
            raise ValueError(f"No .jsonl files found in {dataset_dir}")

        print(f"Found {len(jsonl_files)} dataset file(s):\n")

        # Process each dataset file
        for jsonl_file in jsonl_files:
            print(f"Processing: {jsonl_file.name}")

            dataset_slug = jsonl_file.stem
            dataset_id = f"{benchmark_name}__{dataset_slug}"

            # Parse JSONL tasks
            tasks = self._parse_jsonl_file(jsonl_file)
            task_count = len(tasks)

            # Detect CVDP-specific metadata
            commercial = self._detect_commercial(dataset_slug)
            workflow = self._detect_workflow(tasks)

            # Build dataset structure
            dataset = {
                "dataset_id": dataset_id,
                "benchmark_id": benchmark_name,
                "commercial": commercial,
                "task_count": task_count,
                "metadata": self._create_metadata(custom={
                    "cvdp_workflow": workflow,
                    "source_dataset_id": dataset_slug
                }),
                "tasks": []
            }

            # Build task structures
            for task in tasks:
                original_task_id = task.get("id", "")
                if not original_task_id:
                    continue

                task_id = f"{dataset_id}__{original_task_id}"

                # Extract CVDP-specific metadata from categories
                task_metadata = self.get_task_metadata(task)

                task_data = {
                    "task_id": task_id,
                    "benchmark_id": benchmark_name,
                    "dataset_id": dataset_id,
                    "metadata": self._create_metadata(custom={
                        **task_metadata,
                        "source_task_id": original_task_id
                    })
                }

                dataset["tasks"].append(task_data)

            print(f"  Created dataset: {dataset_id} ({task_count} tasks, {workflow}, commercial={commercial})")
            benchmark_data["datasets"].append(dataset)

        # Save to JSON file
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{benchmark_name}_benchmark.json"
        with open(json_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)

        print(f"\n=== JSON created successfully ===")
        print(f"Location: {json_path}")
        print(f"\nStructure:")
        print(f"  Benchmark: {benchmark_name}")
        print(f"  Datasets: {len(benchmark_data['datasets'])}")
        total_tasks = sum(len(ds['tasks']) for ds in benchmark_data['datasets'])
        print(f"  Total tasks: {total_tasks}")

        return json_path

    def run_task(
        self,
        task_id: str,
        dataset_path: Path,
        agent_name: Optional[str],
        run_dir: Path,
        benchmark_runner_dir: Path,
        repo_root: Path,
        model: Optional[str] = None,
        force_copilot: bool = False,
        network_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, str]:
        """
        Execute a CVDP task using the CVDP benchmark runner.

        Supports two modes:
        1. Agent mode: Uses Docker agent with -g flag
        2. LLM mode: Uses LLM with -l and -m flags, optionally with --force-copilot

        Args:
            task_id: Task identifier (source task ID without namespace)
            dataset_path: Path to dataset JSONL file
            agent_name: Agent name for Docker image (None if using model)
            run_dir: Directory for storing results
            benchmark_runner_dir: CVDP benchmark runner directory
            repo_root: Repository root (unused for CVDP)
            model: Model name for LLM mode (None if using agent)
            force_copilot: Whether to force copilot mode for agentic datasets
            network_name: Docker network name managed by poRTLe (prevents cleanup race conditions)
            **kwargs: Additional arguments (unused for CVDP)

        Returns:
            Tuple of (success: bool, output: str)
        """
        # Prefer the dedicated cvdp_env virtual environment if it exists
        runner_env = benchmark_runner_dir / "cvdp_env"
        env_python = None
        unix_python = runner_env / "bin" / "python"
        win_python = runner_env / "Scripts" / "python.exe"
        if unix_python.exists():
            env_python = unix_python
        elif win_python.exists():
            env_python = win_python

        base_cmd = ["./run_benchmark.py"]
        if env_python:
            # Execute the runner through the venv interpreter so dependencies (e.g., nltk) are available
            base_cmd = [str(env_python), "run_benchmark.py"]

        # Build command based on mode (agent vs LLM)
        cmd = base_cmd + [
            "-f", str(dataset_path.resolve()),
            "-i", task_id,
            "-p", str(run_dir.resolve()),
        ]

        # Add mode-specific flags
        if model:
            # LLM mode
            cmd.extend(["-l", "-m", model])
            if force_copilot:
                cmd.append("--force-copilot")
        elif agent_name:
            # Agent mode
            if task_id.startswith("cvdp_copilot"):
                cmd.append("--force-agentic")
            cmd.extend(["-l", "-g", f"cvdp-{agent_name}:latest"])
        else:
            raise ValueError("Either agent_name or model must be specified")

        # If poRTLe is managing the network (for thread safety), tell CVDP to use it
        # and not to create/cleanup its own network
        if network_name:
            cmd.extend(["--external-network", "--network-name", network_name])

        print(f"\n$ {' '.join(cmd)} (cwd={benchmark_runner_dir})")
        process = subprocess.Popen(
            cmd,
            cwd=str(benchmark_runner_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Capture output and check for pass/fail
        output_lines = []
        found_zero_result = False

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)
            if '"result": 0' in line:
                found_zero_result = True

        return_code = process.wait()
        if return_code != 0:
            print(f"WARNING: Process returned non-zero exit code: {return_code}")

        return found_zero_result, "".join(output_lines)

    def parse_token_count(
        self,
        run_dir: Path,
        task_id: str,
        attempt_index: int
    ) -> int:
        """
        Parse token count from CVDP agent.txt log files.

        Supports multiple formats:
        - Standard CVDP: "Total Final Tokens: 42"
        - Codex format: "tokens used\n84,406"
        - OpenCode format: "Total tokens: 9,965"

        Note: ANSI escape codes are automatically stripped before parsing.

        Args:
            run_dir: Run directory path
            task_id: Task identifier
            attempt_index: Attempt number (0-indexed)

        Returns:
            Token count, or -1 if not found
        """
        # Get task directory name (CVDP removes _NNNN suffix)
        task_dir_name = self.get_task_directory_name(task_id)

        # Look for agent.txt files in CVDP's reports directory
        reports_dir = run_dir / task_dir_name / "reports"
        if not reports_dir.exists():
            return -1

        # CVDP uses 1-indexed file names: 1_agent.txt, 2_agent.txt, etc.
        run_number = attempt_index + 1
        agent_file = reports_dir / f"{run_number}_agent.txt"

        if not agent_file.exists():
            # Fallback to any agent file
            agent_files = sorted(reports_dir.glob("*_agent.txt"))
            if not agent_files:
                return -1
            agent_file = agent_files[0]

        # Define patterns to try
        patterns = [
            # Standard CVDP: "Total Final Tokens: 42"
            r'Total\s+Final\s+Tokens:\s+(\d+)',
            # Codex format: "tokens used\n84,406"
            r'tokens used\s*\n\s*([\d,]+)',
            # OpenCode format: "Total tokens: 9,965"
            r'Total\s+tokens:\s*([\d,]+)',
        ]

        try:
            with open(agent_file, 'r') as f:
                content = f.read()

                # Strip ANSI escape codes for pattern matching
                ansi_escape = re.compile(r'\x1b\[[0-9;]+m')
                content_clean = ansi_escape.sub('', content)

                for pattern in patterns:
                    match = re.search(pattern, content_clean, re.IGNORECASE | re.MULTILINE)
                    if match:
                        # Extract string, remove commas, convert to int
                        token_str = match.group(1).replace(',', '')
                        return int(token_str)
                        
        except Exception as e:
            print(f"  Warning: Could not parse tokens from {agent_file}: {e}")

        return -1

    def parse_raw_results(
        self,
        run_dir: Path,
        task_id: str,
        attempt_index: int
    ) -> Dict[str, Any]:
        """
        Parse execution results from CVDP's raw_result.json.

        CVDP format:
        {
          "task_id": {
            "errors": N,
            "tests": [
              {"execution": time, "result": 0/1, "log": "path"},
              ...
            ]
          }
        }

        A task is considered passed (score=1.0) only if ALL tests in the
        tests array have result=0. If any test has a non-zero result,
        the task is marked as failed (score=0.0).

        Args:
            run_dir: Run directory path
            task_id: Task identifier
            attempt_index: Attempt number (0-indexed, unused - we check all tests)

        Returns:
            Dictionary with time, error, score, log_path
        """
        result = {
            "time": -1.0,
            "error": -1,
            "score": 0.0,
            "log_path": None,
            # Optional per-metric breakdown for benchmarks like notsotiny
            "component_scores": None,
        }

        # Load CVDP's raw_result.json
        raw_result_path = run_dir / "raw_result.json"
        if not raw_result_path.exists():
            print(f"  Warning: raw_result.json not found at {raw_result_path}")
            return result

        try:
            with open(raw_result_path, 'r') as f:
                raw_results = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not parse raw_result.json: {e}")
            return result

        # Extract task-specific data
        task_data = raw_results.get(task_id) if isinstance(raw_results, dict) else None
        if not isinstance(task_data, dict):
            return result

        # Parse error count
        if "errors" in task_data:
            try:
                result["error"] = int(task_data["errors"])
            except (TypeError, ValueError):
                pass

        # Parse test results
        tests = task_data.get("tests") or []
        if tests:
            # Use the first test entry for log_path reference
            result["log_path"] = tests[0].get("log")

            # Sum execution time across all tests
            total_time = 0.0
            for test_entry in tests:
                try:
                    if "execution" in test_entry:
                        total_time += float(test_entry["execution"])
                except (TypeError, ValueError):
                    pass
            result["time"] = total_time

            # Calculate score as average of all tests (result=0 means pass=1.0, result!=0 means pass=0.0)
            passed_count = 0
            total_tests = 0
            for test_entry in tests:
                if "result" in test_entry:
                    try:
                        test_result = int(test_entry["result"])
                        total_tests += 1
                        if test_result == 0:
                            passed_count += 1
                    except (TypeError, ValueError):
                        pass

            if total_tests > 0:
                result["score"] = passed_count / total_tests
            else:
                result["score"] = 0.0

            # If we have a harness log, try to extract per-metric pass/fail (e.g., syntax/equivalence/functionality)
            component_scores = self._parse_component_scores(result["log_path"])
            if component_scores:
                # Normalize to integers for downstream consumers
                component_scores = {k: 1 if v else 0 for k, v in component_scores.items()}
                # Average the 3 metrics equally to produce a single score out of 1
                result["component_scores"] = component_scores
                result["score"] = sum(1 if v else 0 for v in component_scores.values()) / len(component_scores)

        return result

    def _parse_component_scores(self, log_path: Optional[str]) -> Optional[Dict[str, bool]]:
        """Parse per-metric pass/fail from a notsotiny harness log.

        We look for lines like "syntax: pass", "equivalence=fail", or the summary
        "Summary: syntax=pass, equivalence=pass, functionality=pass". Returns a
        dict mapping metric name -> bool if any metrics are found.
        """
        if not log_path:
            return None

        log_file = Path(log_path)
        if not log_file.exists():
            return None

        metrics: Dict[str, bool] = {}
        pattern = re.compile(r"(syntax|equivalence|functionality)\s*[:=]\s*(pass|fail)", re.IGNORECASE)

        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    for match in pattern.finditer(line):
                        name = match.group(1).lower()
                        status = match.group(2).lower() == "pass"
                        metrics[name] = status

            # Only return if we found at least one of the three metrics
            if metrics:
                # Ensure all three known metrics are present if possible
                for key in ["syntax", "equivalence", "functionality"]:
                    if key not in metrics:
                        # Leave missing entries untouched; caller can still use the subset
                        continue
                return metrics
        except Exception as e:
            print(f"  Warning: Failed to parse component scores from {log_file}: {e}")

        return None

    def get_task_metadata(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract CVDP-specific metadata from task definition.

        CVDP tasks have a 'categories' array: [cid, difficulty]

        Args:
            task: Task dictionary from JSONL

        Returns:
            Dictionary with cvdp_cid and cvdp_difficulty
        """
        categories = task.get("categories", [])
        cid = categories[0] if len(categories) > 0 else "unknown"
        difficulty = categories[1] if len(categories) > 1 else "unknown"

        # Special handling for VerilogEval (cid021): infer subtype from task_id prefix
        if cid == "cid021":
            task_id = task.get("id", "")
            verilogeval_type = None
            if "Spec2RTL" in task_id:
                verilogeval_type = "spec-2-rtl"
            elif "Completion" in task_id:
                verilogeval_type = "code-completion"

            return {
                "cvdp_cid": cid,
                "cvdp_difficulty": difficulty,
                "verilogeval_type": verilogeval_type,
            }

        return {
            "cvdp_cid": cid,
            "cvdp_difficulty": difficulty,
        }

    def get_dataset_metadata(
        self,
        dataset_name: str,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract CVDP-specific metadata from dataset.

        Args:
            dataset_name: Dataset identifier
            tasks: List of task dictionaries

        Returns:
            Dictionary with cvdp_workflow and commercial status
        """
        return {
            "cvdp_workflow": self._detect_workflow(tasks),
            "commercial": self._detect_commercial(dataset_name),
        }

    def get_benchmark_runner_dir(self) -> str:
        """Get the CVDP benchmark runner directory name."""
        return self.BENCHMARK_RUNNER_DIR

    def get_task_directory_name(self, task_id: str) -> str:
        """
        Get directory name for CVDP task.

        CVDP removes the _NNNN suffix from task IDs for directory naming.
        E.g., "cvdp_agentic_foo_0001" -> "cvdp_agentic_foo"

        Args:
            task_id: Original task identifier

        Returns:
            Directory name without numeric suffix
        """
        return re.sub(self.TASK_DIR_PATTERN, '', task_id)

    @staticmethod
    def get_model_for_agent(agent_config: Dict[str, Any], force_copilot: bool) -> Optional[str]:
        """
        Extract CVDP model name from agent metadata when using force_copilot mode.

        This is a CVDP-specific feature that allows LLM-only agents to specify
        which model to use via the 'cvdp_llm_name' field in agent_config.custom.

        Args:
            agent_config: Agent configuration dictionary from agents.json
            force_copilot: Whether force_copilot mode is enabled

        Returns:
            Model name string if force_copilot is True, None otherwise

        Raises:
            ValueError: If force_copilot is True but cvdp_llm_name is not set
        """
        if not force_copilot:
            return None

        # Extract cvdp_llm_name from agent config custom fields
        cvdp_llm_name = agent_config.get("agent_config", {}).get("custom", {}).get("cvdp_llm_name")

        if not cvdp_llm_name:
            agent_id = agent_config.get("agent_id", "unknown")
            raise ValueError(
                f"LLM Mode (force_copilot) requires 'cvdp_llm_name' in agent_config.custom.\n"
                f"Agent '{agent_id}' does not have this field set.\n"
                f"Please add cvdp_llm_name to the agent's custom config in agents.json."
            )

        return cvdp_llm_name

    # Private helper methods for CVDP-specific logic

    @staticmethod
    def _create_metadata(
        keys: List[str] = None,
        notes: List[Dict] = None,
        custom: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create standard metadata dictionary."""
        return {
            "keys": keys or [],
            "notes": notes or [],
            "custom": custom or {}
        }

    @staticmethod
    def _parse_jsonl_file(jsonl_path: Path) -> List[Dict[str, Any]]:
        """Parse JSONL file and return list of task dictionaries."""
        tasks = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks

    @staticmethod
    def _detect_commercial(dataset_id: str) -> str:
        """
        Detect if CVDP dataset is commercial based on naming convention.

        Args:
            dataset_id: Dataset identifier

        Returns:
            "yes" or "no"
        """
        if "no_commercial" in dataset_id.lower() or "non_commercial" in dataset_id.lower():
            return "no"
        return "yes"

    @staticmethod
    def _detect_workflow(tasks: List[Dict[str, Any]]) -> str:
        """
        Detect CVDP workflow type (agentic vs non-agentic) from task IDs.

        Args:
            tasks: List of task dictionaries

        Returns:
            "agentic" or "non-agentic"
        """
        if not tasks:
            return "non-agentic"

        # Check first task ID for "agentic" keyword
        first_task_id = tasks[0].get("id", "")
        if "agentic" in first_task_id.lower():
            return "agentic"
        return "non-agentic"
