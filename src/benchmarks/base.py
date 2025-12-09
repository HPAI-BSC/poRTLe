"""
Base class for benchmark adapters.

This module defines the abstract interface that all benchmark-specific adapters
must implement. Each benchmark framework (CVDP, TuRTLe, etc.) should create
a concrete implementation of BenchmarkAdapter.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class BenchmarkAdapter(ABC):
    """
    Abstract base class for benchmark-specific adapters.

    Each benchmark framework should implement this interface to provide
    benchmark-specific logic for:
    - Building benchmark JSON from dataset files
    - Executing tasks
    - Parsing results and token counts
    - Extracting metadata
    """

    @abstractmethod
    def build_benchmark_json(
        self,
        dataset_dir: Path,
        output_dir: Path,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Build benchmark JSON structure from dataset files.

        Args:
            dataset_dir: Directory containing the raw dataset files
            output_dir: Directory where benchmark JSON should be written
            metadata: Additional metadata to include in the benchmark JSON

        Returns:
            Path to the generated benchmark JSON file
        """
        pass

    @abstractmethod
    def run_task(
        self,
        task_id: str,
        dataset_path: Path,
        agent_name: str,
        run_dir: Path,
        benchmark_runner_dir: Path,
        repo_root: Path,
        **kwargs
    ) -> Tuple[bool, str]:
        """
        Execute a single benchmark task.

        Args:
            task_id: Unique identifier for the task
            dataset_path: Path to the benchmark JSON file
            agent_name: Name of the agent being tested
            run_dir: Directory for storing run results
            benchmark_runner_dir: Directory containing benchmark runner scripts
            repo_root: Root directory of the repository
            **kwargs: Additional benchmark-specific arguments

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        pass

    @abstractmethod
    def parse_token_count(
        self,
        run_dir: Path,
        task_id: str,
        attempt_index: int
    ) -> int:
        """
        Parse token count from task execution logs.

        Args:
            run_dir: Directory containing run results
            task_id: Task identifier
            attempt_index: Attempt number (1-indexed)

        Returns:
            Total token count, or 0 if not found
        """
        pass

    @abstractmethod
    def parse_raw_results(
        self,
        run_dir: Path,
        task_id: str,
        attempt_index: int
    ) -> Dict[str, Any]:
        """
        Parse execution results from task run.

        Args:
            run_dir: Directory containing run results
            task_id: Task identifier
            attempt_index: Attempt number (1-indexed)

        Returns:
            Dictionary containing parsed results (format varies by benchmark)
        """
        pass

    @abstractmethod
    def get_task_metadata(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract benchmark-specific metadata from a task definition.

        Args:
            task: Task dictionary from the dataset

        Returns:
            Dictionary of benchmark-specific metadata fields
        """
        pass

    @abstractmethod
    def get_dataset_metadata(
        self,
        dataset_name: str,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract benchmark-specific metadata from a dataset.

        Args:
            dataset_name: Name of the dataset
            tasks: List of task dictionaries

        Returns:
            Dictionary of benchmark-specific dataset metadata
        """
        pass

    def get_benchmark_runner_dir(self) -> str:
        """
        Get the name of the benchmark runner directory.

        Returns:
            Name of the directory (relative to benchmark_runners/)
        """
        return self.__class__.__name__.replace("Adapter", "").lower() + "_benchmark"

    def get_task_directory_name(self, task_id: str) -> str:
        """
        Get the directory name for a specific task.

        Some benchmarks may modify task IDs for directory naming (e.g., removing suffixes).
        Default implementation returns the task_id as-is.

        Args:
            task_id: Original task identifier

        Returns:
            Directory name to use for this task
        """
        return task_id
