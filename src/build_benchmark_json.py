#!/usr/bin/env python3
"""
Description: Build a benchmark JSON from dataset files using benchmark adapters.

This is a generic benchmark JSON builder that works with any registered benchmark adapter.
The specific logic for parsing dataset files and extracting metadata is handled by
the benchmark-specific adapter (e.g., CVDPAdapter, TuRTLeAdapter, etc.).

Assumptions:
- Dataset files live in benchmark_datasets/<benchmark_name>
- Output saved to results/json/<benchmark_name>/<benchmark_name>_benchmark.json
- Dataset and task IDs are namespaced with the benchmark/dataset to ensure uniqueness
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any

from benchmarks import BenchmarkRegistry


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build benchmark JSON files from dataset files using benchmark adapters",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("benchmark_name", type=str, help="Name for the benchmark (e.g., CVDP, TuRTLe)")
    parser.add_argument("--adapter", type=str, default=None,
                       help="Adapter to use (default: benchmark_name.lower()). Available: " +
                            ", ".join(BenchmarkRegistry.list_adapters()))
    parser.add_argument("--metadata-keys", type=str, default="[]",
                       help="Metadata keys as JSON array string")
    parser.add_argument("--metadata-notes", type=str, default="[]",
                       help="Metadata notes as JSON array string")
    parser.add_argument("--metadata-custom", type=str, default="{}",
                       help="Metadata custom as JSON string")

    args = parser.parse_args()

    benchmark_name = args.benchmark_name
    adapter_name = args.adapter or benchmark_name.lower()

    # Parse metadata from JSON strings
    try:
        metadata_keys = json.loads(args.metadata_keys)
        metadata_notes = json.loads(args.metadata_notes)
        metadata_custom = json.loads(args.metadata_custom)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in metadata arguments: {e}", file=sys.stderr)
        return 1

    # Get the appropriate benchmark adapter
    try:
        adapter = BenchmarkRegistry.get_adapter(adapter_name)
    except KeyError:
        available = ", ".join(BenchmarkRegistry.list_adapters())
        print(f"ERROR: No adapter registered for '{adapter_name}'", file=sys.stderr)
        print(f"Available adapters: {available}", file=sys.stderr)
        return 1

    print(f"\n=== Building {benchmark_name} Benchmark JSON ===")
    print(f"Using adapter: {adapter_name}\n")

    # Determine paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    dataset_directory = repo_root / "benchmark_datasets" / benchmark_name

    # Validate input directory
    if not dataset_directory.exists() or not dataset_directory.is_dir():
        print(f"ERROR: Dataset directory not found: {dataset_directory}", file=sys.stderr)
        return 1

    # Create output directory
    output_dir = repo_root / "results" / "json" / benchmark_name

    # Prepare metadata
    metadata = {
        "benchmark_id": benchmark_name,
        "keys": metadata_keys,
        "notes": metadata_notes,
        "custom": metadata_custom
    }

    # Use adapter to build benchmark JSON
    try:
        json_path = adapter.build_benchmark_json(dataset_directory, output_dir, metadata)
        return 0
    except Exception as e:
        print(f"\nERROR: Failed to build benchmark JSON: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
