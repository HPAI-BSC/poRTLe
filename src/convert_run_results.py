#!/usr/bin/env python3
"""Convert an existing benchmark run directory into a run JSON entry."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from run_dataset import (
    create_datapoint,
    create_metadata,
    generate_run_id,
    get_custom_field,
    load_raw_results,
    parse_raw_results,
    parse_token_count,
    save_run_json,
    validate_configuration,
)
from benchmarks import BenchmarkRegistry


def parse_iso_datetime(value: str, field_name: str) -> datetime:
    """Ensure the provided string is a valid ISO-8601 datetime."""
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO datetime for {field_name}: {value}") from exc


def load_metadata(metadata_path: Optional[str]) -> Dict[str, Any]:
    """Load metadata from a JSON file if provided, otherwise return default metadata."""
    if not metadata_path:
        return create_metadata()

    path = Path(metadata_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Metadata file must contain a JSON object: {path}")

    # Ensure required keys exist even if missing from file
    data.setdefault("keys", [])
    data.setdefault("notes", [])
    data.setdefault("custom", {})
    return data


def _sample_sort_key(path: Path) -> tuple:
    match = re.match(r"sample_(\d+)", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def discover_sample_contexts(run_dir: Path) -> List[Dict[str, Any]]:
    """Identify sample directories containing raw_result.json files."""
    candidate_dirs: List[Path] = []

    root_raw = run_dir / "raw_result.json"
    if root_raw.exists():
        candidate_dirs.append(run_dir)

    sample_dirs = [p for p in run_dir.iterdir() if p.is_dir() and (p / "raw_result.json").exists()]
    if sample_dirs:
        sample_dirs.sort(key=_sample_sort_key)
        candidate_dirs.extend([p for p in sample_dirs if p not in candidate_dirs])

    if not candidate_dirs:
        raise ValueError(
            "No raw_result.json found in the provided run directory or any sample_* subdirectories."
        )

    contexts: List[Dict[str, Any]] = []
    for candidate in candidate_dirs:
        raw = load_raw_results(candidate)
        if raw:
            contexts.append({
                "path": candidate,
                "label": candidate.name,
                "raw_results": raw,
            })
        else:
            print(f"Warning: raw_result.json in {candidate} is empty or invalid; skipping.")

    if not contexts:
        raise ValueError("Could not load raw_result.json data from any detected sample directories.")

    return contexts


def collect_datapoints(
    tasks: List[Dict[str, Any]],
    benchmark_id: str,
    dataset_id: str,
    agent_id: str,
    run_id: str,
    sample_contexts: List[Dict[str, Any]],
    source_dataset_id: str,
    adapter,
) -> List[Dict[str, Any]]:
    """Generate datapoint entries from raw_result.json files across samples."""
    datapoints: List[Dict[str, Any]] = []

    for task in tasks:
        task_id = task["task_id"]
        source_task_id = get_custom_field(task, "source_task_id", task_id)
        per_task_counter = 0

        for sample_index, sample in enumerate(sample_contexts):
            raw_results = sample["raw_results"]
            task_results = raw_results.get(source_task_id) if raw_results else None
            tests = task_results.get("tests") if isinstance(task_results, dict) else None
            test_entries = tests or []

            if not test_entries:
                print(
                    f"Warning: No test entries for task '{source_task_id}' in sample '{sample['label']}'."
                )
                continue

            # Parse results aggregated across all tests (adapter handles aggregation)
            metrics = parse_raw_results(
                sample["path"],
                source_task_id,
                adapter,
                attempt_index=0,
                raw_results=raw_results,
            )
            tokens = parse_token_count(
                sample["path"],
                source_task_id,
                adapter,
                attempt_index=0,
                log_path=metrics.get("log_path"),
            )
            datapoint_id = f"{run_id}-{task_id}-{per_task_counter:03d}"
            per_task_counter += 1

            datapoint = create_datapoint(
                datapoint_id=datapoint_id,
                benchmark_id=benchmark_id,
                dataset_id=dataset_id,
                task_id=task_id,
                agent_id=agent_id,
                run_id=run_id,
                score=metrics["score"],
                execution_time=metrics["time"],
                tokens=tokens,
                error_code=metrics["error"],
            )

            metadata = datapoint.get("metadata") or create_metadata()
            custom_meta = metadata.get("custom") or {}
            metadata["custom"] = custom_meta
            custom_meta["source_task_id"] = source_task_id
            custom_meta["source_dataset_id"] = source_dataset_id
            custom_meta["sample_name"] = sample["label"]
            custom_meta["sample_index"] = sample_index
            # If the adapter surfaced per-metric scores (e.g., notsotiny syntax/equivalence/functionality), preserve them
            if metrics.get("component_scores"):
                custom_meta["component_scores"] = metrics["component_scores"]
                # Also expose flattened, numeric scores for easier querying/aggregation
                custom_meta["syntax_score"] = metrics["component_scores"].get("syntax")
                custom_meta["equivalence_score"] = metrics["component_scores"].get("equivalence")
                custom_meta["functionality_score"] = metrics["component_scores"].get("functionality")
            datapoint["metadata"] = metadata

            datapoints.append(datapoint)

        if per_task_counter == 0:
            print(f"Warning: Task '{source_task_id}' produced no datapoints across samples.")

    return datapoints


def determine_output_path(
    output_arg: Optional[str],
    repo_root: Path,
    benchmark_id: str,
    dataset_id: str,
    run_id: str,
) -> Path:
    """Resolve the output path for the run JSON."""
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    return (
        repo_root
        / "results"
        / "json"
        / benchmark_id
        / dataset_id
        / f"{run_id}.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an existing run directory (containing raw_result.json and agent logs) "
            "into a run JSON entry."
        )
    )
    parser.add_argument("--run-dir", required=True, help="Path to the run directory to convert")
    parser.add_argument("--benchmark-id", required=True, help="Benchmark identifier")
    parser.add_argument("--dataset-id", required=True, help="Dataset identifier")
    parser.add_argument("--agent-id", required=True, help="Agent identifier")
    parser.add_argument("--hardware-info", required=True, help="Hardware descriptor")
    parser.add_argument("--run-start", required=True, help="Run start timestamp (ISO-8601)")
    parser.add_argument("--run-end", required=True, help="Run end timestamp (ISO-8601)")
    parser.add_argument("--n", type=int, required=True, help="Number of runs per task")
    parser.add_argument("--threads", type=int, default=1, help="Thread count used for the run")
    parser.add_argument("--metadata-file", help="Optional path to a metadata JSON file")
    parser.add_argument(
        "--output",
        help="Optional explicit output path for the generated run JSON",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory is not a directory: {run_dir}")

    if args.n <= 0:
        raise ValueError("--n must be a positive integer")
    if args.threads <= 0:
        raise ValueError("--threads must be a positive integer")

    run_start_dt = parse_iso_datetime(args.run_start, "run_start")
    run_end_dt = parse_iso_datetime(args.run_end, "run_end")
    if run_end_dt < run_start_dt:
        raise ValueError("run_end cannot be earlier than run_start")

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    # Validate benchmark/dataset/agent definitions using existing helpers
    config = {
        "benchmark_id": args.benchmark_id,
        "dataset_id": args.dataset_id,
        "agent_id": args.agent_id,
        "hardware_info": args.hardware_info,
        "n": args.n,
        "threads": args.threads,
    }
    _agent, dataset, tasks = validate_configuration(config, repo_root)
    source_dataset_id = get_custom_field(dataset, "source_dataset_id", dataset["dataset_id"])

    run_id = generate_run_id(args.agent_id, None, args.hardware_info)
    print(f"Generated run_id: {run_id}")

    # Get benchmark adapter (default to CVDP for backward compatibility)
    adapter_name = config.get("adapter", args.benchmark_id.lower())
    try:
        adapter = BenchmarkRegistry.get_adapter(adapter_name)
        print(f"Using benchmark adapter: {adapter_name}")
    except KeyError:
        print(f"Warning: No adapter found for '{adapter_name}', defaulting to 'cvdp'")
        adapter = BenchmarkRegistry.get_adapter("cvdp")

    metadata = load_metadata(args.metadata_file)
    sample_contexts = discover_sample_contexts(run_dir)
    detected_samples = len(sample_contexts)
    if detected_samples != args.n:
        print(
            f"Warning: --n is set to {args.n}, but detected {detected_samples} sample(s) with raw_result.json files."
        )

    datapoints = collect_datapoints(
        tasks=tasks,
        benchmark_id=args.benchmark_id,
        dataset_id=args.dataset_id,
        agent_id=args.agent_id,
        run_id=run_id,
        sample_contexts=sample_contexts,
        source_dataset_id=source_dataset_id,
        adapter=adapter,
    )

    if not datapoints:
        raise ValueError("No datapoints were generated from the provided run directory.")

    total_time_seconds = (run_end_dt - run_start_dt).total_seconds()
    run_data = {
        "run_id": run_id,
        "benchmark_id": args.benchmark_id,
        "dataset_id": args.dataset_id,
        "agent_id": args.agent_id,
        "hardware_info": args.hardware_info,
        "n": args.n,
        "threads": args.threads,
        "run_start": run_start_dt.isoformat(),
        "run_end": run_end_dt.isoformat(),
        "total_time": f"{total_time_seconds:.2f}s",
        "run_directory": str(run_dir),
        "metadata": metadata,
        "datapoints": datapoints,
    }

    output_path = determine_output_path(args.output, repo_root, args.benchmark_id, args.dataset_id, run_id)
    save_run_json(run_data, output_path)

    passed_datapoints = sum(1 for dp in datapoints if dp.get("score") == 1.0)
    print(
        f"Saved run JSON to {output_path} (run_id={run_id}, passed {passed_datapoints}/{len(datapoints)} datapoints)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
