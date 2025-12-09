#!/usr/bin/env python3
"""
Description: File that takes all the available json files, then turns it into one datatable that is queryable

List of files it uses:
- agents.json
- all of the <name>_benchmark.json files
- all of the <run_id>.json files

The datatable that is created should be made in a way that will make getting the data,
and displaying results in different plots easy. For example the heatmap, or display entry
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Any


def create_schema(conn: sqlite3.Connection):
    """
    Create database schema with 6 tables.

    Args:
        conn: Database connection
    """
    cursor = conn.cursor()

    # Create agents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY,
            about TEXT,
            agent_config TEXT,
            metadata TEXT
        )
    """)

    # Create benchmarks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmarks (
            benchmark_id TEXT PRIMARY KEY,
            metadata TEXT
        )
    """)

    # Create datasets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id TEXT PRIMARY KEY,
            benchmark_id TEXT,
            commercial TEXT,
            task_count INTEGER,
            metadata TEXT,
            FOREIGN KEY (benchmark_id) REFERENCES benchmarks(benchmark_id)
        )
    """)

    # Create tasks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            benchmark_id TEXT,
            dataset_id TEXT,
            metadata TEXT,
            FOREIGN KEY (benchmark_id) REFERENCES benchmarks(benchmark_id),
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
        )
    """)

    # Create runs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            benchmark_id TEXT,
            dataset_id TEXT,
            agent_id TEXT,
            hardware_info TEXT,
            n INTEGER,
            threads INTEGER,
            is_partial_run INTEGER,
            run_start TEXT,
            run_end TEXT,
            total_time TEXT,
            run_directory TEXT,
            metadata TEXT,
            FOREIGN KEY (benchmark_id) REFERENCES benchmarks(benchmark_id),
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
            FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
        )
    """)

    # Create datapoints table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datapoints (
            datapoint_id TEXT PRIMARY KEY,
            benchmark_id TEXT,
            dataset_id TEXT,
            task_id TEXT,
            agent_id TEXT,
            run_id TEXT,
            tokens INTEGER,
            time REAL,
            error INTEGER,
            score REAL,
            metadata TEXT,
            FOREIGN KEY (benchmark_id) REFERENCES benchmarks(benchmark_id),
            FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
            FOREIGN KEY (task_id) REFERENCES tasks(task_id),
            FOREIGN KEY (agent_id) REFERENCES agents(agent_id),
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        )
    """)

    # Create indexes for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_benchmark ON datasets(benchmark_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_benchmark ON tasks(benchmark_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tasks_dataset ON tasks(dataset_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_benchmark ON runs(benchmark_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_datapoints_task ON datapoints(task_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_datapoints_run ON datapoints(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_datapoints_agent ON datapoints(agent_id)")

    conn.commit()


def insert_agents(conn: sqlite3.Connection, agents_path: Path):
    """
    Insert agents from agents.json into database.

    Args:
        conn: Database connection
        agents_path: Path to agents.json
    """
    if not agents_path.exists():
        print(f"Warning: agents.json not found at {agents_path}")
        return

    with open(agents_path, 'r') as f:
        agents = json.load(f)

    cursor = conn.cursor()
    count = 0

    for agent in agents:
        cursor.execute("""
            INSERT OR REPLACE INTO agents (agent_id, about, agent_config, metadata)
            VALUES (?, ?, ?, ?)
        """, (
            agent.get("agent_id"),
            agent.get("about"),
            json.dumps(agent.get("agent_config", {})),
            json.dumps(agent.get("metadata", {}))
        ))
        count += 1

    conn.commit()
    print(f"  Inserted {count} agents")


def insert_benchmarks(conn: sqlite3.Connection, benchmark_jsons: List[Path]):
    """
    Insert benchmarks, datasets, and tasks from benchmark JSON files.

    Args:
        conn: Database connection
        benchmark_jsons: List of paths to benchmark JSON files
    """
    cursor = conn.cursor()
    benchmark_count = 0
    dataset_count = 0
    task_count = 0

    for benchmark_path in benchmark_jsons:
        with open(benchmark_path, 'r') as f:
            benchmark_data = json.load(f)

        # Insert benchmark
        cursor.execute("""
            INSERT OR REPLACE INTO benchmarks (benchmark_id, metadata)
            VALUES (?, ?)
        """, (
            benchmark_data.get("benchmark_id"),
            json.dumps(benchmark_data.get("metadata", {}))
        ))
        benchmark_count += 1

        # Insert datasets and tasks
        for dataset in benchmark_data.get("datasets", []):
            cursor.execute("""
                INSERT OR REPLACE INTO datasets (dataset_id, benchmark_id, commercial, task_count, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                dataset.get("dataset_id"),
                dataset.get("benchmark_id"),
                dataset.get("commercial"),
                dataset.get("task_count"),
                json.dumps(dataset.get("metadata", {}))
            ))
            dataset_count += 1

            # Insert tasks
            for task in dataset.get("tasks", []):
                cursor.execute("""
                    INSERT OR REPLACE INTO tasks (task_id, benchmark_id, dataset_id, metadata)
                    VALUES (?, ?, ?, ?)
                """, (
                    task.get("task_id"),
                    task.get("benchmark_id"),
                    task.get("dataset_id"),
                    json.dumps(task.get("metadata", {}))
                ))
                task_count += 1

    conn.commit()
    print(f"  Inserted {benchmark_count} benchmarks, {dataset_count} datasets, {task_count} tasks")


def insert_runs(conn: sqlite3.Connection, run_jsons: List[Path]):
    """
    Insert runs and datapoints from run JSON files.

    Args:
        conn: Database connection
        run_jsons: List of paths to run JSON files
    """
    cursor = conn.cursor()
    run_count = 0
    datapoint_count = 0

    for run_path in run_jsons:
        with open(run_path, 'r') as f:
            run_data = json.load(f)

        # Insert run
        cursor.execute("""
            INSERT OR REPLACE INTO runs (
                run_id, benchmark_id, dataset_id, agent_id, hardware_info,
                n, threads, is_partial_run, run_start, run_end, total_time, run_directory, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_data.get("run_id"),
            run_data.get("benchmark_id"),
            run_data.get("dataset_id"),
            run_data.get("agent_id"),
            run_data.get("hardware_info"),
            run_data.get("n"),
            run_data.get("threads"),
            1 if run_data.get("is_partial_run", False) else 0,  # Convert boolean to integer
            run_data.get("run_start"),
            run_data.get("run_end"),
            run_data.get("total_time"),
            run_data.get("run_directory"),
            json.dumps(run_data.get("metadata", {}))
        ))
        run_count += 1

        # Insert datapoints
        for datapoint in run_data.get("datapoints", []):
            cursor.execute("""
                INSERT OR REPLACE INTO datapoints (
                    datapoint_id, benchmark_id, dataset_id, task_id, agent_id, run_id,
                    tokens, time, error, score, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datapoint.get("datapoint_id"),
                datapoint.get("benchmark_id"),
                datapoint.get("dataset_id"),
                datapoint.get("task_id"),
                datapoint.get("agent_id"),
                datapoint.get("run_id"),
                datapoint.get("tokens"),
                datapoint.get("time"),
                datapoint.get("error"),
                datapoint.get("score"),
                json.dumps(datapoint.get("metadata", {}))
            ))
            datapoint_count += 1

    conn.commit()
    print(f"  Inserted {run_count} runs, {datapoint_count} datapoints")


def build_database(repo_root: Path, db_path: Path):
    """
    Build complete database from all JSON files.

    Args:
        repo_root: Repository root path
        db_path: Output database path
    """
    print("\n" + "=" * 60)
    print("Building poRTLe Database")
    print("=" * 60)

    # Remove old database if exists
    if db_path.exists():
        print(f"\nRemoving existing database: {db_path}")
        db_path.unlink()

    # Create database and schema
    print(f"Creating new database: {db_path}")
    conn = sqlite3.connect(str(db_path))
    create_schema(conn)
    print("  ✓ Schema created")

    # Find and load agents.json
    print("\nLoading agents...")
    agents_path = repo_root / "results" / "json" / "agents.json"
    insert_agents(conn, agents_path)

    # Find and load benchmark JSONs
    print("\nLoading benchmarks...")
    json_dir = repo_root / "results" / "json"
    benchmark_jsons = []
    for benchmark_dir in json_dir.iterdir():
        if benchmark_dir.is_dir() and benchmark_dir.name != "agents.json":
            benchmark_json = benchmark_dir / f"{benchmark_dir.name}_benchmark.json"
            if benchmark_json.exists():
                benchmark_jsons.append(benchmark_json)

    if benchmark_jsons:
        insert_benchmarks(conn, benchmark_jsons)
    else:
        print("  Warning: No benchmark JSONs found")

    # Find and load run JSONs
    print("\nLoading runs...")
    run_jsons = []
    for benchmark_dir in json_dir.iterdir():
        if benchmark_dir.is_dir():
            for dataset_dir in benchmark_dir.iterdir():
                if dataset_dir.is_dir():
                    for run_file in dataset_dir.glob("*.json"):
                        if not run_file.name.endswith("_benchmark.json"):
                            run_jsons.append(run_file)

    if run_jsons:
        insert_runs(conn, run_jsons)
    else:
        print("  Warning: No run JSONs found")

    # Close connection
    conn.close()

    print("\n" + "=" * 60)
    print("Database built successfully!")
    print("=" * 60)
    print(f"\nDatabase location: {db_path}")
    print("\nYou can now query the database using:")
    print(f"  sqlite3 {db_path}")
    print("Or use the poRTLe UI to view entries")


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    db_path = repo_root / "results" / "portle.db"

    try:
        build_database(repo_root, db_path)
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
