"""
Database Manager for poRTLe UI

Handles all SQLite database queries and operations.
Reuses logic from display_table_entry.py and other display scripts.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

# Import filter engine for advanced filtering
try:
    from ui.utils.filter_engine import FilterSpec
except ImportError:
    # Handle case where filter_engine might not be available
    FilterSpec = None


# Table definitions with their ID column names
TABLES = {
    "agent": ("agents", "agent_id"),
    "benchmark": ("benchmarks", "benchmark_id"),
    "dataset": ("datasets", "dataset_id"),
    "task": ("tasks", "task_id"),
    "run": ("runs", "run_id"),
    "datapoint": ("datapoints", "datapoint_id")
}

# Define table relationships for building JOINs
# Format: (from_table, to_table): (join_condition)
TABLE_JOINS = {
    # Datapoint joins
    ("datapoint", "task"): "datapoints.task_id = tasks.task_id",
    ("datapoint", "run"): "datapoints.run_id = runs.run_id",
    ("datapoint", "agent"): "datapoints.agent_id = agents.agent_id",
    ("datapoint", "dataset"): "datapoints.dataset_id = datasets.dataset_id",
    ("datapoint", "benchmark"): "datapoints.benchmark_id = benchmarks.benchmark_id",

    # Task joins
    ("task", "dataset"): "tasks.dataset_id = datasets.dataset_id",
    ("task", "benchmark"): "tasks.benchmark_id = benchmarks.benchmark_id",
    ("task", "datapoint"): "tasks.task_id = datapoints.task_id",

    # Run joins
    ("run", "agent"): "runs.agent_id = agents.agent_id",
    ("run", "dataset"): "runs.dataset_id = datasets.dataset_id",
    ("run", "benchmark"): "runs.benchmark_id = benchmarks.benchmark_id",
    ("run", "datapoint"): "runs.run_id = datapoints.run_id",

    # Dataset joins
    ("dataset", "benchmark"): "datasets.benchmark_id = benchmarks.benchmark_id",
    ("dataset", "task"): "datasets.dataset_id = tasks.dataset_id",
    ("dataset", "datapoint"): "datasets.dataset_id = datapoints.dataset_id",
    ("dataset", "run"): "datasets.dataset_id = runs.dataset_id",

    # Benchmark joins
    ("benchmark", "dataset"): "benchmarks.benchmark_id = datasets.benchmark_id",
    ("benchmark", "task"): "benchmarks.benchmark_id = tasks.benchmark_id",
    ("benchmark", "run"): "benchmarks.benchmark_id = runs.benchmark_id",
    ("benchmark", "datapoint"): "benchmarks.benchmark_id = datapoints.benchmark_id",

    # Agent joins
    ("agent", "run"): "agents.agent_id = runs.agent_id",
    ("agent", "datapoint"): "agents.agent_id = datapoints.agent_id",
}


def connect_db(db_path: Path) -> sqlite3.Connection:
    """
    Connect to the poRTLe database.

    Args:
        db_path: Path to portle.db

    Returns:
        Database connection

    Raises:
        FileNotFoundError: If database doesn't exist
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Run build_datatable.py first or check the database path."
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def query_entry(
    conn: sqlite3.Connection,
    table_type: str,
    item_id: str
) -> Optional[sqlite3.Row]:
    """
    Query database for a specific entry.

    Args:
        conn: Database connection
        table_type: Type of table (agent, benchmark, etc.)
        item_id: Item identifier

    Returns:
        Row object or None if not found

    Raises:
        ValueError: If table_type is invalid
    """
    if table_type not in TABLES:
        raise ValueError(
            f"Invalid table type: {table_type}. "
            f"Must be one of: {list(TABLES.keys())}"
        )

    table_name, id_column = TABLES[table_type]

    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name} WHERE {id_column} = ?", (item_id,))
    return cursor.fetchone()


def search_entries(
    conn: sqlite3.Connection,
    table_type: str,
    filters: Optional[Dict[str, Any]] = None
) -> List[sqlite3.Row]:
    """
    Search for entries matching filters.

    Args:
        conn: Database connection
        table_type: Type of table
        filters: Dictionary of column: value pairs to filter by

    Returns:
        List of matching rows
    """
    if table_type not in TABLES:
        raise ValueError(f"Invalid table type: {table_type}")

    table_name, id_column = TABLES[table_type]

    query = f"SELECT * FROM {table_name}"
    params = []

    if filters:
        conditions = []
        for column, value in filters.items():
            if value is not None:
                conditions.append(f"{column} = ?")
                params.append(value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

    cursor = conn.cursor()
    cursor.execute(query, params)
    return cursor.fetchall()


def build_query_with_joins(
    base_table_type: str,
    referenced_tables: List[str]
) -> str:
    """
    Build a SQL query with necessary JOINs for cross-table filtering.

    Args:
        base_table_type: The main table being queried (e.g., "task")
        referenced_tables: List of other table types referenced in filters (e.g., ["datapoint"])

    Returns:
        SQL query string with JOINs (without WHERE clause)

    Example:
        >>> build_query_with_joins("task", ["datapoint"])
        "SELECT DISTINCT tasks.* FROM tasks LEFT JOIN datapoints ON tasks.task_id = datapoints.task_id"
    """
    if not referenced_tables:
        # No cross-table filtering needed
        table_name, _ = TABLES[base_table_type]
        return f"SELECT * FROM {table_name}"

    base_table_name, _ = TABLES[base_table_type]

    # Start with base query - use DISTINCT to avoid duplicates from JOINs
    query = f"SELECT DISTINCT {base_table_name}.* FROM {base_table_name}"

    # Add LEFT JOINs for each referenced table
    joined_tables = {base_table_type}

    for ref_table in referenced_tables:
        if ref_table in joined_tables:
            continue

        # Get table name
        ref_table_name, _ = TABLES[ref_table]

        # Find join condition
        join_key = (base_table_type, ref_table)
        if join_key in TABLE_JOINS:
            join_condition = TABLE_JOINS[join_key]
            query += f" LEFT JOIN {ref_table_name} ON {join_condition}"
            joined_tables.add(ref_table)
        else:
            # No direct join - might need multi-hop join in future
            # For now, just skip (will cause SQL error if field is used)
            pass

    return query


def apply_filters(
    conn: sqlite3.Connection,
    table_type: str,
    filter_spec: "FilterSpec"
) -> List[sqlite3.Row]:
    """
    Apply advanced filters using the FilterSpec engine.

    This function enables querying nested metadata fields and custom attributes
    using SQLite JSON functions.

    Args:
        conn: Database connection
        table_type: Type of table (agent, benchmark, dataset, task, run, datapoint)
        filter_spec: FilterSpec object defining the filters to apply

    Returns:
        List of matching rows

    Raises:
        ValueError: If table_type is invalid
        ImportError: If filter_engine is not available

    Example:
        >>> from ui.utils.filter_engine import FilterSpec
        >>> filter_spec = FilterSpec(
        ...     field_filters=[{"field": "score", "op": ">", "value": 0.8}],
        ...     metadata_filters=[{"path": "keys", "op": "contains", "value": "production"}]
        ... )
        >>> results = apply_filters(conn, "datapoint", filter_spec)
    """
    if FilterSpec is None:
        raise ImportError(
            "filter_engine module not available. "
            "Cannot use apply_filters without FilterSpec."
        )

    if table_type not in TABLES:
        raise ValueError(
            f"Invalid table type: {table_type}. "
            f"Must be one of: {list(TABLES.keys())}"
        )

    # Parse field filters to identify cross-table references
    referenced_tables = []
    for field_filter in filter_spec.field_filters:
        field = field_filter.field
        # Check if field has table prefix (e.g., "datapoint.score")
        if "." in field:
            table_prefix = field.split(".", 1)[0]
            # Validate it's a known table
            if table_prefix in TABLES and table_prefix != table_type:
                if table_prefix not in referenced_tables:
                    referenced_tables.append(table_prefix)

    # Build base query with necessary JOINs
    base_query = build_query_with_joins(table_type, referenced_tables)

    # Generate SQL query with filters
    query, params = filter_spec.to_sql(base_query)

    cursor = conn.cursor()
    cursor.execute(query, params)
    return cursor.fetchall()


def get_all_entries(
    conn: sqlite3.Connection,
    table_type: str
) -> pd.DataFrame:
    """
    Get all entries from a table as a pandas DataFrame.

    Args:
        conn: Database connection
        table_type: Type of table

    Returns:
        DataFrame with all entries
    """
    if table_type not in TABLES:
        raise ValueError(f"Invalid table type: {table_type}")

    table_name, _ = TABLES[table_type]

    return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def get_related_entries(
    row: sqlite3.Row,
    table_type: str
) -> List[Tuple[str, str]]:
    """
    Get related entries based on foreign keys.

    Args:
        row: Database row
        table_type: Type of table

    Returns:
        List of (relation_type, relation_id) tuples
    """
    relations = []

    # Define relationships
    if table_type == "dataset":
        if row["benchmark_id"]:
            relations.append(("benchmark", row["benchmark_id"]))

    elif table_type == "task":
        if row["benchmark_id"]:
            relations.append(("benchmark", row["benchmark_id"]))
        if row["dataset_id"]:
            relations.append(("dataset", row["dataset_id"]))

    elif table_type == "run":
        if row["benchmark_id"]:
            relations.append(("benchmark", row["benchmark_id"]))
        if row["dataset_id"]:
            relations.append(("dataset", row["dataset_id"]))
        if row["agent_id"]:
            relations.append(("agent", row["agent_id"]))

    elif table_type == "datapoint":
        if row["benchmark_id"]:
            relations.append(("benchmark", row["benchmark_id"]))
        if row["dataset_id"]:
            relations.append(("dataset", row["dataset_id"]))
        if row["task_id"]:
            relations.append(("task", row["task_id"]))
        if row["agent_id"]:
            relations.append(("agent", row["agent_id"]))
        if row["run_id"]:
            relations.append(("run", row["run_id"]))

    return relations


def get_unique_values(
    conn: sqlite3.Connection,
    table_type: str,
    column: str
) -> List[Any]:
    """
    Get unique values for a column in a table.

    Args:
        conn: Database connection
        table_type: Type of table
        column: Column name

    Returns:
        List of unique values (sorted)
    """
    if table_type not in TABLES:
        raise ValueError(f"Invalid table type: {table_type}")

    table_name, _ = TABLES[table_type]

    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT {column} FROM {table_name} ORDER BY {column}")
    return [row[0] for row in cursor.fetchall() if row[0] is not None]


def get_table_count(conn: sqlite3.Connection, table_type: str) -> int:
    """
    Get count of entries in a table.

    Args:
        conn: Database connection
        table_type: Type of table

    Returns:
        Number of entries
    """
    if table_type not in TABLES:
        raise ValueError(f"Invalid table type: {table_type}")

    table_name, _ = TABLES[table_type]

    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()[0]


def get_table_columns(
    conn: sqlite3.Connection,
    table_type: str,
    exclude_json: bool = True
) -> List[Dict[str, Any]]:
    """
    Get column information for a table using SQLite schema introspection.

    Args:
        conn: Database connection
        table_type: Type of table
        exclude_json: If True, exclude JSON columns (metadata, agent_config)
                     since they have dedicated metadata filter section

    Returns:
        List of column info dicts with keys: name, type, is_primary_key

    Example:
        >>> columns = get_table_columns(conn, "datapoint")
        >>> # Returns: [
        >>> #   {"name": "datapoint_id", "type": "TEXT", "is_primary_key": True},
        >>> #   {"name": "score", "type": "REAL", "is_primary_key": False},
        >>> #   ...
        >>> # ]
    """
    if table_type not in TABLES:
        raise ValueError(f"Invalid table type: {table_type}")

    table_name, _ = TABLES[table_type]

    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")

    columns = []
    json_columns = {"metadata", "agent_config"}  # Columns to exclude from simple filters

    for col_info in cursor.fetchall():
        # col_info format: (cid, name, type, notnull, dflt_value, pk)
        col_name = col_info[1]
        col_type = col_info[2]
        is_pk = bool(col_info[5])

        # Skip JSON columns if requested
        if exclude_json and col_name in json_columns:
            continue

        columns.append({
            "name": col_name,
            "type": col_type,
            "is_primary_key": is_pk
        })

    return columns


def get_benchmarks(conn: sqlite3.Connection) -> List[str]:
    """Get list of all benchmark IDs."""
    return get_unique_values(conn, "benchmark", "benchmark_id")


def get_datasets(
    conn: sqlite3.Connection,
    benchmark_id: Optional[str] = None
) -> List[str]:
    """
    Get list of dataset IDs, optionally filtered by benchmark.

    Args:
        conn: Database connection
        benchmark_id: Optional benchmark ID to filter by

    Returns:
        List of dataset IDs
    """
    if benchmark_id:
        rows = search_entries(conn, "dataset", {"benchmark_id": benchmark_id})
        return sorted([row["dataset_id"] for row in rows])
    else:
        return get_unique_values(conn, "dataset", "dataset_id")


def get_agents(conn: sqlite3.Connection) -> List[str]:
    """Get list of all agent IDs."""
    return get_unique_values(conn, "agent", "agent_id")


def get_tasks(
    conn: sqlite3.Connection,
    benchmark_id: Optional[str] = None,
    dataset_id: Optional[str] = None
) -> List[str]:
    """
    Get list of task IDs, optionally filtered by benchmark and/or dataset.

    Args:
        conn: Database connection
        benchmark_id: Optional benchmark ID to filter by
        dataset_id: Optional dataset ID to filter by

    Returns:
        List of task IDs
    """
    filters = {}
    if benchmark_id:
        filters["benchmark_id"] = benchmark_id
    if dataset_id:
        filters["dataset_id"] = dataset_id

    if filters:
        rows = search_entries(conn, "task", filters)
        return sorted([row["task_id"] for row in rows])
    else:
        return get_unique_values(conn, "task", "task_id")


def get_runs(
    conn: sqlite3.Connection,
    benchmark_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    agent_id: Optional[str] = None
) -> List[str]:
    """
    Get list of run IDs, optionally filtered.

    Args:
        conn: Database connection
        benchmark_id: Optional benchmark ID to filter by
        dataset_id: Optional dataset ID to filter by
        agent_id: Optional agent ID to filter by

    Returns:
        List of run IDs
    """
    filters = {}
    if benchmark_id:
        filters["benchmark_id"] = benchmark_id
    if dataset_id:
        filters["dataset_id"] = dataset_id
    if agent_id:
        filters["agent_id"] = agent_id

    if filters:
        rows = search_entries(conn, "run", filters)
        return sorted([row["run_id"] for row in rows], reverse=True)
    else:
        cursor = conn.cursor()
        cursor.execute("SELECT run_id FROM runs ORDER BY run_id DESC")
        return [row[0] for row in cursor.fetchall()]


def get_database_stats(conn: sqlite3.Connection) -> Dict[str, int]:
    """
    Get statistics about database contents.

    Args:
        conn: Database connection

    Returns:
        Dictionary mapping table names to entry counts
    """
    stats = {}
    for table_type in TABLES.keys():
        stats[table_type] = get_table_count(conn, table_type)
    return stats


def get_existing_keys(
    conn: sqlite3.Connection,
    table_type: str
) -> List[str]:
    """
    Get all unique keys used in metadata for entries of a specific table type.

    Args:
        conn: Database connection
        table_type: Type of table

    Returns:
        Sorted list of unique keys found in metadata

    Example:
        >>> keys = get_existing_keys(conn, "dataset")
        >>> # Returns: ["high-priority", "verified", "experimental", ...]
    """
    if table_type not in TABLES:
        raise ValueError(f"Invalid table type: {table_type}")

    table_name, _ = TABLES[table_type]

    cursor = conn.cursor()
    cursor.execute(f"SELECT metadata FROM {table_name}")

    all_keys = set()

    for row in cursor.fetchall():
        metadata_str = row[0]
        if metadata_str:
            try:
                import json
                metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                keys = metadata.get("keys", [])
                if keys:
                    all_keys.update(keys)
            except (json.JSONDecodeError, TypeError):
                continue

    return sorted(list(all_keys))


def get_recent_runs_with_details(
    conn: sqlite3.Connection,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Get recent runs with full details for display.

    Args:
        conn: Database connection
        limit: Maximum number of runs to return

    Returns:
        List of run dictionaries with all fields
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            run_id, benchmark_id, dataset_id, agent_id,
            hardware_info, n, threads, run_start, run_end,
            total_time, run_directory, is_partial_run, metadata
        FROM runs
        ORDER BY run_start DESC
        LIMIT ?
    """, (limit,))

    runs = []
    for row in cursor.fetchall():
        runs.append({
            "run_id": row[0],
            "benchmark_id": row[1],
            "dataset_id": row[2],
            "agent_id": row[3],
            "hardware_info": row[4],
            "n": row[5],
            "threads": row[6],
            "run_start": row[7],
            "run_end": row[8],
            "total_time": row[9],
            "run_directory": row[10],
            "is_partial_run": row[11],
            "metadata": row[12]
        })

    return runs


def get_run_datapoints(
    conn: sqlite3.Connection,
    run_id: str
) -> List[Dict[str, Any]]:
    """
    Get all datapoints for a specific run.

    Args:
        conn: Database connection
        run_id: Run ID

    Returns:
        List of datapoint dictionaries with sample_index extracted from metadata
    """
    import json
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            datapoint_id, benchmark_id, dataset_id, task_id,
            run_id, agent_id, score, tokens, time, error, metadata
        FROM datapoints
        WHERE run_id = ?
        ORDER BY task_id, datapoint_id
    """, (run_id,))

    datapoints = []
    for row in cursor.fetchall():
        # Extract sample_index from metadata
        metadata_str = row[10]
        sample_index = 0
        if metadata_str:
            try:
                metadata = json.loads(metadata_str)
                sample_index = metadata.get("custom", {}).get("sample_index", 0)
            except (json.JSONDecodeError, TypeError):
                pass
        
        datapoints.append({
            "datapoint_id": row[0],
            "benchmark_id": row[1],
            "dataset_id": row[2],
            "task_id": row[3],
            "run_id": row[4],
            "agent_id": row[5],
            "n": sample_index + 1,  # Convert 0-indexed to 1-indexed for display
            "score": row[6],
            "tokens": row[7],
            "time": row[8],
            "error": row[9],
            "metadata": metadata_str
        })

    return datapoints

