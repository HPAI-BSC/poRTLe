"""
Generic Filter Engine for poRTLe Database

Supports filtering any entity type with metadata and custom fields.
Translates structured filter objects into SQLite queries with JSON functions.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class FilterOperator(Enum):
    """Supported filter operators."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_OR_EQUAL = ">="
    LESS_OR_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"  # For strings and arrays
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    HAS_KEY = "has_key"  # Check if metadata.keys contains value
    HAS_AUTHOR = "has_author"  # Check if any note has author


class FieldFilter:
    """Filter for regular table columns."""

    def __init__(self, field: str, op: str, value: Any = None):
        """
        Create a filter for a table column.

        Args:
            field: Column name (e.g., "agent_id", "score", "benchmark_id")
            op: One of FilterOperator values (e.g., "==", ">", "in")
            value: Comparison value (type depends on operator)
        """
        self.field = field
        self.operator = FilterOperator(op)
        self.value = value

    def to_sql(self) -> tuple[str, List[Any]]:
        """
        Generate SQL WHERE clause and parameters.

        Returns:
            Tuple of (sql_clause, parameters_list)
        """
        params = []

        if self.operator == FilterOperator.EQUALS:
            return f"{self.field} = ?", [self.value]
        elif self.operator == FilterOperator.NOT_EQUALS:
            return f"{self.field} != ?", [self.value]
        elif self.operator == FilterOperator.GREATER_THAN:
            return f"{self.field} > ?", [self.value]
        elif self.operator == FilterOperator.LESS_THAN:
            return f"{self.field} < ?", [self.value]
        elif self.operator == FilterOperator.GREATER_OR_EQUAL:
            return f"{self.field} >= ?", [self.value]
        elif self.operator == FilterOperator.LESS_OR_EQUAL:
            return f"{self.field} <= ?", [self.value]
        elif self.operator == FilterOperator.IN:
            if not isinstance(self.value, (list, tuple)):
                raise ValueError(f"IN operator requires list/tuple value, got {type(self.value)}")
            placeholders = ",".join("?" * len(self.value))
            return f"{self.field} IN ({placeholders})", list(self.value)
        elif self.operator == FilterOperator.NOT_IN:
            if not isinstance(self.value, (list, tuple)):
                raise ValueError(f"NOT_IN operator requires list/tuple value, got {type(self.value)}")
            placeholders = ",".join("?" * len(self.value))
            return f"{self.field} NOT IN ({placeholders})", list(self.value)
        elif self.operator == FilterOperator.CONTAINS:
            return f"{self.field} LIKE ?", [f"%{self.value}%"]
        elif self.operator == FilterOperator.NOT_CONTAINS:
            return f"{self.field} NOT LIKE ?", [f"%{self.value}%"]
        elif self.operator == FilterOperator.STARTS_WITH:
            return f"{self.field} LIKE ?", [f"{self.value}%"]
        elif self.operator == FilterOperator.ENDS_WITH:
            return f"{self.field} LIKE ?", [f"%{self.value}"]
        elif self.operator == FilterOperator.IS_NULL:
            return f"{self.field} IS NULL", []
        elif self.operator == FilterOperator.IS_NOT_NULL:
            return f"{self.field} IS NOT NULL", []

        raise ValueError(f"Unsupported operator: {self.operator}")


class MetadataFilter:
    """Filter for JSON metadata fields using SQLite JSON functions."""

    def __init__(self, path: str, op: str, value: Any = None):
        """
        Create a filter for metadata JSON fields.

        Args:
            path: JSON path (e.g., "keys", "custom.status", "notes")
            op: One of FilterOperator values
            value: Comparison value

        Examples:
            MetadataFilter("keys", "contains", "production")
            MetadataFilter("custom.status", "==", "active")
            MetadataFilter("custom.rating", ">", 3)
            MetadataFilter("notes", "has_author", "Dakota")
        """
        self.path = path
        self.operator = FilterOperator(op)
        self.value = value

    def to_sql(self) -> tuple[str, List[Any]]:
        """
        Generate SQL WHERE clause using JSON functions.

        Returns:
            Tuple of (sql_clause, parameters_list)
        """
        params = []
        json_path = f"$.{self.path}"

        # Special handlers for metadata structure
        if self.path == "keys" and self.operator in (FilterOperator.CONTAINS, FilterOperator.HAS_KEY):
            # Check if keys array contains value
            return (
                "EXISTS (SELECT 1 FROM json_each(json_extract(metadata, '$.keys')) "
                "WHERE value = ?)",
                [self.value]
            )

        if self.path == "keys" and self.operator == FilterOperator.NOT_CONTAINS:
            # Check if keys array does NOT contain value
            return (
                "NOT EXISTS (SELECT 1 FROM json_each(json_extract(metadata, '$.keys')) "
                "WHERE value = ?)",
                [self.value]
            )

        if self.path.startswith("notes") and self.operator == FilterOperator.HAS_AUTHOR:
            # Check if any note has this author
            return (
                "EXISTS (SELECT 1 FROM json_each(json_extract(metadata, '$.notes')) "
                "WHERE json_extract(value, '$.author') = ?)",
                [self.value]
            )

        # Standard JSON path queries
        extract = f"json_extract(metadata, '{json_path}')"

        if self.operator == FilterOperator.EQUALS:
            return f"{extract} = ?", [self.value]
        elif self.operator == FilterOperator.NOT_EQUALS:
            return f"{extract} != ?", [self.value]
        elif self.operator == FilterOperator.GREATER_THAN:
            return f"CAST({extract} AS REAL) > ?", [self.value]
        elif self.operator == FilterOperator.LESS_THAN:
            return f"CAST({extract} AS REAL) < ?", [self.value]
        elif self.operator == FilterOperator.GREATER_OR_EQUAL:
            return f"CAST({extract} AS REAL) >= ?", [self.value]
        elif self.operator == FilterOperator.LESS_OR_EQUAL:
            return f"CAST({extract} AS REAL) <= ?", [self.value]
        elif self.operator == FilterOperator.IN:
            if not isinstance(self.value, (list, tuple)):
                raise ValueError(f"IN operator requires list/tuple value, got {type(self.value)}")
            placeholders = ",".join("?" * len(self.value))
            return f"{extract} IN ({placeholders})", list(self.value)
        elif self.operator == FilterOperator.NOT_IN:
            if not isinstance(self.value, (list, tuple)):
                raise ValueError(f"NOT_IN operator requires list/tuple value, got {type(self.value)}")
            placeholders = ",".join("?" * len(self.value))
            return f"{extract} NOT IN ({placeholders})", list(self.value)
        elif self.operator == FilterOperator.CONTAINS:
            # For string fields in custom metadata
            return f"{extract} LIKE ?", [f"%{self.value}%"]
        elif self.operator == FilterOperator.NOT_CONTAINS:
            return f"{extract} NOT LIKE ?", [f"%{self.value}%"]
        elif self.operator == FilterOperator.STARTS_WITH:
            return f"{extract} LIKE ?", [f"{self.value}%"]
        elif self.operator == FilterOperator.ENDS_WITH:
            return f"{extract} LIKE ?", [f"%{self.value}"]
        elif self.operator == FilterOperator.IS_NULL:
            return f"{extract} IS NULL", []
        elif self.operator == FilterOperator.IS_NOT_NULL:
            return f"{extract} IS NOT NULL", []

        raise ValueError(f"Unsupported operator: {self.operator}")


class FilterSpec:
    """Complete filter specification combining field and metadata filters."""

    def __init__(
        self,
        field_filters: Optional[List[Dict]] = None,
        metadata_filters: Optional[List[Dict]] = None,
        logic: str = "AND"
    ):
        """
        Create a complete filter specification.

        Args:
            field_filters: List of field filter dicts with keys: field, op, value
            metadata_filters: List of metadata filter dicts with keys: path, op, value
            logic: Combination logic ("AND" or "OR")

        Example:
            FilterSpec(
                field_filters=[
                    {"field": "score", "op": ">", "value": 0.8}
                ],
                metadata_filters=[
                    {"path": "keys", "op": "contains", "value": "production"},
                    {"path": "custom.status", "op": "==", "value": "active"}
                ],
                logic="AND"
            )
        """
        self.field_filters = [
            FieldFilter(**f) for f in (field_filters or [])
        ]
        self.metadata_filters = [
            MetadataFilter(**f) for f in (metadata_filters or [])
        ]
        self.logic = logic.upper()

        if self.logic not in ("AND", "OR"):
            raise ValueError(f"Logic must be 'AND' or 'OR', got '{logic}'")

    def to_sql(self, base_query: str) -> tuple[str, List[Any]]:
        """
        Generate complete SQL query with filters.

        Args:
            base_query: Base SELECT query (e.g., "SELECT * FROM agents")

        Returns:
            Tuple of (query_string, parameters_list)
        """
        all_filters = self.field_filters + self.metadata_filters

        if not all_filters:
            return base_query, []

        clauses = []
        params = []

        for filter_obj in all_filters:
            clause, filter_params = filter_obj.to_sql()
            clauses.append(f"({clause})")
            params.extend(filter_params)

        connector = f" {self.logic} "
        where_clause = connector.join(clauses)

        # Add WHERE clause to query
        if "WHERE" in base_query.upper():
            query = f"{base_query} AND ({where_clause})"
        else:
            query = f"{base_query} WHERE {where_clause}"

        return query, params

    def __repr__(self) -> str:
        """String representation of filter spec."""
        parts = []
        if self.field_filters:
            parts.append(f"{len(self.field_filters)} field filters")
        if self.metadata_filters:
            parts.append(f"{len(self.metadata_filters)} metadata filters")
        return f"FilterSpec({', '.join(parts)}, logic={self.logic})"


# Convenience functions for common filter patterns

def filter_by_metadata_key(key: str) -> FilterSpec:
    """
    Create filter for entries that have a specific metadata key.

    Args:
        key: Metadata key to filter by

    Returns:
        FilterSpec that matches entries with this key
    """
    return FilterSpec(
        metadata_filters=[
            {"path": "keys", "op": "contains", "value": key}
        ]
    )


def filter_by_custom_field(field_path: str, value: Any) -> FilterSpec:
    """
    Create filter for custom metadata field value.

    Args:
        field_path: Path within custom metadata (e.g., "status", "rating")
        value: Expected value

    Returns:
        FilterSpec that matches entries where custom.{field_path} == value
    """
    return FilterSpec(
        metadata_filters=[
            {"path": f"custom.{field_path}", "op": "==", "value": value}
        ]
    )


def filter_by_author(author: str) -> FilterSpec:
    """
    Create filter for entries with notes by a specific author.

    Args:
        author: Author name to filter by

    Returns:
        FilterSpec that matches entries with notes by this author
    """
    return FilterSpec(
        metadata_filters=[
            {"path": "notes", "op": "has_author", "value": author}
        ]
    )


def combine_filters(*filter_specs: FilterSpec, logic: str = "AND") -> FilterSpec:
    """
    Combine multiple FilterSpec objects into one.

    Args:
        *filter_specs: Variable number of FilterSpec objects to combine
        logic: Combination logic ("AND" or "OR")

    Returns:
        Combined FilterSpec
    """
    all_field_filters = []
    all_metadata_filters = []

    for spec in filter_specs:
        # Extract the filter dicts from each spec
        for ff in spec.field_filters:
            all_field_filters.append({
                "field": ff.field,
                "op": ff.operator.value,
                "value": ff.value
            })
        for mf in spec.metadata_filters:
            all_metadata_filters.append({
                "path": mf.path,
                "op": mf.operator.value,
                "value": mf.value
            })

    return FilterSpec(
        field_filters=all_field_filters,
        metadata_filters=all_metadata_filters,
        logic=logic
    )
