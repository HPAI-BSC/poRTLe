"""
Registry system for benchmark adapters.

This module provides a centralized registry for managing and accessing
benchmark adapters. New benchmark frameworks can be registered and
retrieved dynamically.
"""

import warnings
from typing import Type, Dict, Optional
from .base import BenchmarkAdapter


class BenchmarkRegistry:
    """
    Central registry for benchmark adapters.

    This class maintains a mapping of benchmark names to their adapter classes,
    allowing dynamic lookup and instantiation of adapters.
    """

    _adapters: Dict[str, Type[BenchmarkAdapter]] = {}
    _default_adapter: Optional[Type[BenchmarkAdapter]] = None

    @classmethod
    def register(cls, name: str, adapter_class: Type[BenchmarkAdapter]) -> None:
        """
        Register a benchmark adapter.

        Args:
            name: Unique identifier for the benchmark (e.g., "cvdp", "turtle")
            adapter_class: Class that implements BenchmarkAdapter interface

        Raises:
            ValueError: If adapter_class doesn't inherit from BenchmarkAdapter
        """
        if not issubclass(adapter_class, BenchmarkAdapter):
            raise ValueError(
                f"Adapter class {adapter_class.__name__} must inherit from BenchmarkAdapter"
            )

        cls._adapters[name.lower()] = adapter_class

    @classmethod
    def set_default_adapter(cls, adapter_class: Type[BenchmarkAdapter]) -> None:
        """
        Set the default adapter to use when a benchmark is not explicitly registered.

        Args:
            adapter_class: Class that implements BenchmarkAdapter interface

        Raises:
            ValueError: If adapter_class doesn't inherit from BenchmarkAdapter
        """
        if not issubclass(adapter_class, BenchmarkAdapter):
            raise ValueError(
                f"Adapter class {adapter_class.__name__} must inherit from BenchmarkAdapter"
            )
        cls._default_adapter = adapter_class

    @classmethod
    def get_adapter(cls, name: str) -> BenchmarkAdapter:
        """
        Get an instance of a registered benchmark adapter.

        If the benchmark is not explicitly registered and a default adapter has been set,
        returns an instance of the default adapter with a warning.

        Args:
            name: Benchmark identifier

        Returns:
            Instance of the requested benchmark adapter (or default if not registered)

        Raises:
            KeyError: If no adapter is registered with the given name and no default is set
        """
        name_lower = name.lower()
        if name_lower not in cls._adapters:
            if cls._default_adapter is not None:
                warnings.warn(
                    f"No adapter explicitly registered for benchmark '{name}'. "
                    f"Using default adapter: {cls._default_adapter.__name__}",
                    UserWarning
                )
                return cls._default_adapter()
            else:
                available = ", ".join(cls._adapters.keys())
                raise KeyError(
                    f"No adapter registered for benchmark '{name}'. "
                    f"Available adapters: {available}"
                )

        return cls._adapters[name_lower]()

    @classmethod
    def list_adapters(cls) -> list[str]:
        """
        Get a list of all registered adapter names.

        Returns:
            List of registered benchmark names
        """
        return list(cls._adapters.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if an adapter is registered.

        Args:
            name: Benchmark identifier

        Returns:
            True if adapter is registered, False otherwise
        """
        return name.lower() in cls._adapters
