"""
Benchmark adapter system for poRTLe.

This package provides a pluggable architecture for supporting multiple benchmark frameworks.
Each benchmark (e.g., CVDP, TuRTLe) can have its own adapter that handles benchmark-specific
logic while keeping the core poRTLe system generic.
"""

from .base import BenchmarkAdapter
from .registry import BenchmarkRegistry
from .cvdp_adapter import CVDPAdapter

# Auto-register available adapters
BenchmarkRegistry.register("cvdp", CVDPAdapter)
BenchmarkRegistry.register("cvdp_example", CVDPAdapter)

# Set CVDPAdapter as the default fallback for unregistered benchmarks
BenchmarkRegistry.set_default_adapter(CVDPAdapter)


__all__ = ["BenchmarkAdapter", "BenchmarkRegistry", "CVDPAdapter"]
