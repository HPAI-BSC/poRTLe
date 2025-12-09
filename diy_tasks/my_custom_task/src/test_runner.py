#!/usr/bin/env python3
"""
Test runner template for DIY tasks.
This script always passes - replace with actual test logic.
"""

import subprocess
import sys
from pathlib import Path


def run_test():
    """Compile and run the testbench."""

    # Paths - update these for your module
    rtl_file = Path("/code/rtl/my_module.sv")
    tb_file = Path("/code/verif/my_module_tb.sv")
    output_file = Path("/tmp/test.out")

    print("=" * 60)
    print("DIY Task Test Runner Template")
    print("=" * 60)

    # Check if files exist
    if not rtl_file.exists():
        print(f"ERROR: RTL file not found: {rtl_file}")
        return False
    if not tb_file.exists():
        print(f"ERROR: Testbench file not found: {tb_file}")
        return False

    # Compile with iverilog
    print("\n[1/2] Compiling Verilog files...")
    compile_cmd = [
        "iverilog",
        "-g2012",  # SystemVerilog 2012
        "-o", str(output_file),
        str(rtl_file),
        str(tb_file)
    ]

    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print("COMPILATION FAILED!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

        print("✓ Compilation successful")

    except subprocess.TimeoutExpired:
        print("ERROR: Compilation timed out")
        return False
    except Exception as e:
        print(f"ERROR during compilation: {e}")
        return False

    # Run simulation with vvp
    print("\n[2/2] Running simulation...")
    sim_cmd = ["vvp", str(output_file)]

    try:
        result = subprocess.run(
            sim_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/tmp"
        )

        print("\nSimulation Output:")
        print("-" * 60)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        print("-" * 60)

        # Check for test results
        if "FAIL" in result.stdout:
            print("\n❌ Some tests FAILED")
            return False
        elif "PASS" in result.stdout:
            print("\n✅ All tests PASSED")
            return True
        else:
            print("\n⚠️  Unable to determine test results")
            return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("ERROR: Simulation timed out")
        return False
    except Exception as e:
        print(f"ERROR during simulation: {e}")
        return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
