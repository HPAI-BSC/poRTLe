"""
RTLLM CVDP evaluation harness.

This script mirrors the RTLLM evaluation flow from the turtle task:
- Finds Verilog/SystemVerilog sources in ../rtl (supports multiple files)
- Compiles all RTL + local testbench with Verilator (binary)
- Runs the simulation and reports pass/fail using RTLLM's criteria
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


def find_rtl_sources(rtl_dir: Path) -> list[Path]:
    """Locate Verilog/SystemVerilog sources in rtl_dir."""
    candidates = sorted(p for p in rtl_dir.iterdir() if p.suffix in (".v", ".sv"))
    if not candidates:
        raise FileNotFoundError(f"No Verilog files found in {rtl_dir}")
    return candidates


def run_command(cmd, timeout: int = 15) -> subprocess.CompletedProcess:
    """Run a command with timeout, capturing stdout/stderr."""
    return subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def detect_top_module(vfile: Path) -> Optional[str]:
    """Grab the first module name from a Verilog file."""
    text = vfile.read_text()
    m = re.search(r"\bmodule\s+([A-Za-z_]\w*)", text)
    return m.group(1) if m else None


def evaluate():
    script_dir = Path(__file__).parent.resolve()
    print(f"[rtllm_harness] script_dir={script_dir}")
    rtl_dir = script_dir.parent / "rtl"
    print(f"[rtllm_harness] looking for RTL in {rtl_dir}")

    rtl_sources = find_rtl_sources(rtl_dir)
    print(f"[rtllm_harness] found RTL sources:")
    for src in rtl_sources:
        print(f"  - {src}")
    testbench = script_dir / "testbench.v"
    if not testbench.exists():
        raise FileNotFoundError(f"Missing testbench: {testbench}")
    else:
        print(f"[rtllm_harness] using testbench: {testbench}")

    # Golden reference (not used by RTLLM scoring, kept for parity and debugging)
    golden = next(iter(script_dir.glob("verified_*.v")), None)
    if golden:
        print(f"[rtllm_harness] found golden reference (for info): {golden}")

    # Build with Verilator (binary)
    mdir = Path.cwd() / "obj_dir"
    binary = mdir / "sim.out"
    top_module = detect_top_module(testbench) or "tb"
    print(f"[rtllm_harness] detected top module: {top_module}")
    print(f"[rtllm_harness] build output: {binary} (Mdir={mdir})")

    compile_cmd = [
        "verilator",
        "--build",
        "--binary",
        "--compiler",
        "gcc",
        "-CFLAGS",
        "-std=c++20",
        "-Wno-fatal",
        "-Wno-DECLFILENAME",
        "-Wno-TIMESCALEMOD",
        "-Wno-WIDTHTRUNC",
        "-Wno-INITIALDLY",
        "-Wno-BLKANDNBLK",
        "--top-module",
        top_module,
        "--Mdir",
        str(mdir),
        "-o",
        str(binary),
        *[str(p) for p in rtl_sources],
        str(testbench),
    ]
    print(f"[rtllm_harness] compile command: {' '.join(compile_cmd)}")
    env = os.environ.copy()
    env.update({"CC": "gcc", "CXX": "g++"})
    compile_result = run_command(compile_cmd, timeout=180)
    if compile_result.returncode != 0:
        print("Compile failed")
        print("Command:", " ".join(compile_cmd))
        print("stdout:\n", compile_result.stdout)
        print("stderr:\n", compile_result.stderr)
        return 1
    else:
        print(f"[rtllm_harness] compile exit code: {compile_result.returncode}")

    sim_cmd = [str(binary)]
    print(f"[rtllm_harness] sim command: {' '.join(sim_cmd)}")
    sim_result = run_command(sim_cmd, timeout=120)
    stdout = sim_result.stdout or ""
    stderr = sim_result.stderr or ""
    print(f"[rtllm_harness] sim exit code: {sim_result.returncode}, timeout={False}")
    passed = False
    reasons = []
    if "Passed" in stdout and "ERROR" not in stdout:
        passed = True
    else:
        if "Passed" not in stdout:
            reasons.append("missing 'Passed' in stdout")
        if "ERROR" in stdout:
            reasons.append("'ERROR' present in stdout")
        if sim_result.returncode not in (0, None):
            reasons.append(f"non-zero exit ({sim_result.returncode})")
        if sim_result.stderr:
            reasons.append("stderr not empty")
    if reasons:
        print(f"[rtllm_harness] pass checks failed because: {', '.join(reasons)}")

    print("Simulation stdout:\n", stdout)
    if stderr:
        print("Simulation stderr:\n", stderr)

    if golden:
        print(f"Golden solution (not used in scoring): {golden}")

    if passed:
        print("Result: passed")
        return 0
    else:
        print("Result: failed")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(evaluate())
    except subprocess.TimeoutExpired as e:
        print(f"Timed out while running: {e.cmd}")
        sys.exit(1)
