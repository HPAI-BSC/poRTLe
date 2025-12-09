"""
VerilogEval CVDP evaluation harness.

Mirrors the turtle VerilogEval evaluation:
- Finds Verilog/SystemVerilog sources in ../rtl (supports multiple files)
- Compiles RTL + testbench + golden reference with Verilator (SV2012)
- Runs simulation and marks pass/fail using mismatch/runtime checks
"""

import re
import os
import subprocess
import sys
from pathlib import Path


def find_rtl_sources(rtl_dir: Path) -> list[Path]:
    """Locate Verilog/SystemVerilog sources in rtl_dir."""
    candidates = sorted(p for p in rtl_dir.iterdir() if p.suffix in (".v", ".sv"))
    if not candidates:
        raise FileNotFoundError(f"No Verilog files found in {rtl_dir}")
    return candidates


def run_command(cmd, timeout: int = 60, env=None) -> subprocess.CompletedProcess:
    """Run a command with timeout, capturing stdout/stderr."""
    return subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def evaluate():
    script_dir = Path(__file__).parent.resolve()
    print(f"[verilogeval_harness] script_dir={script_dir}")
    rtl_dir = script_dir.parent / "rtl"
    print(f"[verilogeval_harness] looking for RTL in {rtl_dir}")

    rtl_sources = find_rtl_sources(rtl_dir)
    print(f"[verilogeval_harness] found RTL sources:")
    for src in rtl_sources:
        print(f"  - {src}")

    # Testbench: prefer *_test.sv naming
    test_candidates = sorted(script_dir.glob("*_test.sv"))
    if not test_candidates:
        test_candidates = sorted(script_dir.glob("*test*.v"))
    if not test_candidates:
        raise FileNotFoundError(f"Missing testbench matching '*_test.sv' in {script_dir}")
    testbench = test_candidates[0]
    print(f"[verilogeval_harness] using testbench: {testbench}")

    # Golden reference file (required for VerilogEval): prefer *_ref.sv
    ref_candidates = sorted(script_dir.glob("*_ref.sv"))
    if not ref_candidates:
        ref_candidates = sorted(script_dir.glob("verified_*.v"))
    if not ref_candidates:
        raise FileNotFoundError(f"No *_ref.sv or verified_*.v golden reference found in {script_dir}")
    ref = ref_candidates[0]
    print(f"[verilogeval_harness] using golden reference: {ref}")

    mdir = Path.cwd() / "obj_dir"
    exe_path = mdir / "sim.out"
    print(f"[verilogeval_harness] build output: {exe_path} (Mdir={mdir})")

    compile_cmd = [
        "verilator",
        "--build",
        "--binary",
        "--compiler",
        "gcc",
        '-CFLAGS',
        "-std=c++20",
        "-Wno-fatal",
        "-Wno-DECLFILENAME",
        "-Wno-TIMESCALEMOD",
        "-Wno-WIDTHTRUNC",
        "-Wno-INITIALDLY",
        "-Wno-BLKANDNBLK",
        "--top-module",
        "tb",
        "--Mdir",
        str(mdir),
        "-o",
        str(exe_path),
        *[str(p) for p in rtl_sources],
        str(testbench),
        str(ref),
    ]
    print(f"[verilogeval_harness] compile command: {' '.join(compile_cmd)}")
    # Force gcc/g++ in case the image defaults to clang++
    env = os.environ.copy()
    env.update({"CC": "gcc", "CXX": "g++"})

    compile_result = run_command(compile_cmd, timeout=120, env=env)
    if compile_result.returncode != 0:
        print("Compile failed")
        print("stdout:\n", compile_result.stdout)
        print("stderr:\n", compile_result.stderr)
        return 1
    else:
        print(f"[verilogeval_harness] compile exit code: {compile_result.returncode}")

    sim_cmd = [str(exe_path)]
    print(f"[verilogeval_harness] sim command: {' '.join(sim_cmd)}")
    sim_result = run_command(sim_cmd, timeout=120)
    stdout = sim_result.stdout or ""
    stderr = sim_result.stderr or ""
    print(f"[verilogeval_harness] sim exit code: {sim_result.returncode}, timeout={False}")

    passed = False
    reasons = []

    mismatch = re.search(r"Mismatches:\s+(\d+)\s+in\s+(\d+)\s+samples", stdout)

    if "syntax error" in stderr.lower():
        reasons.append("syntax error in simulation stderr")
    elif stderr.strip():
        reasons.append("runtime error during simulation (stderr not empty)")
    elif mismatch:
        mismatches = int(mismatch.group(1))
        total = int(mismatch.group(2))
        if mismatches == 0:
            passed = True
        else:
            reasons.append(f"{mismatches} mismatches out of {total}")
    else:
        reasons.append("test did not report pass (no mismatch line with zero errors)")

    print("Simulation stdout:\n", stdout)
    if stderr:
        print("Simulation stderr:\n", stderr)

    if passed:
        print("Result: passed")
        return 0
    else:
        if not reasons:
            reasons.append("unknown failure")
        print(f"[verilogeval_harness] pass checks failed because: {', '.join(reasons)}")
        print("Result: failed")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(evaluate())
    except subprocess.TimeoutExpired as e:
        print(f"Timed out while running: {e.cmd}")
        sys.exit(1)
