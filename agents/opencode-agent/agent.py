#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Codex CVDP agent implementation for the agentic workflow.
This agent reads prompt.json and makes changes to files in the mounted directories.
"""

import sys
import subprocess
import os
import json
from pathlib import Path


def count_tokens(output_lines):
    """Parse JSON output and sum input+output tokens from step_finish events."""
    total = 0
    for line in output_lines:
        try:
            data = json.loads(line)
            if data.get("type") == "step_finish":
                tokens = data.get("part", {}).get("tokens", {})
                total += tokens.get("input", 0) + tokens.get("output", 0)
        except json.JSONDecodeError:
            continue
    return total


def main():
    """Main agent function"""
    print("Starting CVDP opencode-agent...")

    # Handle using OpenAI API 
    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_USER_KEY")

    SYSTEM_PROMPT = """
## TLDR
You are an expert hardware design engineer given a task to execute with full control of the files in your current working directory (/code).

## Your Environment
# OSVB Tools - You have the following tools available through the cmdline.
   Verilator - Fast Verilog simulator and lint tool 'verilator'
   Icarus Verilog - Open-source Verilog/SystemVerilog simulator 'iverilog', 'vvp'
   Standard build tools (make, gcc, python, bash, coreutils)

# File Structure - You have the following files and directories. NOTE: Some directories may be empty.
   /code
      /docs - Documentation or specification files relevant to the task
      /rtl - Hardware design files (SystemVerilog or Verilog)
      /verif - Verification files (testbenches, test scripts, etc.)
      /rundir - Directory for build and simulation outputs
      prompt.json - JSON file containing your specific instructions

## Your Task
   1. Read your instructions in the prompt.json file.
   2. Read all the files provided to you in the docs/, code/, and verif/ directories if available.
   3. Edit or create the SystemVerilog/Verilog files as necessary to implement the instructions from the prompt.json file.
   4. Run linting to validate the syntax of your code with Verilator.
   5. Create or update comprehensive testbenches in the verif/ directory to verify your implementation.
   6. Run simulations with your testbenches to ensure your design works as intended with Icarus Verilog.
   7. Return a summary of the actions you took and the changes you have made.

## Tips
Do not ask for permission, just edit files as needed and return when you have finished.
Be sure to keep careful track of all key implementation details from the documents provided.
Weigh the different possible implementations, and choose the best one suited for the high level requirements.
"""

    try:

        cmd = [
            "opencode",
            "run",
            SYSTEM_PROMPT,                            # prompt for codex
            "--format",
            "json", 
        ]
        print("Running Command")


        # Open a subprocess and stream its combined stdout/stderr to our stdout as it appears,
        # while also collecting the output to write to a file afterwards.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        output_lines = []
        if proc.stdout is not None:
            for line in iter(proc.stdout.readline, ""):
                sys.stdout.write(line)
                sys.stdout.flush()
                output_lines.append(line)
        proc.wait()
        opencode_returncode = proc.returncode
        opencode_stdout_combined = "".join(output_lines)

        # Also print a summary line for clarity
        sys.stdout.write(f"\nOpenCode return code: {opencode_returncode}\n")
        sys.stdout.flush()

        # Count and print tokens for poRTLe tracking
        total_tokens = count_tokens(output_lines)
        print(f"Total tokens: {total_tokens:,}")

    except Exception as e:
        print(f"An unexpected error occurred during simulation: {str(e)}")

    print("Agent execution completed successfully")
    sys.exit(0)

if __name__ == "__main__":
    main() 
