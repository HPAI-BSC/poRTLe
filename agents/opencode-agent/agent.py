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
from prompts import SYSTEM_PROMPT
from pathlib import Path

def main():
    """Main agent function"""
    print("Starting CVDP opencode-agent...")

    # Handle using OpenAI API 
    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_USER_KEY")

    try:

        cmd = [
            "opencode",
            "run",
            SYSTEM_PROMPT,                            # prompt for codex
            "--format",
            "json", 
        ]
        print("Running Command")

        # opencode_cmd = subprocess.run(cmd, check=False, capture_output=True, text=True, input="")

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

    except Exception as e:
        print(f"An unexpected error occurred during simulation: {str(e)}")

    print("Agent execution completed successfully")
    sys.exit(0)

if __name__ == "__main__":
    main() 
