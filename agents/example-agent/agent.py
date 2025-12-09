#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple CVDP agent implementation for the agentic workflow.
This agent reads prompt.json and makes changes to files in the mounted directories.
"""

import os
import json
import sys
import glob
import time
import random

def read_prompt():
    """Read the prompt from prompt.json"""
    try:
        with open("/code/prompt.json", "r") as f:
            prompt_data = json.load(f)
            return prompt_data.get("prompt", "")
    except Exception as e:
        print(f"Error reading prompt.json: {e}")
        return ""

def list_directory_files(dir_path):
    """List all files in a directory recursively"""
    files = []
    if os.path.exists(dir_path):
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, dir_path)
                files.append(rel_path)
    return files

def read_file(file_path):
    """Read the contents of a file"""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def write_file(file_path, content):
    """Write content to a file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Successfully wrote to {file_path}")
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")

def analyze_prompt_and_modify_files(prompt):
    """Analyze the prompt and modify files accordingly"""
    print(f"Analyzing prompt: {prompt}")
    
    # List files in each directory
    rtl_files = list_directory_files("/code/rtl")
    verif_files = list_directory_files("/code/verif")
    docs_files = list_directory_files("/code/docs")
    
    print(f"Found {len(rtl_files)} RTL files")
    # Process RTL files to replace "input" with "loompa"
    for rtl_file in rtl_files:
        rtl_path = os.path.join("/code/rtl", rtl_file)
        try:
            # Read the file content
            with open(rtl_path, 'r') as file:
                content = file.read()
            
            # Replace all occurrences of "input" with "loompa"
            modified_content = content.replace("input", "loompa")
            
            # Write the modified content back to the file
            with open(rtl_path, 'w') as file:
                file.write(modified_content)
                
            print(f"Replaced 'input' with 'loompa' in {rtl_file}")
        except Exception as e:
            print(f"Error processing {rtl_file}: {e}")
    
    # Example: Add a timestamp file to the rundir
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    write_file("/code/rundir/agent_executed.txt", f"Agent executed at {timestamp}\nPrompt: {prompt}")
    
    # Example: Check if we need to modify an RTL file based on prompt
    if "fix" in prompt.lower() and rtl_files:
        # Example: Take the first RTL file and add a comment
        rtl_file = rtl_files[0]
        rtl_path = os.path.join("/code/rtl", rtl_file)
        
        content = read_file(rtl_path)
        modified_content = f"// Modified by agent at {timestamp}\n" + content
        
        write_file(rtl_path, modified_content)
        print(f"Modified RTL file: {rtl_file}")
    
    # Example: Add a documentation file
    if docs_files or "document" in prompt.lower():
        write_file("/code/docs/agent_report.md", f"""# Agent Report
        
## Execution Summary
- Executed at: {timestamp}
- Prompt: {prompt}
- RTL files found: {len(rtl_files)}
- Verification files found: {len(verif_files)}
        
## Analysis
This is a sample agent report created during the agentic workflow execution.
        """)
        print("Created documentation file: agent_report.md")

def main():
    """Main agent function"""
    # print("Starting CVDP agent...")
    
    # # Read the prompt
    prompt = read_prompt()
    if not prompt:
        print("No prompt found in prompt.json. Exiting.")
        sys.exit(1)
    
    # Process the prompt and modify files
    analyze_prompt_and_modify_files(prompt)
    
    print("Agent execution completed successfully")
    token_count = random.randint(100, 100000)
    print(f"Total Final Tokens: {token_count}")
    sys.exit(0)

if __name__ == "__main__":
    main() 