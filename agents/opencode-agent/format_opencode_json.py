#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for replaying OpenCode/Claude NDJSON logs with ANSI colors."""

import sys
import json
import os
import re

# ANSI color codes used while replaying logs
COLORS = {
    'RESET': '\033[0m',
    'BOLD': '\033[1m',
    'DIM': '\033[2m',
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'ORANGE': '\033[38;5;214m',
    'RED': '\033[91m',
    'CYAN': '\033[96m',
    'MAGENTA': '\033[95m',
}

def emit(*parts, sep=' ', end='\n'):
    """Write output immediately so it shows up in streaming environments."""
    sys.stdout.write(sep.join(str(part) for part in parts) + end)
    sys.stdout.flush()

def colorize(text, color):
    """Add color to text"""
    return f"{COLORS.get(color, '')}{text}{COLORS['RESET']}"

LINE_NUMBER_PATTERN = re.compile(r"(?m)^(\d{5}\|)")

def dim_line_numbers(text):
    """Dim leading 5-digit line numbers (e.g. 00001|) inside file excerpts"""
    if not text:
        return text
    return LINE_NUMBER_PATTERN.sub(lambda match: colorize(match.group(1), 'DIM'), text)

def process_line(line, total_stats=None):
    """Process a single NDJSON line from OpenCode-style logs"""
    try:
        data = json.loads(line)
        msg_type = data.get('type')

        # OpenCode format handling
        if msg_type == 'step_start':
            session_id = data.get('sessionID', 'unknown')
            session_text = colorize(f"(session: {session_id[:16]}...)", 'DIM')
            emit(f"\n{colorize('▶ Step started', 'CYAN')} {session_text}")

        elif msg_type == 'text':
            part = data.get('part', {})
            text = part.get('text', '').strip()
            if text:
                emit(f"{colorize('💬 Agent:', 'GREEN')} {text}")

        elif msg_type == 'tool_use':
            part = data.get('part', {})
            tool_name = part.get('tool', '')
            state = part.get('state', {})
            tool_input = state.get('input', {})
            status = state.get('status', 'unknown')

            # Format the tool call
            if tool_name == "bash":
                cmd = tool_input.get('command', '')
                emit(f"{colorize('🔧 BASH:', 'YELLOW')} {cmd}")
            elif tool_name == "read":
                path = tool_input.get('filePath', '')
                emit(f"{colorize('📖 READ:', 'ORANGE')} {path}")
            elif tool_name == "write":
                path = tool_input.get('filePath', '')
                emit(f"{colorize('✏️  WRITE:', 'ORANGE')} {path}")
            elif tool_name == "list":
                path = tool_input.get('path', '.')
                emit(f"{colorize('📂 LIST:', 'ORANGE')} {path}")
            elif tool_name.startswith("skills_"):
                skill = tool_name.replace("skills_", "")
                emit(f"{colorize('⚡ SKILL:', 'MAGENTA')} {skill}")
            else:
                emit(f"{colorize(f'🔧 {tool_name.upper()}:', 'YELLOW')} {json.dumps(tool_input, indent=2)}")

            # Show output if completed
            if status == 'completed':
                output = state.get('output', '')
                if output and len(output) > 0:
                    if len(output) > 300:
                        snippet = dim_line_numbers(output[:150])
                        emit(f"   {colorize('✓ Result:', 'BOLD')} {snippet}... ({len(output)} chars)")
                    else:
                        # For short outputs, show them
                        if '\n' in output and len(output) > 100:
                            emit(f"   {colorize('✓ Result:', 'BOLD')} (multiline, {len(output)} chars)")
                        else:
                            snippet = dim_line_numbers(output[:150])
                            emit(f"   {colorize('✓ Result:', 'BOLD')} {snippet}")

        elif msg_type == 'step_finish':
            part = data.get('part', {})
            reason = part.get('reason', 'unknown')
            cost = part.get('cost', 0)
            tokens = part.get('tokens', {})
            cache = tokens.get('cache', {})
            input_tokens = tokens.get('input', 0)
            output_tokens = tokens.get('output', 0)
            reasoning_tokens = tokens.get('reasoning', 0)
            cache_read = cache.get('read', 0)
            cache_write = cache.get('write', 0)

            details = f"{reason} (cost: ${cost:.6f}, tokens: {input_tokens}→{output_tokens})"
            emit(f"{colorize('✓ Step finished:', 'BLUE')} {colorize(details, 'DIM')}")

            # Update total stats if provided
            if total_stats is not None:
                total_stats['steps'] += 1
                total_stats['input_tokens'] += input_tokens
                total_stats['output_tokens'] += output_tokens
                total_stats['reasoning_tokens'] += reasoning_tokens
                total_stats['cache_read'] += cache_read
                total_stats['cache_write'] += cache_write
                total_stats['total_cost'] += cost

    except json.JSONDecodeError:
        # Not JSON, might be regular output
        if line.strip():
            emit(line.strip())
    except Exception as e:
        print(f"{colorize('Parse error:', 'RED')} {e}", file=sys.stderr)

def run_agent(log_file_path):
    """Replay a log file and pretty-print the contained JSON lines."""
    emit(colorize("OpenCode Log Viewer", 'BOLD'))
    emit(colorize("=" * 80, 'DIM'))
    emit(f"Reading logs from: {log_file_path}\n")

    # Token tracking
    total_stats = {
        'input_tokens': 0,
        'output_tokens': 0,
        'reasoning_tokens': 0,
        'cache_read': 0,
        'cache_write': 0,
        'total_cost': 0.0,
        'steps': 0
    }

    try:
        # Check if the log file exists
        if not os.path.exists(log_file_path):
            emit(f"{colorize('ERROR:', 'RED')} Log file not found: {log_file_path}")
            return

        # Read and parse each line from the log file
        line_count = 0
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                line = line.strip()
                if line:  # Skip empty lines
                    process_line(line, total_stats)
                    line_count += 1

        # Print summary
        emit(f"\n{colorize('='*80, 'DIM')}")
        emit(f"{colorize('Log replay complete!', 'GREEN')}")
        emit(f"{colorize('Total lines processed:', 'BOLD')} {line_count}")
        emit(f"{colorize('Total steps:', 'BOLD')} {total_stats['steps']}\n")

        # Display token statistics
        emit(f"{colorize('Token Usage Summary:', 'BOLD')}")
        emit(f"  {colorize('Input tokens:', 'CYAN')} {total_stats['input_tokens']:,}")
        emit(f"  {colorize('Output tokens:', 'CYAN')} {total_stats['output_tokens']:,}")
        if total_stats['reasoning_tokens'] > 0:
            emit(f"  {colorize('Reasoning tokens:', 'CYAN')} {total_stats['reasoning_tokens']:,}")
        if total_stats['cache_read'] > 0:
            emit(f"  {colorize('Cache read:', 'CYAN')} {total_stats['cache_read']:,}")
        if total_stats['cache_write'] > 0:
            emit(f"  {colorize('Cache write:', 'CYAN')} {total_stats['cache_write']:,}")

        total_tokens = (total_stats['input_tokens'] + total_stats['output_tokens'] +
                       total_stats['reasoning_tokens'])
        emit(f"  {colorize('Total tokens:', 'BOLD')} {total_tokens:,}")
        emit(f"  {colorize('Total cost:', 'BOLD')} ${total_stats['total_cost']:.6f}\n")

    except Exception as e:
        emit(f"{colorize('An unexpected error occurred reading log file:', 'RED')} {str(e)}")
        return

    emit(colorize("Log display completed successfully", 'GREEN'))

def main():
    """CLI entry point."""
    # Allow specifying the log file as the first CLI arg
    log_file_path = sys.argv[1] if len(sys.argv) > 1 else "output.txt"

    # Print current working directory as first output
    emit(f"{colorize('Working Directory:', 'BOLD')} {os.getcwd()}\n")

    try:
        run_agent(log_file_path=log_file_path)
        sys.exit(0)
    except KeyboardInterrupt:
        emit(f"\n{colorize('Interrupted by user', 'YELLOW')}")
        sys.exit(0)

if __name__ == "__main__":
    main()
