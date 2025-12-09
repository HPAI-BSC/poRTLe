#!/usr/bin/env python3
"""
clean_task.py

Clean up DIY task directories by removing binary files, Git repositories,
and other artifacts that shouldn't be included in benchmark datasets.

This script helps prepare task directories for use with create_diy_task.py
by removing:
- Git repositories and version control files
- Binary documentation (PDFs, images)
- Compiled binaries and test executables
- Simulation artifacts and generated files
- Large binary files that cause JSONL bloat

Usage:
    python diy_tasks/clean_task.py <task_directory> [--dry-run]

    Or import and use programmatically:
    from diy_tasks.clean_task import clean_task_directory
    clean_task_directory(Path("diy_tasks/my_task"), dry_run=False)
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple


# File patterns to remove
PATTERNS_TO_REMOVE = {
    # Version control
    ".git": "Git repository",
    ".githooks": "Git hooks",
    ".gitignore": "Git ignore file",
    ".gitmodules": "Git submodules",
    ".gitattributes": "Git attributes",

    # Binary documentation
    "*.pdf": "PDF files",
    "*.PDF": "PDF files",
    "*.png": "PNG images",
    "*.PNG": "PNG images",
    "*.jpg": "JPEG images",
    "*.JPG": "JPEG images",
    "*.jpeg": "JPEG images",
    "*.JPEG": "JPEG images",
    "*.gif": "GIF images",
    "*.GIF": "GIF images",
    "*.svg": "SVG images",
    "*.SVG": "SVG images",
    "*.bmp": "Bitmap images",
    "*.BMP": "Bitmap images",

    # Office documents
    "*.doc": "Word documents",
    "*.docx": "Word documents",
    "*.xls": "Excel files",
    "*.xlsx": "Excel files",
    "*.ppt": "PowerPoint files",
    "*.pptx": "PowerPoint files",
    "*.ods": "OpenDocument spreadsheets",
    "*.odt": "OpenDocument text",
    "*.odp": "OpenDocument presentations",

    # CAD and design files
    "*.dwg": "AutoCAD files",
    "*.dxf": "DXF files",

    # Simulation artifacts
    "*.qdb": "Simulation database files",
    "*.qtl": "Simulation files",
    "*.qpg": "Simulation files",
    "*.wlf": "Waveform files",
    "*.vcd": "VCD waveform files (keep if small/example)",

    # Compiled binaries and objects
    "*.o": "Object files",
    "*.a": "Archive files",
    "*.so": "Shared libraries",
    "*.dylib": "Dynamic libraries",
    "*.dll": "Windows DLLs",
    "*.exe": "Executables",
    "*.out": "Output executables",

    # Archive files
    "*.zip": "ZIP archives",
    "*.tar": "TAR archives",
    "*.gz": "Gzip files",
    "*.bz2": "Bzip2 files",
    "*.7z": "7-Zip archives",
    "*.rar": "RAR archives",

    # Cache and temporary files
    "__pycache__": "Python cache",
    "*.pyc": "Python compiled files",
    "*.pyo": "Python optimized files",
    ".DS_Store": "macOS metadata",
    "Thumbs.db": "Windows thumbnails",
    "*.swp": "Vim swap files",
    "*.swo": "Vim swap files",
    "*~": "Backup files",
}

# Additional patterns for RISC-V test binaries
RISCV_TEST_PATTERNS = [
    "rv32*-p-*",
    "rv32*-v-*",
    "rv64*-p-*",
    "rv64*-v-*",
]


def should_remove_file(file_path: Path, patterns: dict) -> Tuple[bool, str]:
    """
    Check if a file should be removed based on patterns.

    Args:
        file_path: Path to check
        patterns: Dictionary of patterns to match

    Returns:
        Tuple of (should_remove, reason)
    """
    # Check directory patterns
    for pattern, reason in patterns.items():
        if not pattern.startswith("*."):
            if file_path.name == pattern or pattern in str(file_path):
                return True, reason

    # Check file extension patterns
    for pattern, reason in patterns.items():
        if pattern.startswith("*."):
            ext = pattern[1:]  # Remove the *
            if file_path.name.endswith(ext):
                return True, reason

    # Check RISC-V test binary patterns
    for pattern in RISCV_TEST_PATTERNS:
        # Remove the wildcard for simple matching
        prefix = pattern.split('*')[0]
        suffix = pattern.split('*')[-1] if '*' in pattern else ''

        if file_path.name.startswith(prefix):
            # Only remove if it's a binary (no extension or non-text extension)
            if not file_path.suffix or file_path.suffix not in ['.sv', '.v', '.vh', '.svh', '.txt', '.md', '.sh']:
                return True, f"RISC-V test binary matching {pattern}"

    return False, ""


def get_files_to_remove(task_dir: Path, patterns: dict) -> List[Tuple[Path, str]]:
    """
    Get list of files that should be removed.

    Args:
        task_dir: Root directory to scan
        patterns: Patterns to match

    Returns:
        List of (file_path, reason) tuples
    """
    files_to_remove = []

    for root, dirs, files in os.walk(task_dir):
        root_path = Path(root)

        # Check directories
        for dir_name in dirs[:]:  # Copy list to allow modification
            dir_path = root_path / dir_name
            should_remove, reason = should_remove_file(dir_path, patterns)
            if should_remove:
                files_to_remove.append((dir_path, reason))
                dirs.remove(dir_name)  # Don't traverse into this directory

        # Check files
        for file_name in files:
            file_path = root_path / file_name
            should_remove, reason = should_remove_file(file_path, patterns)
            if should_remove:
                files_to_remove.append((file_path, reason))

    return files_to_remove


def clean_task_directory(task_dir: Path, dry_run: bool = False, verbose: bool = True) -> dict:
    """
    Clean a DIY task directory by removing unwanted files.

    Args:
        task_dir: Path to task directory
        dry_run: If True, only show what would be removed
        verbose: If True, print detailed output

    Returns:
        Dictionary with cleanup statistics
    """
    if not task_dir.exists():
        raise ValueError(f"Task directory does not exist: {task_dir}")

    if not task_dir.is_dir():
        raise ValueError(f"Path is not a directory: {task_dir}")

    # Get files to remove
    files_to_remove = get_files_to_remove(task_dir, PATTERNS_TO_REMOVE)

    stats = {
        "total_files": len(files_to_remove),
        "removed_files": 0,
        "removed_dirs": 0,
        "freed_bytes": 0,
        "errors": [],
    }

    if verbose:
        mode = "DRY RUN - " if dry_run else ""
        print(f"{mode}Cleaning task directory: {task_dir}")
        print(f"Found {len(files_to_remove)} items to remove\n")

    # Remove files and directories
    for file_path, reason in sorted(files_to_remove):
        try:
            # Get size before removal
            if file_path.exists():
                if file_path.is_file():
                    size = file_path.stat().st_size
                elif file_path.is_dir():
                    size = sum(f.stat().st_size for f in file_path.rglob('*') if f.is_file())
                else:
                    size = 0

                if verbose:
                    rel_path = file_path.relative_to(task_dir)
                    size_str = format_size(size)
                    print(f"  {'[DRY RUN] ' if dry_run else ''}Removing: {rel_path} ({reason}, {size_str})")

                if not dry_run:
                    if file_path.is_dir():
                        shutil.rmtree(file_path)
                        stats["removed_dirs"] += 1
                    else:
                        file_path.unlink()
                        stats["removed_files"] += 1

                    stats["freed_bytes"] += size

        except Exception as e:
            error_msg = f"Failed to remove {file_path}: {e}"
            stats["errors"].append(error_msg)
            if verbose:
                print(f"  ERROR: {error_msg}")

    if verbose:
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Cleanup complete!")
        print(f"  Files removed: {stats['removed_files']}")
        print(f"  Directories removed: {stats['removed_dirs']}")
        print(f"  Space freed: {format_size(stats['freed_bytes'])}")
        if stats["errors"]:
            print(f"  Errors: {len(stats['errors'])}")

    return stats


def format_size(bytes_size: int) -> str:
    """Format byte size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def main():
    """Command-line interface for cleaning task directories."""
    parser = argparse.ArgumentParser(
        description="Clean DIY task directory by removing binary files and artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what would be removed (dry run)
  python diy_tasks/clean_task.py diy_tasks/my_task --dry-run

  # Actually clean the directory
  python diy_tasks/clean_task.py diy_tasks/my_task

  # Clean quietly (only show summary)
  python diy_tasks/clean_task.py diy_tasks/my_task --quiet
        """
    )

    parser.add_argument(
        "task_dir",
        type=Path,
        help="Path to DIY task directory to clean"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing anything"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output, only show summary"
    )

    args = parser.parse_args()

    try:
        stats = clean_task_directory(
            args.task_dir,
            dry_run=args.dry_run,
            verbose=not args.quiet
        )

        # Exit with error code if there were errors
        if stats["errors"]:
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
