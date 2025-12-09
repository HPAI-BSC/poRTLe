#!/usr/bin/env python3
"""
Test runner template for DIY tasks.
This script always passes - replace with actual test logic.
"""

import subprocess
import sys
from pathlib import Path


def run_test():
    """Validate that ../rtl/portle.md exists and print its contents."""

    print("=" * 60)
    print("DIY Task Test Runner - portle.md Validation")
    print("=" * 60)

    # Check for ../rtl/portle.md relative to this script's location
    script_dir = Path(__file__).parent
    portle_md_path = script_dir / ".." / "rtl" / "portle.md"
    portle_md_path = portle_md_path.resolve()

    print(f"\nLooking for: {portle_md_path}")

    if portle_md_path.exists():
        print(f"\n✅ Found portle.md at: {portle_md_path}")
        print("\n" + "-" * 60)
        print("Contents of portle.md:")
        print("-" * 60)
        try:
            contents = portle_md_path.read_text()
            print(contents)
            print("-" * 60)
            print("\n✅ Validation PASSED")
            return True
        except Exception as e:
            print(f"\n❌ Error reading file: {e}")
            return False
    else:
        print(f"\n❌ FAILED: portle.md not found at: {portle_md_path}")
        print("Expected location: ../rtl/portle.md (relative to src/)")
        return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
