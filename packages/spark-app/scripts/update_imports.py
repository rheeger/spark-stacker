#!/usr/bin/env python3
"""
Script to update import statements to use app. prefix across the entire codebase.
This helps align tests and Docker setup with the new import structure.
"""

import argparse
import os
import re
import sys
from pathlib import Path


def update_imports(file_path, dry_run=False):
    """
    Update imports in a file to use the app. prefix for internal modules.

    Args:
        file_path: Path to the file to update
        dry_run: If True, don't actually modify files, just print what would be changed

    Returns:
        int: Number of lines changed
    """
    # Modules that need to be prefixed with app.
    modules = [
        'connectors', 'core', 'indicators', 'metrics',
        'risk_management', 'utils', 'webhook'
    ]

    # Combine into a regex pattern for both 'import X' and 'from X import Y' formats
    import_pattern = re.compile(r'^(from|import)\s+(' + '|'.join(modules) + r')(\s+import|\s*\.|\s*$)')

    # Read the file content
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Process each line
    changed_lines = 0
    updated_lines = []
    for line in lines:
        match = import_pattern.match(line)
        if match:
            # Skip if 'app.' is already present
            if "app." not in line:
                from_or_import = match.group(1)
                module = match.group(2)
                rest = match.group(3) or ''

                # Replace the module name with app.module
                new_line = f"{from_or_import} app.{module}{rest}{line[match.end():]}"
                updated_lines.append(new_line)
                changed_lines += 1

                if dry_run:
                    print(f"Would change: {line.strip()} -> {new_line.strip()}")
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    # Write the updated content back to the file
    if not dry_run and changed_lines > 0:
        with open(file_path, 'w') as f:
            f.writelines(updated_lines)

    return changed_lines

def main():
    parser = argparse.ArgumentParser(description='Update imports to use app. prefix')
    parser.add_argument('--dir', default='.', help='Directory to process (default: current directory)')
    parser.add_argument('--dry-run', action='store_true', help='Print changes without modifying files')
    args = parser.parse_args()

    # Find Python files
    root_dir = Path(args.dir)
    python_files = list(root_dir.glob('**/*.py'))

    total_files = 0
    total_changes = 0

    print(f"Scanning {len(python_files)} Python files...")

    for file_path in python_files:
        changes = update_imports(file_path, args.dry_run)
        if changes > 0:
            total_files += 1
            total_changes += changes
            if not args.dry_run:
                print(f"Updated {changes} imports in {file_path}")

    if args.dry_run:
        print(f"\nDRY RUN: Would update {total_changes} imports in {total_files} files")
    else:
        print(f"\nUpdated {total_changes} imports in {total_files} files")

if __name__ == "__main__":
    main()
