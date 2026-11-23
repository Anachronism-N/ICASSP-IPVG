#!/usr/bin/env python3
"""
Consolidate Matrix Dataset
==========================
Analyzes the fragmented Matrix dataset across two cache directories,
identifies duplicates and corrupted files, and generates a clean manifest.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Paths
MANIFEST_PATH = "/srv/wm_teacher/dataset/manifests/matrix_archive_audit.jsonl"
DATASET_BASE = "/srv/wm_teacher/dataset/general_action/matrix_dataset"
MAIN_DATA_DIR = os.path.join(DATASET_BASE, "data")
SNAPSHOT_DATA_DIR = os.path.join(
    DATASET_BASE,
    "datasets/TheMatrixDataset--the_matrix_dataset_8M_1920_1080/snapshots/master/data"
)
OUTPUT_MANIFEST = "/srv/wm_teacher/dataset/manifests/matrix_consolidated_manifest.jsonl"

# Known .incomplete files (from context, confirmed on disk)
INCOMPLETE_FILES = [
    ("part_111.zip.incomplete", "part_111.zip"),
    ("part_1109.zip.incomplete", "part_1109.zip"),
    ("part_1110.zip.incomplete", "part_1110.zip"),
    ("part_1111.zip.incomplete", "part_1111.zip"),
]


def load_manifest(manifest_path):
    """Load the audit manifest JSONL file."""
    entries = []
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Loaded {len(entries)} entries from manifest")
    return entries


def get_part_number(filename):
    """Extract the part number from a filename like 'part_123.zip'."""
    # e.g., part_0.zip -> 0, part_1003.zip -> 1003
    name = filename.replace('.zip', '')
    parts = name.split('_')
    if len(parts) >= 2:
        return int(parts[1])
    return None


def is_in_main_dir(path):
    """Check if path is in the main data/ directory (not the snapshot dir)."""
    return not ("datasets/TheMatrixDataset" in path)


def analyze_manifest(entries):
    """Group entries by filename and analyze duplicates."""
    # Group by part number
    by_part = defaultdict(list)
    for entry in entries:
        part = get_part_number(entry['filename'])
        if part is not None:
            by_part[part].append(entry)

    # Analyze each group
    keep = {}           # part_number -> entry to keep
    remove = []         # entries to remove (duplicates)
    corrupted = []      # corrupted entries (invalid_zip, unique)
    duplicate_corrupted = []  # corrupted entries that also have a valid copy

    for part, group in sorted(by_part.items()):
        if len(group) == 1:
            entry = group[0]
            if entry['status'] == 'central_directory_ok':
                keep[part] = entry
            else:
                corrupted.append(entry)
        else:
            # Multiple entries for same part - duplicate
            main_entries = [e for e in group if is_in_main_dir(e['path'])]
            snapshot_entries = [e for e in group if not is_in_main_dir(e['path'])]

            # First, try to find a valid copy
            valid_entries = [e for e in group if e['status'] == 'central_directory_ok']
            if valid_entries:
                # Prefer main data/ dir, then any valid
                main_valid = [e for e in valid_entries if is_in_main_dir(e['path'])]
                if main_valid:
                    keep[part] = main_valid[0]
                else:
                    keep[part] = valid_entries[0]

                # Mark others for removal
                for e in group:
                    if e is not keep[part]:
                        if e['status'] == 'central_directory_ok':
                            remove.append(e)
                        else:
                            duplicate_corrupted.append(e)
            else:
                # All copies are invalid - keep one as corrupted
                corrupted.append(group[0])
                for e in group[1:]:
                    remove.append(e)

    return keep, remove, corrupted, duplicate_corrupted


def analyze_incomplete_files(snapshot_dir):
    """Check for .incomplete files."""
    incomplete_info = []
    for incomplete_name, target_name in INCOMPLETE_FILES:
        fpath = os.path.join(snapshot_dir, incomplete_name)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            incomplete_info.append({
                'incomplete_path': fpath,
                'target_name': target_name,
                'size_bytes': size,
            })
    return incomplete_info


def format_bytes(size):
    """Format bytes to human-readable string."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024**2:
        return f"{size/1024:.1f} KB"
    elif size < 1024**3:
        return f"{size/1024**2:.1f} MB"
    else:
        return f"{size/1024**3:.2f} GB"


def generate_report(keep, remove, corrupted, duplicate_corrupted, incomplete_info, entries):
    """Generate and print the consolidation report."""
    print("=" * 70)
    print("  MATRIX DATASET CONSOLIDATION REPORT")
    print("=" * 70)

    # Summary
    total_entries = len(entries)
    unique_valid = len(keep)
    duplicates_removed = len(remove)
    total_corrupted = len(corrupted) + len(duplicate_corrupted)

    print(f"\n  Manifest entries analyzed: {total_entries}")
    print(f"  Unique valid archives to keep: {unique_valid}")
    print(f"  Duplicates to remove: {duplicates_removed}")
    print(f"  Corrupted files needing replacement: {total_corrupted}")
    print(f"  Incomplete files needing download: {len(incomplete_info)}")

    # Space saved from duplicate removal
    saved_bytes = sum(r.get('archive_bytes', 0) for r in remove)
    print(f"\n  Space saved by removing duplicates: {format_bytes(saved_bytes)}")

    # Breakdown by directory
    main_valid = sum(1 for e in keep.values() if is_in_main_dir(e['path']))
    snap_valid = unique_valid - main_valid
    print(f"\n  Valid archives by directory:")
    print(f"    Main data/ dir: {main_valid}")
    print(f"    Snapshot datasets/ dir: {snap_valid}")

    # Corrupted files detail
    all_corrupted = list(corrupted) + list(duplicate_corrupted)
    if all_corrupted:
        print(f"\n  Corrupted files (need replacement):")
        for c in all_corrupted:
            size = c.get('archive_bytes', 0)
            reason = c.get('error', 'unknown')
            print(f"    - {c['path']}")
            print(f"      Size on disk: {format_bytes(size)}, Error: {reason}")

    # Duplicate detail
    if remove:
        print(f"\n  Duplicate archives to remove:")
        # Group by directory for readability
        main_deleted = [r for r in remove if is_in_main_dir(r['path'])]
        snap_deleted = [r for r in remove if not is_in_main_dir(r['path'])]
        if main_deleted:
            print(f"    From main data/ ({len(main_deleted)} files):")
            for r in sorted(main_deleted, key=lambda x: get_part_number(x['filename']) or 0):
                print(f"      {r['filename']} ({format_bytes(r.get('archive_bytes', 0))})")
        if snap_deleted:
            print(f"    From snapshot datasets/ ({len(snap_deleted)} files):")
            for r in sorted(snap_deleted, key=lambda x: get_part_number(x['filename']) or 0):
                print(f"      {r['filename']} ({format_bytes(r.get('archive_bytes', 0))})")

    # Incomplete files detail
    if incomplete_info:
        print(f"\n  Incomplete files needing download:")
        for info in incomplete_info:
            print(f"    - {os.path.basename(info['incomplete_path'])}")
            print(f"      Current size: {format_bytes(info['size_bytes'])}")
            print(f"      Target name: {info['target_name']}")

    # Duplicate corrupted detail (these are duplicate entries where one is valid, one corrupted)
    if duplicate_corrupted:
        print(f"\n  Corrupted duplicates (valid copy exists):")
        for c in duplicate_corrupted:
            print(f"    - {c['filename']}: {c['path']} (will be removed)")

    # Validation
    expected = unique_valid + duplicates_removed + total_corrupted
    if expected != total_entries:
        print(f"\n  WARNING: Entry count mismatch!")
        print(f"    Keep ({unique_valid}) + Remove ({duplicates_removed}) + Corrupted ({total_corrupted})")
        print(f"    = {expected}, but manifest has {total_entries}")
    else:
        print(f"\n  Entry count check: {unique_valid} + {duplicates_removed} + {total_corrupted} = {total_entries} PASS")

    # Total space in clean dataset
    clean_size = sum(e.get('archive_bytes', 0) for e in keep.values())
    print(f"\n  Clean dataset total size: {format_bytes(clean_size)}")


def write_clean_manifest(keep, output_path):
    """Write the clean manifest of unique valid archives."""
    with open(output_path, 'w') as f:
        for part in sorted(keep.keys()):
            entry = keep[part]
            f.write(json.dumps(entry) + '\n')
    print(f"\n  Clean manifest written to: {output_path}")
    print(f"  Total entries: {len(keep)}")


def write_download_list(corrupted, duplicate_corrupted, incomplete_info, output_dir):
    """Write lists of files needing download/replacement."""
    all_corrupted = list(corrupted) + list(duplicate_corrupted)

    # Files that need re-download because they're corrupted
    corrupted_list_path = os.path.join(output_dir, "matrix_files_to_re_download.txt")
    with open(corrupted_list_path, 'w') as f:
        for c in sorted(all_corrupted, key=lambda x: x['filename']):
            f.write(f"{c['filename']}  # {c.get('error', 'corrupted')}\n")
    print(f"  Re-download list written to: {corrupted_list_path} ({len(all_corrupted)} files)")

    # Files that need initial download (incomplete)
    if incomplete_info:
        incomplete_list_path = os.path.join(output_dir, "matrix_files_to_download.txt")
        with open(incomplete_list_path, 'w') as f:
            for info in incomplete_info:
                f.write(f"{info['target_name']}  # was {os.path.basename(info['incomplete_path'])} ({format_bytes(info['size_bytes'])})\n")
        print(f"  Download list written to: {incomplete_list_path} ({len(incomplete_info)} files)")


def write_removal_script(remove, duplicate_corrupted, output_dir):
    """Write a script that can be run to remove duplicate files."""
    to_remove = list(remove) + list(duplicate_corrupted)
    script_path = os.path.join(output_dir, "remove_duplicates.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Script to remove duplicate Matrix dataset files\n")
        f.write("# Review carefully before running!\n\n")
        total = 0
        for entry in sorted(to_remove, key=lambda x: x['path']):
            abs_path = os.path.join("/srv/wm_teacher", entry['path'])
            f.write(f"rm -v \"{abs_path}\"\n")
            total += entry.get('archive_bytes', 0)
        f.write(f"\n# Total space to be freed: {format_bytes(total)}\n")
    os.chmod(script_path, 0o755)
    print(f"  Removal script written to: {script_path}")
    print(f"  Would free: {format_bytes(total)}")
    print(f"  Review and run manually if you wish to delete the files.")


def main():
    print("Loading manifest...")
    entries = load_manifest(MANIFEST_PATH)

    print("Analyzing entries...")
    keep, remove, corrupted, duplicate_corrupted = analyze_manifest(entries)

    print("Checking for incomplete files...")
    incomplete_info = analyze_incomplete_files(SNAPSHOT_DATA_DIR)

    # Generate report
    generate_report(keep, remove, corrupted, duplicate_corrupted, incomplete_info, entries)

    # Write outputs
    manifest_dir = os.path.dirname(OUTPUT_MANIFEST)
    write_clean_manifest(keep, OUTPUT_MANIFEST)
    write_download_list(corrupted, duplicate_corrupted, incomplete_info, manifest_dir)
    write_removal_script(remove, duplicate_corrupted, manifest_dir)

    # Return data for testability
    return {
        'keep': keep,
        'remove': remove,
        'corrupted': corrupted,
        'duplicate_corrupted': duplicate_corrupted,
        'incomplete': incomplete_info,
    }


if __name__ == '__main__':
    results = main()
