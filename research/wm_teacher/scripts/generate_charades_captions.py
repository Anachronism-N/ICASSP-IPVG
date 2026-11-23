#!/usr/bin/env python3
"""
Generate proper captions for Charades videos by mapping video IDs to action classes.
Uses the official Charades annotations (train + test CSV) and class definitions.
"""

import csv
import os
import json
import re
import sys

# Paths
CLASSES_FILE = "/tmp/Charades_annotations/Charades/Charades_v1_classes.txt"
TRAIN_CSV = "/tmp/Charades_annotations/Charades/Charades_v1_train.csv"
TEST_CSV = "/tmp/Charades_annotations/Charades/Charades_v1_test.csv"
VIDEO_DIR = "/srv/wm_teacher/dataset/wan22_training/charades_processed"
OUTPUT_FILE = os.path.join(VIDEO_DIR, "manifest.jsonl")


def load_class_map(classes_file):
    """Load c000 -> action description mapping."""
    class_map = {}
    with open(classes_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                class_id, description = parts
                class_map[class_id] = description
    print(f"Loaded {len(class_map)} action classes")
    return class_map


def load_annotations(train_csv, test_csv):
    """Load video_id -> list of action class IDs from train and test CSVs."""
    vid_to_actions = {}

    for csv_file, split_name in [(train_csv, 'train'), (test_csv, 'test')]:
        count = 0
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row['id']
                actions_str = row.get('actions', '')
                action_ids = []
                if actions_str.strip():
                    for action_seg in actions_str.split(';'):
                        action_seg = action_seg.strip()
                        if action_seg:
                            # Format: "cXXX start end"
                            class_id = action_seg.split(' ')[0]
                            action_ids.append(class_id)
                vid_to_actions[vid] = action_ids
                count += 1
        print(f"Loaded {count} videos from {split_name} split")

    print(f"Total unique videos with annotations: {len(vid_to_actions)}")
    return vid_to_actions


def naturalize_label(label):
    """Convert annotation label to natural English gerund phrase."""
    # Remove "Someone is" prefix
    if label.startswith("Someone is "):
        rest = label[len("Someone is "):]
        # Convert to gerund if not already
        if not rest.endswith('ing'):
            # Handle common patterns
            words = rest.split()
            if words[0] in ('going', 'standing', 'sitting', 'walking', 'running'):
                pass  # Already gerund-ish
            elif words[0] == 'dressing':
                rest = 'getting dressed'
            elif words[0] == 'undressing':
                rest = 'getting undressed'
            elif words[0] == 'cooking':
                rest = 'cooking something'
            elif words[0] == 'eating':
                rest = 'eating something'
            elif words[0] == 'laughing':
                rest = 'laughing'
            elif words[0] == 'smiling':
                rest = 'smiling'
            elif words[0] == 'sneezing':
                rest = 'sneezing'
            elif words[0] == 'awakening':
                rest = 'waking up'
            else:
                # Add -ing if not present
                if not words[0].endswith('ing'):
                    rest = words[0] + 'ing ' + ' '.join(words[1:])
        return rest
    else:
        # Already in -ing form
        return label.lower()


def generate_i2v_caption(action_labels, scene=""):
    """Generate 50-100 char caption for I2V."""
    if not action_labels:
        return "Person doing daily activity indoors"

    # Pick the 2-3 most salient actions
    # Priority: avoid generic actions like walking, standing, sitting
    generic = {'walking through a doorway', 'standing up from somewhere',
               'going from standing to sitting', 'smiling', 'laughing'}
    specific_actions = [a for a in action_labels if a.lower() not in generic]
    if not specific_actions:
        specific_actions = action_labels

    # Naturalize and take top 2
    natural = [naturalize_label(a) for a in specific_actions[:2]]
    combined = " and ".join(natural)

    result = f"Person {combined}"
    if len(result) < 50:
        if scene:
            result += f" in {scene.lower()}"
        else:
            result += " indoors"
    return result[:100]


def generate_caption(action_labels, scene="", descriptions=""):
    """Generate 100-200 char natural language caption."""
    if not action_labels:
        return "A person is performing daily activities in an indoor setting."

    # If we have human-written descriptions, use the first one (most relevant)
    if descriptions:
        desc_list = descriptions.split(';')
        if desc_list and desc_list[0].strip():
            caption = desc_list[0].strip()
            if len(caption) >= 100 and len(caption) <= 200:
                return caption
            elif len(caption) > 200:
                # Truncate at sentence boundary
                end = caption.rfind('.', 0, 200)
                if end > 100:
                    return caption[:end+1]
                return caption[:200]
            # Otherwise use it as a base

    # Naturalize labels and pick meaningful ones
    natural = [naturalize_label(a) for a in action_labels]

    # Remove overly generic actions when we have many
    if len(natural) > 4:
        generic = {'walking through a doorway', 'standing up from somewhere',
                   'going from standing to sitting', 'smiling', 'laughing'}
        filtered = [a for a in natural if a not in generic]
        if filtered:
            natural = filtered

    if len(natural) == 0:
        return "A person is performing daily activities in an indoor setting."

    # Build the action description
    if len(natural) == 1:
        action_desc = natural[0]
    elif len(natural) == 2:
        action_desc = f"{natural[0]} and then {natural[1]}"
    elif len(natural) <= 4:
        action_desc = ", ".join(natural[:-1]) + f", and then {natural[-1]}"
    else:
        action_desc = ", ".join(natural[:3]) + f", among other activities"

    if scene:
        caption = f"In a {scene.lower()}, a person is {action_desc}."
    else:
        caption = f"A person is {action_desc} indoors."

    # Ensure 100-200 chars
    if len(caption) < 100 and scene:
        caption += f" The scene is recorded from an egocentric viewpoint."
    elif len(caption) < 100:
        caption += " The scene is recorded from an egocentric viewpoint."

    if len(caption) > 200:
        # Try a shorter version
        if len(natural) > 2:
            short_desc = f"{natural[0]} and {natural[1]}"
            if scene:
                caption = f"In a {scene.lower()}, a person is {short_desc}."
            else:
                caption = f"A person is {short_desc} indoors."

    return caption[:200]


def get_video_list(video_dir):
    """Get list of video files and extract video IDs."""
    videos = []
    for fname in sorted(os.listdir(video_dir)):
        if fname.endswith('.mp4'):
            # Format: {video_id}_1280x704_24fps.mp4
            match = re.match(r'^([A-Za-z0-9]+)_\d+x\d+_\d+fps\.mp4$', fname)
            if match:
                video_id = match.group(1)
                videos.append((fname, os.path.join(video_dir, fname), video_id))
    return videos


def main():
    print("=" * 60)
    print("Charades Caption Generator")
    print("=" * 60)

    # Load class map
    print("\n[1/4] Loading action class definitions...")
    class_map = load_class_map(CLASSES_FILE)

    # Load annotations
    print("\n[2/4] Loading video-to-action annotations...")
    vid_to_actions = load_annotations(TRAIN_CSV, TEST_CSV)
    vid_full = {}
    with open(TRAIN_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid_full[row['id']] = row
    with open(TEST_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid_full[row['id']] = row

    # Get video files
    print("\n[3/4] Scanning video files...")
    videos = get_video_list(VIDEO_DIR)
    print(f"Found {len(videos)} video files")

    # Match and generate captions
    print("\n[4/4] Generating captions...")
    entries = []
    matched = 0
    unmatched = 0
    unmatched_ids = []

    for fname, fpath, video_id in videos:
        if video_id in vid_to_actions:
            action_ids = vid_to_actions[video_id]
            action_labels = [class_map.get(aid, aid) for aid in action_ids]
            # Remove duplicates while preserving order
            seen = set()
            unique_labels = []
            for label in action_labels:
                if label not in seen:
                    seen.add(label)
                    unique_labels.append(label)
            action_labels = unique_labels

            full_info = vid_full.get(video_id, {})
            scene = full_info.get('scene', '')
            descriptions = full_info.get('descriptions', '')

            caption = generate_caption(action_labels, scene, descriptions)
            caption_i2v = generate_i2v_caption(action_labels, scene)

            entry = {
                "video": fname,
                "caption": caption,
                "caption_i2v": caption_i2v,
                "video_id": video_id,
                "action_classes": action_labels,
                "scene": scene,
            }
            entries.append(entry)
            matched += 1
        else:
            unmatched += 1
            unmatched_ids.append(video_id)
            entry = {
                "video": fname,
                "caption": "A person performing a daily activity indoors.",
                "caption_i2v": "Person doing daily activity",
                "video_id": video_id,
                "action_classes": ["unknown"],
                "scene": "",
            }
            entries.append(entry)

    # Write manifest
    print(f"\nWriting manifest to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')

    # Stats
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total videos: {len(videos)}")
    print(f"Matched with annotations: {matched}")
    print(f"Unmatched (fallback caption): {unmatched}")
    if unmatched_ids:
        print(f"Unmatched IDs: {unmatched_ids[:20]}{'...' if len(unmatched_ids) > 20 else ''}")

    # Sample captions
    print("\n--- Sample Captions (first 8) ---")
    for entry in entries[:8]:
        print(f"\n  Video: {entry['video']}")
        print(f"  Scene: {entry.get('scene', 'N/A')}")
        actions = entry.get('action_classes', [])
        if len(actions) > 4:
            print(f"  Actions ({len(actions)}): {actions[:4]}...")
        else:
            print(f"  Actions: {actions}")
        print(f"  Caption ({len(entry['caption'])}c): {entry['caption']}")
        print(f"  Caption I2V ({len(entry['caption_i2v'])}c): {entry['caption_i2v']}")

    # Action distribution
    from collections import Counter
    action_counts = Counter()
    for entry in entries:
        for action in entry.get('action_classes', []):
            if action != 'unknown':
                action_counts[action] += 1

    print(f"\n--- Top 20 Action Classes ---")
    for action, count in action_counts.most_common(20):
        print(f"  {action}: {count} videos")

    print(f"\n--- Total unique actions observed: {len(action_counts)} ---")

    # Check caption lengths
    lengths = [len(e['caption']) for e in entries]
    i2v_lengths = [len(e['caption_i2v']) for e in entries]
    print(f"\n--- Caption Length Stats ---")
    print(f"  Caption: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    print(f"  Caption I2V: min={min(i2v_lengths)}, max={max(i2v_lengths)}, avg={sum(i2v_lengths)/len(i2v_lengths):.1f}")

    # Count videos with descriptions
    with_descriptions = sum(1 for e in entries if e.get('action_classes') and e['action_classes'] != ['unknown'] and len(e.get('action_classes', [])) == 1)
    empty_actions = sum(1 for e in entries if not e.get('action_classes') or e['action_classes'] == ['unknown'] or len(e['action_classes']) == 0)
    print(f"  Videos with no action labels: {empty_actions}")

    print(f"\nOutput written to: {OUTPUT_FILE}")
    print("Done!")


if __name__ == '__main__':
    main()
