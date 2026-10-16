"""Manifest validation and leakage-safe deterministic dataset splitting."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


SPLITS = ("train", "dev", "holdout")
PathLike = Union[str, Path]


class ManifestError(ValueError):
    """The manifest is malformed or cannot be split without leakage."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ManifestError("{} must be a nonempty string".format(label))
    return value


def _resolved(path: str, root: Optional[PathLike]) -> Path:
    item = Path(path).expanduser()
    if not item.is_absolute() and root is not None:
        item = Path(root).expanduser() / item
    return item.resolve()


def validate_manifest(
    samples: Iterable[Mapping[str, Any]],
    *,
    root: Optional[PathLike] = None,
    check_paths: bool = False,
) -> Dict[str, Any]:
    """Validate records, raising ManifestError; return a compact inventory report.

    Relative media paths resolve against ``root`` (or the current directory).
    Existing split leakage is reported rather than rejected, so callers can
    repair an old split using ``split_manifest(..., respect_existing=False)``.
    Track 2 requires a nonempty timeline. Adjacent events are allowed, overlapping
    events are rejected; supplied event order need not already be chronological.
    """
    rows = list(samples)
    seen = set()
    for number, sample in enumerate(rows, 1):
        where = "record {}".format(number)
        if not isinstance(sample, Mapping):
            raise ManifestError("{} must be an object".format(where))
        sample_id = _text(sample.get("sample_id"), where + ".sample_id")
        if sample_id in (".", "..") or "/" in sample_id or "\\" in sample_id:
            raise ManifestError("{} has an unsafe sample_id".format(where))
        if sample_id in seen:
            raise ManifestError("duplicate sample_id: {}".format(sample_id))
        seen.add(sample_id)
        for key in ("identity_id", "source_video_id", "prompt"):
            _text(sample.get(key), "{}.{}".format(sample_id, key))
        track = sample.get("track")
        if type(track) is not int or track not in (1, 2):
            raise ManifestError("{}.track must be integer 1 or 2".format(sample_id))
        references = sample.get("reference_images")
        if not isinstance(references, list) or not references:
            raise ManifestError("{}.reference_images must be a nonempty list".format(sample_id))
        paths = [_text(path, sample_id + ".reference_images") for path in references]
        if "video_path" in sample:
            paths.append(_text(sample["video_path"], sample_id + ".video_path"))
        if check_paths:
            for path in paths:
                resolved = _resolved(path, root)
                if not resolved.is_file():
                    raise ManifestError("{} media file not found: {}".format(sample_id, resolved))
        if "split" in sample and sample["split"] not in SPLITS:
            raise ManifestError("{}.split must be train, dev, or holdout".format(sample_id))
        timeline = sample.get("timeline", [])
        if not isinstance(timeline, list) or (track == 2 and not timeline):
            raise ManifestError("{}.timeline must be a list, nonempty for track 2".format(sample_id))
        intervals = []
        for index, event in enumerate(timeline):
            label = "{}.timeline[{}]".format(sample_id, index)
            if not isinstance(event, Mapping):
                raise ManifestError(label + " must be an object")
            _text(event.get("action"), label + ".action")
            start, end = event.get("start_s"), event.get("end_s")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                   for x in (start, end)):
                raise ManifestError(label + " times must be finite numbers")
            if start < 0 or end <= start:
                raise ManifestError(label + " requires 0 <= start_s < end_s")
            intervals.append((start, end))
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1]:
                raise ManifestError("{}.timeline contains overlapping events".format(sample_id))
    return split_report(rows)


def read_manifest(
    path: PathLike, *, root: Optional[PathLike] = None, check_paths: bool = False
) -> List[Dict[str, Any]]:
    """Read UTF-8 JSONL, validate the whole manifest, and preserve extra fields."""
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ManifestError("{}:{}: invalid JSON".format(path, line_number)) from exc
            if not isinstance(item, dict):
                raise ManifestError("{}:{}: record must be an object".format(path, line_number))
            rows.append(item)
    validate_manifest(rows, root=root, check_paths=check_paths)
    return rows


def write_manifest(path: PathLike, samples: Iterable[Mapping[str, Any]]) -> None:
    """Validate and write deterministic UTF-8 JSONL; paths remain as supplied."""
    rows = list(samples)
    validate_manifest(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _components(samples: Sequence[Mapping[str, Any]]) -> List[List[int]]:
    parents = list(range(len(samples)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[b] = a

    owners = {}  # namespace identities and videos to avoid accidental collisions
    for index, row in enumerate(samples):
        for key in ("identity_id", "source_video_id"):
            node = (key, row[key])
            if node in owners:
                union(index, owners[node])
            else:
                owners[node] = index
    grouped = defaultdict(list)
    for index in range(len(samples)):
        grouped[find(index)].append(index)
    return list(grouped.values())


def split_report(samples: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize assigned splits and report identity/source leakage explicitly.

    Input must already conform to the schema; this function does not recursively
    call the validator. Counts refer to records, not generated candidate videos.
    """
    rows = list(samples)
    counts = Counter(row.get("split", "unassigned") for row in rows)
    identities, videos = defaultdict(set), defaultdict(set)
    for row in rows:
        if row.get("split") in SPLITS:
            identities[row["identity_id"]].add(row["split"])
            videos[row["source_video_id"]].add(row["split"])
    bad_id = sorted(key for key, value in identities.items() if len(value) > 1)
    bad_video = sorted(key for key, value in videos.items() if len(value) > 1)
    groups = _components(rows)
    return {
        "total_samples": len(rows),
        "counts": {name: counts[name] for name in SPLITS + ("unassigned",)},
        "identity_count": len({row["identity_id"] for row in rows}),
        "source_video_count": len({row["source_video_id"] for row in rows}),
        "connected_group_count": len(groups),
        "largest_group_samples": max((len(group) for group in groups), default=0),
        "tracks": {str(track): sum(row["track"] == track for row in rows) for track in (1, 2)},
        "leakage": {
            "safe": not bad_id and not bad_video,
            "identity_count": len(bad_id),
            "source_video_count": len(bad_video),
            "identity_examples": bad_id[:100],
            "source_video_examples": bad_video[:100],
        },
    }


def split_manifest(
    samples: Iterable[Mapping[str, Any]],
    *,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: Union[int, str] = 0,
    respect_existing: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Split complete identity/source connected components, never single rows.

    Assignment is stable across input permutations and Python hash randomization.
    Largest groups are assigned first to the greatest remaining sample deficit;
    ratios are targets, not a guarantee when connected groups are large. Existing
    assignments can be retained; contradictory assignments in one group raise.
    Rows are returned in their original order, without mutating input records.
    """
    rows = list(samples)
    validate_manifest(rows)
    if len(ratios) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                               or not math.isfinite(x) or x < 0 for x in ratios):
        raise ManifestError("ratios must be three finite nonnegative numbers")
    total_ratio = sum(ratios)
    if total_ratio <= 0:
        raise ManifestError("at least one split ratio must be positive")
    normalized = [float(value) / total_ratio for value in ratios]
    targets = {name: len(rows) * ratio for name, ratio in zip(SPLITS, normalized)}
    counts = {name: 0 for name in SPLITS}
    result = [dict(row) for row in rows]

    def key(group: List[int]) -> Tuple[int, str]:
        ids = sorted(rows[index]["sample_id"] for index in group)
        digest = hashlib.sha256(json.dumps([str(seed), ids], ensure_ascii=False).encode("utf-8")).hexdigest()
        return -len(group), digest

    groups = sorted(_components(rows), key=key)
    pending = []
    for group in groups:
        assigned = {rows[index]["split"] for index in group if "split" in rows[index]} if respect_existing else set()
        if len(assigned) > 1:
            raise ManifestError("existing split assignments conflict in connected identity/source group")
        if assigned:
            name = next(iter(assigned))
            counts[name] += len(group)
            for index in group:
                result[index]["split"] = name
        else:
            pending.append(group)
    allowed = [name for name, ratio in zip(SPLITS, normalized) if ratio > 0]
    for group in pending:
        # Choosing the largest absolute deficit minimizes the increase in total
        # squared error for this whole-group placement. Ties use split order.
        name = max(allowed, key=lambda candidate: targets[candidate] - counts[candidate])
        counts[name] += len(group)
        for index in group:
            result[index]["split"] = name
    report = split_report(result)
    report.update({"seed": seed, "target_ratios": dict(zip(SPLITS, normalized)),
                   "target_counts": targets, "respect_existing": respect_existing})
    report["warnings"] = ["{} is empty; ratios may be unattainable with connected groups".format(name)
                          for name in allowed if not counts[name]]
    return result, report
