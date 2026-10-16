"""Explicit, size-capped Hugging Face plans with opt-in downloads.

No import or planning operation performs a download. Authentication and gated
access remain the responsibility of the user's Hugging Face account.
"""

import fnmatch
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Union
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class ResourceError(ValueError):
    """A resource is unpinned, over budget, inaccessible, or inconsistent."""


def _repo(repo_id: str, repo_type: str) -> None:
    if (not isinstance(repo_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id)
            or any(part in (".", "..") for part in repo_id.split("/"))):
        raise ResourceError("repo_id must be an explicit owner/repository")
    if repo_type not in ("model", "dataset"):
        raise ResourceError("repo_type must be model or dataset")


def _filename(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ResourceError("invalid repository filename")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in value.split("/")) or ":" in value:
        raise ResourceError("unsafe repository filename: {}".format(value))
    return value


def fetch_hf_metadata(
    repo_id: str, *, revision: str, repo_type: str = "model", token: Optional[str] = None,
    timeout: float = 30,
) -> Dict[str, Any]:
    """Explicitly fetch metadata only; HTTP access errors never bypass a gate.

    A symbolic revision can be resolved here; pass the returned full ``sha`` to
    build_download_plan. Tokens are used in request headers and never returned.
    """
    _repo(repo_id, repo_type)
    if not isinstance(revision, str) or not revision.strip():
        raise ResourceError("an explicit revision is required")
    collection = "models" if repo_type == "model" else "datasets"
    url = "https://huggingface.co/api/{}/{}/revision/{}?blobs=true".format(
        collection, repo_id, quote(revision, safe=""))
    headers = {"User-Agent": "ICASSP-IPVG-resource-planner"}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise ResourceError("Hugging Face access denied; accept the repository terms and obtain account approval first") from exc
        raise ResourceError("Hugging Face metadata request failed with HTTP {}".format(exc.code)) from exc
    except (URLError, ValueError) as exc:
        raise ResourceError("could not retrieve valid Hugging Face metadata") from exc
    if not isinstance(result, dict):
        raise ResourceError("Hugging Face metadata must be an object")
    return result


def build_download_plan(
    metadata: Mapping[str, Any], *, repo_id: str, revision: str,
    allow_patterns: Sequence[str], max_bytes: int, repo_type: str = "model",
) -> Dict[str, Any]:
    """Select allowlisted files with known sizes at an immutable revision.

    The metadata must be a Hub ``?blobs=true`` model/dataset response with
    ``sha`` and ``siblings``. Symbolic revisions and unknown selected sizes are
    rejected. Exceeding the cap rejects the entire plan, never silently selects
    an arbitrary partial dataset. ``fnmatchcase`` matches repository paths.
    """
    _repo(repo_id, repo_type)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ResourceError("revision must be the full immutable 40-character commit SHA")
    if (not isinstance(metadata, Mapping) or not isinstance(metadata.get("sha"), str)
            or metadata["sha"].lower() != revision.lower()):
        raise ResourceError("metadata SHA does not match the requested revision")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ResourceError("max_bytes must be a positive integer")
    if isinstance(allow_patterns, str) or not allow_patterns or any(
        not isinstance(pattern, str) or not pattern for pattern in allow_patterns
    ):
        raise ResourceError("allow_patterns must be an explicit nonempty list")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise ResourceError("metadata must include siblings with file sizes")
    selected, seen = [], set()
    for item in siblings:
        if not isinstance(item, Mapping):
            raise ResourceError("invalid sibling metadata")
        name = _filename(item.get("rfilename"))
        if name in seen:
            raise ResourceError("duplicate repository filename: {}".format(name))
        seen.add(name)
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in allow_patterns):
            continue
        lfs = item.get("lfs") or {}
        size = item.get("size")
        if size is None and isinstance(lfs, Mapping):
            size = lfs.get("size")
        if type(size) is not int or size < 0:
            raise ResourceError("unknown or invalid size for selected file: {}".format(name))
        entry = {"path": name, "size_bytes": size}
        digest = lfs.get("sha256", lfs.get("oid")) if isinstance(lfs, Mapping) else None
        if isinstance(digest, str):
            digest = digest.removeprefix("sha256:")
            if re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                entry["sha256"] = digest.lower()
        selected.append(entry)
    selected.sort(key=lambda item: item["path"])
    if not selected:
        raise ResourceError("allow_patterns match no repository files")
    total = sum(item["size_bytes"] for item in selected)
    if total > max_bytes:
        raise ResourceError("selected files require {} bytes, exceeding cap {}".format(total, max_bytes))
    gated = metadata.get("gated", False)
    card = metadata.get("cardData") or {}
    return {"schema_version": 1, "repo_id": repo_id, "repo_type": repo_type,
            "revision": revision.lower(), "allow_patterns": list(allow_patterns),
            "max_bytes": max_bytes, "total_bytes": total, "file_count": len(selected),
            "files": selected, "gated": gated,
            "requires_access_approval": bool(gated),
            "license": card.get("license") if isinstance(card, Mapping) else None,
            "notice": "A plan grants no access or license; gated terms and account approval must be satisfied before download."}


def _checked_plan(plan: Mapping[str, Any]) -> None:
    """Revalidate a saved plan before it can cause writes or network calls."""
    try:
        siblings = [{"rfilename": item["path"], "size": item["size_bytes"]}
                    for item in plan["files"]]
        rebuilt = build_download_plan(
            {"sha": plan["revision"], "siblings": siblings},
            repo_id=plan["repo_id"], repo_type=plan["repo_type"], revision=plan["revision"],
            allow_patterns=plan["allow_patterns"], max_bytes=plan["max_bytes"],
        )
        if (rebuilt["total_bytes"] != plan["total_bytes"] or rebuilt["file_count"] != plan["file_count"]
                or rebuilt["file_count"] != len(plan["files"])):
            raise ResourceError("saved plan totals do not match its files")
        for item in plan["files"]:
            if "sha256" in item and (not isinstance(item["sha256"], str)
                                      or not re.fullmatch(r"[0-9a-fA-F]{64}", item["sha256"])):
                raise ResourceError("invalid saved file checksum")
    except (KeyError, TypeError) as exc:
        raise ResourceError("invalid saved download plan") from exc


def download_plan(
    plan: Mapping[str, Any], target_dir: Union[str, Path], *, execute: bool = False,
    token: Optional[str] = None, access_approved: bool = False,
) -> Dict[str, Any]:
    """Default dry run; execute downloads only listed files, never a snapshot.

    ``access_approved`` records the caller's confirmation for a gated resource;
    the Hub still independently enforces actual account approval. Optional
    huggingface_hub is imported only after an explicit execution request.
    """
    _checked_plan(plan)
    target = Path(target_dir).expanduser().resolve()
    if not execute:
        return {"status": "planned", "target_dir": str(target), "plan": dict(plan)}
    if (plan.get("requires_access_approval") or plan.get("gated")) and not access_approved:
        raise ResourceError("gated resource requires accepted terms and account approval; no download attempted")
    # Recheck sizes at the pinned commit before a saved/edited plan causes any
    # large downloads. A forged small size must not evade the download cap.
    metadata = fetch_hf_metadata(plan["repo_id"], revision=plan["revision"],
                                 repo_type=plan["repo_type"], token=token)
    current = build_download_plan(metadata, repo_id=plan["repo_id"], revision=plan["revision"],
                                  repo_type=plan["repo_type"], allow_patterns=plan["allow_patterns"],
                                  max_bytes=plan["max_bytes"])
    expected = [(item["path"], item["size_bytes"]) for item in plan["files"]]
    actual = [(item["path"], item["size_bytes"]) for item in current["files"]]
    if sorted(expected) != actual:
        raise ResourceError("saved plan differs from metadata at the pinned revision")
    if current["requires_access_approval"] and not access_approved:
        raise ResourceError("gated resource requires accepted terms and account approval")
    for item in plan["files"]:
        destination = (target / item["path"]).resolve()
        try:
            destination.relative_to(target)
        except ValueError as exc:
            raise ResourceError("download destination escapes target_dir") from exc
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ResourceError("install optional huggingface_hub to execute downloads") from exc
    target.mkdir(parents=True, exist_ok=True)
    completed = []
    for item in plan["files"]:
        hf_hub_download(repo_id=plan["repo_id"], repo_type=plan["repo_type"],
                        revision=plan["revision"], filename=item["path"],
                        local_dir=str(target), token=token)
        destination = target / item["path"]
        if not destination.is_file() or destination.stat().st_size != item["size_bytes"]:
            raise ResourceError("downloaded size does not match plan: {}".format(item["path"]))
        if item.get("sha256"):
            digest = hashlib.sha256()
            with destination.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise ResourceError("downloaded checksum mismatch: {}".format(item["path"]))
        completed.append(item["path"])
    return {"status": "downloaded", "target_dir": str(target),
            "file_count": len(completed), "total_bytes": plan["total_bytes"], "files": completed}
