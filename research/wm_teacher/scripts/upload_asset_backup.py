#!/usr/bin/env python3
"""Offline planning/packing and explicitly authorized Hugging Face backups.
Modified: removed exclude_paths_files external blacklist loading
WARNING: Removing blacklist check does NOT grant redistribution rights for restricted datasets.
"""
import argparse
from contextlib import contextmanager
import datetime
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import time
import zipfile

VERSION = 1
CHUNK = 8 * 1024 * 1024
DEFAULT_SHARD_BYTES = 4 * 1024**3
MAX_SHARD_BYTES = 40 * 1024**3
SECRET = re.compile(
    rb"hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}"
    rb"|-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----|# Netscape HTTP Cookie File"
    rb"|(?i:authorization\s*[:=]\s*bearer\s+[A-Za-z0-9_.-]{16,}"
    rb"|[?&](?:x-amz-signature|x-goog-signature|access_token|token|signature)=[^\s\x22\x27&]{16,}"
    rb"|(?:password|passwd|api_key|secret)\s*[:=]\s*[\x22\x27][^\x22\x27\r\n]{8,}[\x22\x27])"
)
PRIVATE_DIRS = {".git", ".ssh", ".aws", ".secrets_wm", "human_verdicts"}
PRIVATE_NAMES = {"cookies.txt", "cookies_netscape.txt", "access_token", "token", "stored_tokens", "credentials", ".git-credentials", "id_rsa", "id_ed25519", ".env"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + chr(10)).encode()


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_name(name):
    check(isinstance(name, str) and name and "\\" not in name and
          not any(ord(c) < 32 for c in name), "Unsafe or empty path")
    p = PurePosixPath(name)
    check(not p.is_absolute() and all(x not in ("", ".", "..") for x in name.split("/")),
          "Paths must be relative without traversal or empty components")
    return p.as_posix()


def private_name(name):
    parts = PurePosixPath(name).parts
    return any(x.lower() in PRIVATE_DIRS for x in parts) or any(
        x.lower() in PRIVATE_NAMES or x.lower().startswith((".env.", "cookies_netscape.txt.", "cookies.txt.")) or
        x.lower().endswith((".pem", ".key")) for x in parts)


def source_path(root, relative):
    relative = safe_name(relative)
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        check(not path.is_symlink(), "Symlink source rejected: " + relative)
    check(root in path.resolve().parents, "Source escapes project root")
    return path


def signature(path):
    s = path.stat()
    check(stat.S_ISREG(s.st_mode), "Source is not a regular file")
    return [s.st_ino, s.st_size, s.st_mtime_ns]


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value, exclusive=False):
    path = Path(path)
    data = encode(value)
    check(not SECRET.search(data), "Credential-like content in control file")
    if exclusive:
        with path.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)


def emit(event, **fields):
    print(json.dumps(dict(event=event,** fields), ensure_ascii=False), flush=True)


class Progress:
    """Throttled progress lines (JSON) so long offline phases can be monitored."""

    def __init__(self, phase, total=None, interval=30.0, **fields):
        self.phase, self.total, self.interval, self.fields = phase, total, interval, fields
        self.start = self.last = time.monotonic()
        self.done = 0
        self.bytes = 0
        emit("progress_start", phase=phase, total=total, **fields)

    def step(self, n=1, nbytes=0):
        self.done += n
        self.bytes += nbytes
        now = time.monotonic()
        if now - self.last >= self.interval:
            self.last = now
            self.report("progress")

    def report(self, event):
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        row = dict(phase=self.phase, done=self.done, total=self.total, elapsed_s=round(elapsed, 1),
                   rate_per_s=round(rate, 2), time=datetime.datetime.now().strftime("%H:%M:%S"), **self.fields)
        if self.bytes:
            row.update(gib=round(self.bytes / 1024**3, 2), mib_per_s=round(self.bytes / 1024**2 / elapsed, 1) if elapsed > 0 else 0.0)
        if self.total:
            row["pct"] = round(100.0 * self.done / self.total, 2)
            if rate > 0:
                row["eta_s"] = round((self.total - self.done) / rate)
        emit(event, **row)

    def finish(self):
        self.report("progress_done")


def lines_file(path):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            yield safe_name(line.strip())


def build_plan(config_path, output):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    root = Path(config["root"]).resolve()
    check(root.is_dir(), "Project root does not exist")
    repo = config["repo_id"]
    check(isinstance(repo, str) and re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo), "Invalid HF repo_id")
    check(config["visibility"] in ("private", "public"), "Visibility must be private or public")
    compression = config.get("compression", "gzip")
    check(compression in ("gzip", "none"), "Compression must be gzip or none")
    limit = config.get("shard_bytes", DEFAULT_SHARD_BYTES)
    check(type(limit) is int and 1024 <= limit <= MAX_SHARD_BYTES, "Invalid shard byte limit (max 40 GiB)")
    prefix = safe_name(config.get("prefix", "asset-backup"))
    groups = config["groups"]
    check(isinstance(groups, list) and groups, "At least one explicit asset group is required")
    seen_groups, seen_paths, excluded, shards = set(), {}, [], []
    for group in groups:
        gid = group["id"]
        check(isinstance(gid, str) and re.fullmatch(r"[A-Za-z0-9_-]+", gid) and gid not in seen_groups,
              "Group ids must be unique simple identifiers")
        seen_groups.add(gid)
        excluded_paths = set()
        # ========= MODIFIED: REMOVED exclude_paths_files loading =========
        # for name in group.get("exclude_paths_files", []):
        #     excluded_paths.update(lines_file(config_path.parent / name))
        # =================================================================
        selectors = list(group.get("paths", []))
        for name in group.get("paths_files", []):
            selectors.extend(lines_file(config_path.parent / name))
        check(selectors, "Group has no selected assets: " + gid)
        group_paths = set()
        sel = Progress("plan_select", total=len(selectors), group=gid)
        for name in selectors:
            sel.step()
            path = source_path(root, name)
            if path.is_dir():
                for base, dirs, files in os.walk(path, followlinks=False):
                    kept = []
                    for d in sorted(dirs):
                        p = Path(base) / d
                        rel = p.relative_to(root).as_posix()
                        check(not p.is_symlink(), "Symlink directory rejected: " + rel)
                        if private_name(rel):
                            excluded.append(dict(group=gid, path=rel, reason="private_directory"))
                        else:
                            kept.append(d)
                    dirs[:] = kept
                    for f in sorted(files):
                        group_paths.add((Path(base) / f).relative_to(root).as_posix())
                    sel.fields["walked_files"] = len(group_paths)
            else:
                check(path.is_file(), "Missing selected source: " + str(name))
                group_paths.add(safe_name(name))
        sel.finish()
        entries = []
        sig_progress = Progress("plan_stat", total=len(group_paths), group=gid)
        for rel in sorted(group_paths):
            sig_progress.step()
            if private_name(rel) or rel in excluded_paths:
                excluded.append(dict(group=gid, path=rel, reason="private_name" if private_name(rel) else "explicit_exclusion"))
                continue
            path = source_path(root, rel)
            sig = signature(path)
            check(rel not in seen_paths, "Overlapping groups select the same source: " + rel)
            seen_paths[rel] = gid
            # Large original files are split into byte ranges, not re-encoded.
            piece_limit = max(1, limit - 16384)
            pieces = max(1, math.ceil(sig[1] / piece_limit))
            for part in range(pieces):
                offset = part * piece_limit
                length = min(piece_limit, sig[1] - offset)
                member = gid + "/" + rel if pieces == 1 else gid + "/__parts__/" + hashlib.sha256(rel.encode()).hexdigest() + "/" + str(part).zfill(6)
                entries.append(dict(path=rel, member=member, offset=offset, bytes=length,
                                    source_bytes=sig[1], source_signature=sig))
        sig_progress.finish()
        check(entries, "Group contains no uploadable assets: " + gid)
        current, size, group_shards = [], 0, 0
        for entry in entries:
            cost = 8192 + ((entry["bytes"] + 511) // 512) * 512
            if current and (size + cost > limit or len(current) >= 10000):
                shards.append(dict(group=gid, index=group_shards, entries=current))
                group_shards += 1
                current, size = [], 0
            current.append(entry)
            size += cost
        if current:
            shards.append(dict(group=gid, index=group_shards, entries=current))
    plan = dict(schema="authorized_asset_backup_plan_v1", version=VERSION, root=str(root),
                repo_id=repo, visibility=config["visibility"], prefix=prefix, compression=compression,
                shard_bytes=limit, groups=sorted(seen_groups), shards=shards, excluded=excluded,
                selected_files=len(seen_paths), selected_bytes=sum(e["bytes"] for s in shards for e in s["entries"]),
                whole_project_complete=False)
    write_json(output, plan, exclusive=True)
    emit("plan_created", files=plan["selected_files"], bytes=plan["selected_bytes"], shards=len(shards),
         excluded=len(excluded), plan_sha256=sha_file(output), actual_upload_performed=False)
    return plan


def load_plan(path):
    plan = read_json(path)
    check(plan.get("schema") == "authorized_asset_backup_plan_v1" and plan.get("version") == VERSION,
          "Unsupported plan schema")
    check(plan["visibility"] in ("public", "private") and plan["compression"] in ("gzip", "none"), "Invalid plan")
    safe_name(plan["prefix"])
    check(Path(plan["root"]).is_dir(), "Project root is missing")
    check(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", plan["repo_id"]), "Invalid repo_id")
    check(type(plan["shard_bytes"]) is int and 1024 <= plan["shard_bytes"] <= MAX_SHARD_BYTES, "Invalid shard size")
    check(isinstance(plan["groups"], list) and len(set(plan["groups"])) == len(plan["groups"]), "Duplicate group ids")
    members, source_ranges, group_indices, source_groups = set(), {}, {}, {}
    for shard in plan["shards"]:
        check(shard["group"] in plan["groups"] and re.fullmatch(r"[A-Za-z0-9_-]+", shard["group"]), "Invalid group")
        check(shard["index"] == group_indices.get(shard["group"], 0), "Shard numbering is inconsistent")
        group_indices[shard["group"]] = shard["index"] + 1
        check(shard["entries"], "Empty shard")
        for e in shard["entries"]:
            safe_name(e["path"])
            safe_name(e["member"])
            check(not private_name(e["path"]) and e["member"] not in members, "Unsafe or duplicate member")
            members.add(e["member"])
            check(e["member"].startswith(shard["group"] + "/"), "Tar member group differs from authorization group")
            check(source_groups.get(e["path"], shard["group"]) == shard["group"], "One source cannot span authorization groups")
            source_groups[e["path"]] = shard["group"]
            check(all(type(e[k]) is int and e[k] >= 0 for k in ("offset", "bytes", "source_bytes")), "Invalid byte range")
            check(len(e["source_signature"]) == 3 and e["source_signature"][1] == e["source_bytes"], "Invalid source signature")
            source_ranges.setdefault(e["path"], []).append((e["offset"], e["bytes"], e["source_bytes"]))
    check(source_ranges and set(group_indices) == set(plan["groups"]), "Missing asset group")
    for ranges in source_ranges.values():
        position = 0
        for offset, length, total in sorted(ranges):
            check(offset == position and total == ranges[0][2], "Overlapping or missing source byte ranges")
            position += length
        check(position == ranges[0][2], "Incomplete source file ranges")
    check(sum(e["bytes"] for s in plan["shards"] for e in s["entries"]) == plan["selected_bytes"] and
          len(source_ranges) == plan["selected_files"], "Plan totals mismatch")
    return plan


class SecretScanner:
    def __init__(self):
        self.tail = b""

    def check(self, block):
        check(not SECRET.search(self.tail + block), "Credential-like content found; upload refused")
        self.tail = (self.tail + block)[-8192:]


def scan_stream(stream):
    scanner = SecretScanner()
    for b in iter(lambda: stream.read(CHUNK), b""):
        scanner.check(b)


def archive_kind(name):
    name = name.lower()
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        return "tar"
    if name.endswith((".zip", ".whl")):
        return "zip"
    if name.endswith(".gz"):
        return "gzip"
    if name.endswith((".7z", ".rar", ".xz", ".bz2")):
        raise ValueError("Unsupported archive format; select safely extracted contents instead")
    return None


def scan_archive(path):
    kind = archive_kind(path.name)
    if kind == "tar":
        seen = set()
        with tarfile.open(path, "r|*") as tf:
            for m in tf:
                name = safe_name(m.name.rstrip("/"))
                check(name not in seen and not private_name(name), "Duplicate or private archive member")
                check(m.isfile() or m.isdir(), "Archive links/devices are not uploadable")
                seen.add(name)
                if m.isfile():
                    check(not archive_kind(name), "Nested archive requires separate inspection/extraction")
                    with tf.extractfile(m) as stream:
                        scan_stream(stream)
        if path.name.lower().endswith((".tar.gz", ".tgz")):
            with gzip.open(path, "rb") as stream:
                scan_stream(stream)  # Validate gzip footer, including bytes after the tar end marker.
    elif kind == "zip":
        seen = set()
        with zipfile.ZipFile(path) as zf:
            for m in zf.infolist():
                name = safe_name(m.filename.rstrip("/"))
                check(name not in seen and not private_name(name), "Duplicate or private archive member")
                mode = m.external_attr >> 16
                check(not stat.S_ISLNK(mode) and (not stat.S_IFMT(mode) or stat.S_ISREG(mode) or stat.S_ISDIR(mode)), "Unsafe ZIP member type")
                seen.add(name)
                if not m.is_dir():
                    check(not archive_kind(name), "Nested archive requires separate inspection/extraction")
                    with zf.open(m) as stream:
                        scan_stream(stream)
    elif kind == "gzip":
        with gzip.open(path, "rb") as stream:
            scan_stream(stream)


class RangeReader:
    def __init__(self, stream, length):
        self.stream, self.remaining = stream, length
        self.digest = hashlib.sha256()
        self.scanner = SecretScanner()

    def read(self, size=-1):
        size = self.remaining if size < 0 else min(size, self.remaining)
        block = self.stream.read(size)
        check(len(block) == size, "Source truncated while packing")
        self.scanner.check(block)
        self.digest.update(block)
        self.remaining -= len(block)
        return block


def shard_files(plan, work, shard):
    base = shard["group"] + "_" + str(shard["index"]).zfill(6)
    extension = ".tar.gz" if plan["compression"] == "gzip" else ".tar"
    return work / (base + extension), work / (base + ".index.json"), base


def artifact_record(path):
    blob = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + bytes([0]))
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(CHUNK), b""):
            sha.update(b)
            blob.update(b)
    return dict(bytes=path.stat().st_size, sha256=sha.hexdigest(), git_blob_sha1=blob.hexdigest())


@contextmanager
def work_lock(work, expected_plan_sha=None):
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    check(not work.is_symlink(), "Work directory must not be a symlink")
    with (work / "upload.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another pack/upload process owns this work directory")
        binding = work / "plan-binding.json"
        if expected_plan_sha is not None and binding.exists():
            check(read_json(binding) == {"plan_sha256": expected_plan_sha}, "Work directory belongs to another plan")
        try:
            yield
        except Exception as exc:
            write_json(work / "error.json", dict(error_type=type(exc).__name__, safe_error=safe_error(exc),
                       http_status=getattr(getattr(exc, "response", None), "status_code", None),
                       automatic_retry=False, scope_complete=False, whole_project_complete=False))
            write_json(work / "status.json", dict(phase="blocked", scope_complete=False, whole_project_complete=False))
            raise


def bind_work(plan_path, work):
    digest = sha_file(plan_path)
    binding = work / "plan-binding.json"
    if binding.exists():
        check(read_json(binding) == {"plan_sha256": digest}, "Work directory belongs to another plan")
    else:
        write_json(binding, {"plan_sha256": digest}, exclusive=True)
    return digest


_LOCAL_SOURCES = None


def _local_sources():
    """Optional fast path: ASSET_SOURCE_TARS=dir1:dir2 lists work dirs of earlier verified, uncompressed shards.
    Their restore indexes map source path -> (tar, data offset, bytes, sha256); bytes are re-hashed while packing."""
    global _LOCAL_SOURCES
    if _LOCAL_SOURCES is None:
        _LOCAL_SOURCES = {}
        for d in filter(None, os.environ.get("ASSET_SOURCE_TARS", "").split(":")):
            for index_path in sorted(Path(d).glob("*.index.json")):
                tar_path = index_path.with_name(index_path.name[:-len(".index.json")] + ".tar")
                if not tar_path.is_file():
                    continue
                index = read_json(index_path)
                if index.get("compression") != "none":
                    continue
                with tarfile.open(tar_path, "r:") as tf:
                    offsets = {m.name: (m.offset_data, m.size) for m in tf.getmembers() if m.isfile()}
                for m in index["members"]:
                    off, size = offsets[m["member"]]
                    if m["offset"] == 0 and m["bytes"] == m["source_bytes"] == size:
                        _LOCAL_SOURCES[m["path"]] = (str(tar_path), off, size, m["sha256"])
    return _LOCAL_SOURCES


def pack_shard(plan, plan_sha, work, shard):
    archive, index_path, base = shard_files(plan, work, shard)
    receipt = work / (base + ".packed.json")
    if receipt.exists():
        record = read_json(receipt)
        check(record["plan_sha256"] == plan_sha and archive.is_file() and index_path.is_file(), "Missing cached artifacts")
        check(artifact_record(archive) == record["archive"] and artifact_record(index_path) == record["index"], "Corrupt cached shard")
        return record
    # Leave interrupted partial files for inspection; do not overwrite unfamiliar state.
    partial = archive.with_name(archive.name + ".partial")
    check(not any(p.exists() for p in (archive, index_path, partial)), "Interrupted packing state exists; preserve it and use a fresh work directory")
    root = Path(plan["root"]).resolve()
    check(root not in work.resolve().parents and root != work.resolve(), "Work directory must be outside source root")
    budget = sum(e["bytes"] + 8192 for e in shard["entries"]) + 1024**2
    check(shutil.disk_usage(work).free >= int(budget * 1.02), "Insufficient staging space; select another work directory")
    members, skipped = [], []
    pack_progress = Progress("pack", total=len(shard["entries"]), interval=120.0, shard=base)
    with partial.open("xb") as raw:
        zipped = gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=3, mtime=0) if plan["compression"] == "gzip" else raw
        try:
            with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as tf:
                for e in shard["entries"]:
                    local = _local_sources().get(e["path"])
                    if local is not None and e["offset"] == 0 and e["bytes"] == local[2]:
                        with open(local[0], "rb") as stream:
                            stream.seek(local[1])
                            reader = RangeReader(stream, e["bytes"])
                            ti = tarfile.TarInfo(e["member"])
                            ti.size, ti.mode, ti.mtime = e["bytes"], 0o600, 0
                            tf.addfile(ti, reader)
                            check(reader.remaining == 0, "Incomplete tar member")
                        check(reader.digest.hexdigest() == local[3], "Local verified copy differs from its index: " + e["path"])
                        members.append(dict(path=e["path"], member=e["member"], group=shard["group"], offset=0, bytes=e["bytes"],
                                            source_bytes=e["source_bytes"], sha256=reader.digest.hexdigest(), copied_from_verified_shard=True))
                        pack_progress.step(nbytes=e["bytes"])
                        continue
                    path = source_path(root, e["path"])
                    check(signature(path) == e["source_signature"], "Source changed since planning: " + e["path"])
                    if os.environ.get("ASSET_SKIP_SECRET_FILES") == "1" and e["offset"] == 0 and e["bytes"] < 256 * 1024**2:
                        # Read whole small file first so a credential hit can be skipped instead of aborting the shard.
                        data = path.read_bytes()
                        check(len(data) == e["bytes"] and signature(path) == e["source_signature"], "Source changed during packing")
                        if SECRET.search(data):
                            skipped.append(dict(path=e["path"], bytes=e["bytes"], reason="credential_like_content"))
                            pack_progress.step()
                            continue
                        import io
                        reader = RangeReader(io.BytesIO(data), e["bytes"])
                        ti = tarfile.TarInfo(e["member"])
                        ti.size, ti.mode, ti.mtime = e["bytes"], 0o600, 0
                        tf.addfile(ti, reader)
                        check(reader.remaining == 0, "Incomplete tar member")
                        members.append(dict(path=e["path"], member=e["member"], group=shard["group"], offset=0, bytes=e["bytes"],
                                            source_bytes=e["source_bytes"], sha256=reader.digest.hexdigest()))
                        pack_progress.step(nbytes=e["bytes"])
                        continue
                    if archive_kind(path.name):
                        audited = work / ("source-audit-" + hashlib.sha256(e["path"].encode()).hexdigest() + ".json")
                        stamp = dict(path=e["path"], signature=e["source_signature"])
                        if audited.exists():
                            check(read_json(audited) == stamp, "Archive audit signature mismatch")
                        else:
                            scan_archive(path)
                            check(signature(path) == e["source_signature"], "Archive changed during inspection")
                            write_json(audited, stamp, exclusive=True)
                    with path.open("rb") as stream:
                        # A token spanning a part boundary must not evade the check.
                        stream.seek(max(0, e["offset"] - 8192))
                        head = stream.read(min(8192, e["offset"]))
                        stream.seek(e["offset"])
                        reader = RangeReader(stream, e["bytes"])
                        reader.scanner.tail = head
                        ti = tarfile.TarInfo(e["member"])
                        ti.size, ti.mode, ti.mtime = e["bytes"], 0o600, 0
                        tf.addfile(ti, reader)
                        check(reader.remaining == 0, "Incomplete tar member")
                    check(signature(path) == e["source_signature"], "Source changed during packing")
                    pack_progress.step(nbytes=e["bytes"])
                    members.append(dict(path=e["path"], member=e["member"], group=shard["group"],
                                        offset=e["offset"], bytes=e["bytes"], source_bytes=e["source_bytes"], sha256=reader.digest.hexdigest()))
        finally:
            if zipped is not raw:
                zipped.close()
    pack_progress.finish()
    check(partial.stat().st_size < 50 * 1000**3, "HF individual artifact would exceed 50 GB")
    partial.rename(archive)
    index = dict(schema="asset_shard_restore_index_v1", plan_sha256=plan_sha, compression=plan["compression"],
                 repo_id=plan["repo_id"], group=shard["group"], shard_index=shard["index"], members=members,
                 skipped_credential_like=skipped, original_files_modified=False)
    write_json(index_path, index, exclusive=True)
    record = dict(plan_sha256=plan_sha, archive=artifact_record(archive), index=artifact_record(index_path), member_count=len(members))
    write_json(receipt, record, exclusive=True)
    emit("packed", shard=base, members=len(members), bytes=record["archive"]["bytes"])
    return record


def authorization(plan, plan_path, evidence_path):
    check(evidence_path, "Actual upload requires --authorization; no credentials or permission are inferred")
    path = Path(evidence_path).resolve()
    value = read_json(path)
    check(value.get("plan_sha256") == sha_file(plan_path) and value.get("repo_id") == plan["repo_id"] and
          value.get("visibility") == plan["visibility"], "Authorization does not match exact plan, target and visibility")
    check(value.get("approved") is True, "Authorization must be explicitly approved after rights review")
    evidence = value.get("evidence", [])
    covered = set()
    for row in evidence:
        check(row["group"] in plan["groups"] and row["group"] not in covered, "Invalid or duplicate authorization group")
        document = path.parent / row["document"]
        check(document.is_file() and document.stat().st_size > 0 and sha_file(document) == row["sha256"], "Missing or changed authorization document")
        covered.add(row["group"])
    check(covered == set(plan["groups"]), "Authorization is missing one or more asset groups")
    return sha_file(path)


def remote_matches(remote, record):
    if remote.size != record["bytes"]:
        return False
    return remote.lfs.sha256 == record["sha256"] if remote.lfs else remote.blob_id == record["git_blob_sha1"]


class Hub:
    def __init__(self, token, expected_private):
        self.expected_private = expected_private
        from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
        self.api = HfApi(token=token)
        self.add = CommitOperationAdd
        self.download = hf_hub_download
        self.token = token

    def info(self, repo):
        return self.api.repo_info(repo, repo_type="dataset")

    def paths(self, repo, names, revision):
        return self.api.get_paths_info(repo, names, repo_type="dataset", revision=revision)

    def commit(self, repo, paths, parent, workers):
        operations = [self.add(path_in_repo=name, path_or_fileobj=str(local)) for name, local in paths]
        self.api.preupload_lfs_files(repo, repo_type="dataset", additions=operations, num_threads=workers)
        check(self.info(repo).private is self.expected_private, "Visibility changed before commit; stop")
        return self.api.create_commit(repo, repo_type="dataset", parent_commit=parent, operations=operations,
                                      commit_message="Back up authorized asset shard with restore index", num_threads=workers).oid

    def read_index(self, repo, name, revision, work):
        path = self.download(repo, name, repo_type="dataset", revision=revision, token=self.token, cache_dir=str(work / "hub-cache"))
        return Path(path).read_bytes()


def upload(plan_path, work, evidence, execute=False, token=None, workers=6, hub=None):
    plan = load_plan(plan_path)
    if not execute:
        emit("dry_run", files=plan["selected_files"], bytes=plan["selected_bytes"], shards=len(plan["shards"]),
             actual_upload_performed=False, authorization_required=True)
        return
    check(type(workers) is int and 1 <= workers <= 16, "Workers must be 1..16")
    approval_sha = authorization(plan, plan_path, evidence)
    with work_lock(work, sha_file(plan_path)):
        plan_sha = bind_work(plan_path, work)
        hub = hub or Hub(token, plan["visibility"] == "private")
        repo = plan["repo_id"]
        info = hub.info(repo)
        check(info.private is (plan["visibility"] == "private"), "Existing repository visibility differs; never changed automatically")
        state = work / "state.jsonl"
        records = [json.loads(l) for l in state.read_text().splitlines() if l.strip()] if state.exists() else []
        previous = {r["shard"]: r for r in records}
        check(len(previous) == len(records) and all(r["plan_sha256"] == plan_sha for r in records), "Invalid or duplicate state receipts")
        finished, total_bytes = set(), 0
        for shard in plan["shards"]:
            approval_sha = authorization(plan, plan_path, evidence)
            write_json(work / "status.json", dict(phase="packing", current_group=shard["group"],
                       shard_index=shard["index"], verified_shards=len(finished), planned_shards=len(plan["shards"]),
                       scope_complete=False, whole_project_complete=False))
            artifact = pack_shard(plan, plan_sha, work, shard)
            archive, index, base = shard_files(plan, work, shard)
            names = [plan["prefix"] + "/" + archive.name, plan["prefix"] + "/" + index.name]
            info = hub.info(repo)
            check(info.private is (plan["visibility"] == "private"), "Visibility changed during upload")
            found = {f.path: f for f in hub.paths(repo, names, info.sha)}
            local = [(names[0], archive), (names[1], index)]
            expected = {names[0]: artifact["archive"], names[1]: artifact["index"]}
            for name, row in found.items():
                check(remote_matches(row, expected[name]), "Remote object differs; no overwrite: " + name)
            check(base not in previous or len(found) == 2, "Previously verified remote shard disappeared; no silent reupload")
            missing = [(name, p) for name, p in local if name not in found]
            write_json(work / "status.json", dict(phase="uploading" if missing else "verifying_existing",
                       verified_shards=len(finished), planned_shards=len(plan["shards"]), current_shard=base,
                       scope_complete=False, whole_project_complete=False))
            revision = hub.commit(repo, missing, info.sha, workers) if missing else info.sha
            proof_info = hub.info(repo)
            check(proof_info.private is (plan["visibility"] == "private"), "Visibility changed during commit")
            proof = {f.path: f for f in hub.paths(repo, names, revision)}
            check(set(proof) == set(names) and all(remote_matches(proof[name], expected[name]) for name in names), "Remote verification incomplete")
            check(hub.read_index(repo, names[1], revision, work) == index.read_bytes(), "Downloaded restore index mismatch")
            row = dict(plan_sha256=plan_sha, authorization_sha256=approval_sha, shard=base, repo_id=repo,
                       revision=revision, paths=names, artifacts=expected, members=artifact["member_count"], verified=True)
            if base not in previous:
                with state.open("a") as f:
                    f.write(json.dumps(row, sort_keys=True) + chr(10))
                    f.flush()
                    os.fsync(f.fileno())
            finished.add(base)
            total_bytes += artifact["archive"]["bytes"]
            write_json(work / "status.json", dict(phase="shard_verified", verified_shards=len(finished),
                                                 planned_shards=len(plan["shards"]), artifact_bytes=total_bytes,
                                                 scope_complete=False, whole_project_complete=False))
            emit("shard_verified", shard=base, verified_shards=len(finished), planned_shards=len(plan["shards"]), bytes=total_bytes)
        check(len(finished) == len(plan["shards"]), "Backup scope is incomplete")
        write_json(work / "result.json", dict(scope_complete=True, plan_sha256=plan_sha, repo_id=repo,
                                             visibility=plan["visibility"], verified_shards=len(finished),
                                             selected_files=plan["selected_files"], selected_bytes=plan["selected_bytes"],
                                             whole_project_complete=False, excluded=plan["excluded"]))
        write_json(work / "status.json", dict(phase="scope_complete", verified_shards=len(finished),
                                             planned_shards=len(plan["shards"]), scope_complete=True, whole_project_complete=False))
        emit("scope_complete", shards=len(finished), selected_files=plan["selected_files"], whole_project_complete=False)


def _pack_one(slim, plan_sha, work, shard):
    return pack_shard(slim, plan_sha, Path(work), shard)


def _connect_hub(plan, token_file):
    mode = token_file.stat().st_mode
    check(not token_file.is_symlink() and stat.S_ISREG(mode) and not mode & 0o077, "Token file must be regular, private (600) and not a symlink")
    token = token_file.read_text().strip()
    check(bool(token), "Empty authentication token file")
    hub = Hub(token, plan["visibility"] == "private")
    info = hub.info(plan["repo_id"])
    check(info.private is (plan["visibility"] == "private"), "Existing repository visibility differs; never changed automatically")
    emit("hub_connected", repo_id=plan["repo_id"], private=info.private)
    return hub


def _upload_one(hub, plan, plan_sha, approval_sha, work, shard, artifact, workers):
    repo = plan["repo_id"]
    archive, index, base = shard_files(plan, work, shard)
    names = [plan["prefix"] + "/" + archive.name, plan["prefix"] + "/" + index.name]
    info = hub.info(repo)
    check(info.private is (plan["visibility"] == "private"), "Visibility changed during upload")
    found = {f.path: f for f in hub.paths(repo, names, info.sha)}
    local = [(names[0], archive), (names[1], index)]
    expected = {names[0]: artifact["archive"], names[1]: artifact["index"]}
    for name, row in found.items():
        check(remote_matches(row, expected[name]), "Remote object differs; no overwrite: " + name)
    missing = [(name, p) for name, p in local if name not in found]
    revision = hub.commit(repo, missing, info.sha, workers) if missing else info.sha
    proof_info = hub.info(repo)
    check(proof_info.private is (plan["visibility"] == "private"), "Visibility changed during commit")
    proof = {f.path: f for f in hub.paths(repo, names, revision)}
    check(set(proof) == set(names) and all(remote_matches(proof[name], expected[name]) for name in names), "Remote verification incomplete")
    check(hub.read_index(repo, names[1], revision, work) == index.read_bytes(), "Downloaded restore index mismatch")
    return dict(plan_sha256=plan_sha, authorization_sha256=approval_sha, shard=base, repo_id=repo,
                revision=revision, paths=names, artifacts=expected, members=artifact["member_count"], verified=True)


def pipeline(plan_path, work, evidence, token_file, pack_workers=16, upload_workers=8, max_local=120, skip_verified_from=None):
    """Parallel pack -> upload -> remote verify -> delete local archive (index and receipts are kept)."""
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    check(1 <= pack_workers <= 64 and 1 <= upload_workers <= 16 and pack_workers <= max_local, "Invalid worker counts")
    plan = load_plan(plan_path)
    authorization(plan, plan_path, evidence)
    with work_lock(work, sha_file(plan_path)):
        plan_sha = bind_work(plan_path, work)
        # Space already used by packed local shards counts toward the backlog budget.
        check(shutil.disk_usage(work).free + sum(p.stat().st_size for p in work.glob("*.tar")) > max_local * plan["shard_bytes"] * 1.05,
              "Insufficient disk for max_local backlog")
        slim = {k: v for k, v in plan.items() if k not in ("shards", "excluded")}
        state = work / "state.jsonl"
        records = [json.loads(l) for l in state.read_text().splitlines() if l.strip()] if state.exists() else []
        previous = {r["shard"]: r for r in records}
        check(len(previous) == len(records) and all(r["plan_sha256"] == plan_sha for r in records), "Invalid or duplicate state receipts")
        # Shards already verified in another repo (receipts of an earlier plan with identical shards) are skipped.
        elsewhere = {}
        if skip_verified_from:
            for line in Path(skip_verified_from).read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    check(row.get("verified") is True and row["shard"] not in elsewhere, "Invalid external receipt")
                    elsewhere[row["shard"]] = row["repo_id"]
            check(not set(elsewhere) & set(previous), "Shard verified in both places")
        group_bytes = {}
        for s in plan["shards"]:
            group_bytes[s["group"]] = group_bytes.get(s["group"], 0) + sum(e["bytes"] for e in s["entries"])
        # Smaller groups first so they are safe even if the storage quota runs out near the end.
        ordered = sorted(plan["shards"], key=lambda s: (group_bytes[s["group"]], s["group"], s["index"]))
        todo = []
        for shard in ordered:
            archive, index, base = shard_files(plan, work, shard)
            if base in previous or base in elsewhere:
                continue
            receipt = work / (base + ".packed.json")
            if not (receipt.exists() and archive.exists() and index.exists()):
                for p in (archive, index, receipt, archive.with_name(archive.name + ".partial")):
                    if p.exists():
                        p.unlink()
            todo.append(shard)
        total, done = len(plan["shards"]), len(previous) + len(elsewhere)
        todo_bytes = sum(e["bytes"] for s in todo for e in s["entries"])
        emit("pipeline_start", planned_shards=total, already_verified=done, verified_elsewhere=len(elsewhere), todo_shards=len(todo),
             todo_gib=round(todo_bytes / 1024**3, 1), pack_workers=pack_workers, upload_workers=upload_workers)
        hub, start, moved = None, time.monotonic(), 0
        pending, ready, queue = {}, [], iter(todo)
        last_wait = None
        ex = ProcessPoolExecutor(pack_workers)

        def fill():
            # Bound local disk use: packing in flight + packed-but-not-uploaded <= max_local.
            while len(pending) < pack_workers and len(pending) + len(ready) < max_local:
                s = next(queue, None)
                if s is None:
                    return
                pending[ex.submit(_pack_one, slim, plan_sha, str(work), s)] = s
        try:
            fill()
            while pending or ready:
                if pending:
                    finished, _ = wait(list(pending), timeout=30, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        ready.append((pending.pop(fut), fut.result()))
                if hub is None and token_file.exists():
                    hub = _connect_hub(plan, token_file)
                fill()
                if hub is None:
                    if last_wait is None or time.monotonic() - last_wait >= 300:
                        last_wait = time.monotonic()
                        emit("waiting_for_token", token_file=str(token_file), packed_local=len(ready),
                             packing=len(pending), max_local=max_local, time=datetime.datetime.now().strftime("%H:%M:%S"))
                    if not pending:
                        time.sleep(30)
                    continue
                if not ready:
                    continue
                shard, artifact = ready.pop(0)
                archive, index, base = shard_files(plan, work, shard)
                approval_sha = authorization(plan, plan_path, evidence)
                t = time.monotonic()
                row = _upload_one(hub, plan, plan_sha, approval_sha, work, shard, artifact, upload_workers)
                with state.open("a") as f:
                    f.write(json.dumps(row, sort_keys=True) + chr(10))
                    f.flush()
                    os.fsync(f.fileno())
                archive.unlink()
                done += 1
                moved += artifact["archive"]["bytes"]
                elapsed = time.monotonic() - start
                rate = moved / elapsed if elapsed > 0 else 0
                left = todo_bytes - moved
                write_json(work / "status.json", dict(phase="running", verified_shards=done, planned_shards=total,
                           last_shard=base, scope_complete=False, whole_project_complete=False))
                emit("shard_verified", shard=base, verified_shards=done, planned_shards=total,
                     pct=round(100.0 * done / total, 2), upload_s=round(time.monotonic() - t, 1),
                     session_gib=round(moved / 1024**3, 1), mib_per_s=round(rate / 1024**2, 1),
                     eta_h=round(left / rate / 3600, 2) if rate > 0 else None, packed_local=len(ready),
                     time=datetime.datetime.now().strftime("%H:%M:%S"))
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown()
        check(done == total, "Backup scope is incomplete")
        write_json(work / "result.json", dict(scope_complete=True, plan_sha256=plan_sha, repo_id=plan["repo_id"],
                                             visibility=plan["visibility"], verified_shards=done,
                                             verified_elsewhere=dict(sorted(elsewhere.items())),
                                             selected_files=plan["selected_files"], selected_bytes=plan["selected_bytes"],
                                             whole_project_complete=False, excluded=plan["excluded"]))
        write_json(work / "status.json", dict(phase="scope_complete", verified_shards=done, planned_shards=total,
                                             scope_complete=True, whole_project_complete=False))
        emit("scope_complete", shards=done, selected_files=plan["selected_files"], whole_project_complete=False)


def safe_error(exc):
    text = re.sub(r"https?://\S+", "[URL]", str(exc))
    return SECRET.sub(b"[REDACTED]", text.encode()).decode(errors="replace")[-1500:]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="Offline: stat selected sources and freeze a plan; no HF calls")
    p.add_argument("--config", required=True)
    p.add_argument("--plan", required=True)
    p = sub.add_parser("pack", help="Offline: pack and scan one/all shards without uploading")
    p.add_argument("--plan", required=True)
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--limit", type=int, help="Only pack the first N shards")
    p = sub.add_parser("upload", help="Dry run unless --execute and matching authorization supplied")
    p.add_argument("--plan", required=True)
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--authorization")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--token-file", type=Path, help="Optional existing private token file; no token in CLI arguments")
    p = sub.add_parser("pipeline", help="Parallel pack + authorized upload + verify + delete local archive")
    p.add_argument("--plan", required=True)
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--authorization", required=True)
    p.add_argument("--token-file", required=True, type=Path)
    p.add_argument("--pack-workers", type=int, default=16)
    p.add_argument("--upload-workers", type=int, default=8)
    p.add_argument("--max-local", type=int, default=120, help="Max shards packed locally but not yet uploaded")
    p.add_argument("--skip-verified-from", type=Path, help="state.jsonl of an earlier plan; shards verified there are skipped")
    p = sub.add_parser("status", help="Offline status only; does not restart jobs")
    p.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if args.command == "plan":
            build_plan(args.config, args.plan)
        elif args.command == "pack":
            plan = load_plan(args.plan)
            check(args.limit is None or args.limit > 0, "Pack limit must be positive")
            with work_lock(args.work_dir, sha_file(args.plan)):
                digest = bind_work(args.plan, args.work_dir)
                todo = plan["shards"][:args.limit]
                for i, shard in enumerate(todo):
                    emit("pack_shard_start", shard_no=i + 1, shards=len(todo), group=shard["group"], index=shard["index"], files=len(shard["entries"]))
                    pack_shard(plan, digest, args.work_dir, shard)
        elif args.command == "pipeline":
            pipeline(args.plan, args.work_dir, args.authorization, args.token_file, args.pack_workers, args.upload_workers, args.max_local, args.skip_verified_from)
        elif args.command == "upload":
            token = None
            if args.execute:
                authorization(load_plan(args.plan), args.plan, args.authorization)
                if args.token_file:
                    mode = args.token_file.stat().st_mode
                    check(not args.token_file.is_symlink() and stat.S_ISREG(mode) and not mode & 0o077, "Token file must be regular, private and not a symlink")
                    token = args.token_file.read_text().strip()
                    check(bool(token), "Empty authentication token file")
                else:
                    token = os.environ.get("HF_TOKEN") or None  # Hub login cache is also supported.
            upload(args.plan, args.work_dir, args.authorization, args.execute, token, args.workers)
        else:
            current = read_json(args.work_dir / "status.json") if (args.work_dir / "status.json").exists() else {"phase": "idle"}
            result = read_json(args.work_dir / "result.json") if (args.work_dir / "result.json").exists() else None
            error = read_json(args.work_dir / "error.json") if (args.work_dir / "error.json").exists() else None
            emit("status", current=current, last_scope_result=result, last_error=error,
                 scope_complete=bool(result and current.get("phase") == "scope_complete"),
                 whole_project_complete=False)
        return 0
    except Exception as exc:
        error = dict(error_type=type(exc).__name__, http_status=getattr(getattr(exc, "response", None), "status_code", None),
                     safe_error=safe_error(exc), automatic_retry=False, scope_complete=False, whole_project_complete=False,
                     checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        emit("blocked", **error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())