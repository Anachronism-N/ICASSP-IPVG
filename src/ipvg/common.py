"""Small shared helpers; credentials and environment dumps are deliberately absent."""
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def resolve_path(value, root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def safe_name(value, label="name"):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value) or ".." in value:
        raise ValueError("%s must be a simple alphanumeric filename identifier" % label)
    return value


def positive(value, label, *, allow_zero=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("%s must be finite numeric" % label)
    if value < 0 or (not allow_zero and value == 0) or (integer and int(value) != value):
        raise ValueError("%s has invalid range or integer value" % label)
    return value
