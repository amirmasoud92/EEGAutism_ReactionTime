"""Writing pooled results as JSON."""

from __future__ import annotations

import json
from pathlib import Path


def write_json(path, obj) -> Path:
    """Write ``obj`` as indented JSON with sorted keys (UTF-8, trailing newline); returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
    return p
