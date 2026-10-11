"""Sinh data/MANIFEST.csv: đường dẫn, kích thước, SHA-256 của mọi tệp dữ liệu.

Dùng để đối chiếu khi chép dữ liệu giữa các máy (phần lớn tệp không commit vì DUA).
Chạy từ gốc repo:  python scripts/make_data_manifest.py
"""
from __future__ import annotations

import csv
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [
    ROOT / "data",
    ROOT / "experiments" / "07_v6_multidomain" / "artifacts" / "hard_domain_splits",
    ROOT / "experiments" / "08_v6_locked_protocol" / "artifacts",
    ROOT / "models",
]
OUT = ROOT / "data" / "MANIFEST.csv"
SKIP_NAMES = {"MANIFEST.csv", ".gitkeep"}
SKIP_SUFFIXES = {".md"}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, in tiếng Việt sẽ lỗi
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    rows = []
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name in SKIP_NAMES or path.suffix in SKIP_SUFFIXES:
                continue
            rel = path.relative_to(ROOT).as_posix()
            rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})
            print(f"{rows[-1]['bytes']:>12}  {rel}", file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} tệp -> {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
