#!/usr/bin/env python3
"""
Show how Python resolves GRADE_DATABASE_URL host (IPv4 vs IPv6).
Usage (repo root): PYTHONPATH=src python scripts/diagnose_pg_dns.py
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse


def _load_dotenv() -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        from dotenv import load_dotenv

        try:
            load_dotenv(root / ".env", interpolate=False, override=True)
        except TypeError:
            load_dotenv(root / ".env", override=True)
    except ImportError:
        pass


def main() -> int:
    _load_dotenv()
    raw = (os.environ.get("GRADE_DATABASE_URL") or os.environ.get("DATABASE_URL") or "").strip()
    if not raw:
        print("GRADE_DATABASE_URL / DATABASE_URL is not set.", file=sys.stderr)
        return 1
    p = urlparse(raw)
    host = p.hostname
    port = p.port or 5432
    print(f"Host: {host!r}  port: {port}")
    if not host:
        print("Could not parse hostname from URL.", file=sys.stderr)
        return 1

    for fam, label in ((socket.AF_INET, "IPv4 (AF_INET)"), (socket.AF_INET6, "IPv6 (AF_INET6)")):
        try:
            infos = socket.getaddrinfo(host, port, fam, socket.SOCK_STREAM)
            addrs = [x[4][0] for x in infos[:5]]
            print(f"  {label}: OK -> {addrs}")
        except socket.gaierror as e:
            print(f"  {label}: FAIL -> {e}")

    try:
        infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        print(f"  AF_UNSPEC: {len(infos)} addrinfo(s)")
    except socket.gaierror as e:
        print(f"  AF_UNSPEC: FAIL -> {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
