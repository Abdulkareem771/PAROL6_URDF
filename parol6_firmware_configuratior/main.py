#!/usr/bin/env python3
"""Compatibility wrapper for the old misspelled configurator path."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    real_main = repo_root / "parol6_firmware_configurator" / "main.py"
    os.execv(sys.executable, [sys.executable, str(real_main), *sys.argv[1:]])


if __name__ == "__main__":
    main()
