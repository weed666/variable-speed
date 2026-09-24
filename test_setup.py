#!/usr/bin/env python3
"""Lightweight MyOSys installation verification; does not start training."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from setuptools import find_packages


REPOSITORY_ROOT = Path(__file__).resolve().parent
REQUIRED_PACKAGES = ("myosuite", "myoassist_utils", "rl_train", "tcn")


def main() -> int:
    if sys.version_info < (3, 11):
        raise RuntimeError(f"MyOSys requires Python 3.11 or newer, found {sys.version.split()[0]}")

    discovered = set(find_packages(where=str(REPOSITORY_ROOT)))
    failures: list[str] = []
    for package in REQUIRED_PACKAGES:
        if package not in discovered:
            failures.append(f"package discovery did not find {package}")
            continue
        try:
            module = importlib.import_module(package)
        except Exception as exc:
            failures.append(f"import {package} failed: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS import {package}: {module.__file__}")

    deploy_model = REPOSITORY_ROOT / "tcn/deployment/unified_tcn_latest_100hz_deploy.pt"
    if not deploy_model.is_file():
        failures.append(f"missing deployment model: {deploy_model.relative_to(REPOSITORY_ROOT)}")
    else:
        print(f"PASS deployment model: {deploy_model.relative_to(REPOSITORY_ROOT)}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1

    print("MyOSys setup verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
