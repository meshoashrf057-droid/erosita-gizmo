#!/usr/bin/env python3
"""Check the exported workflow environment without importing research scripts.

Only the fixed third-party dependency list is imported, in isolated child Python
processes. Repository modules are parsed as text. No snapshots are opened and no
simulation, analysis pipeline, executable under --gizmo, or download is started.
"""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
DEPENDENCIES = (
    ("numpy", "numpy"), ("scipy", "scipy"), ("h5py", "h5py"),
    ("matplotlib", "matplotlib"), ("yt", "yt"), ("healpy", "healpy"),
    ("PIL", "Pillow"), ("unyt", "unyt"), ("astropy", "astropy"),
    ("tqdm", "tqdm"),
)
ROOT_MODULES = (
    "gizmo_compat.py", "gizmo_schema.py", "gizmo_diagnostics_compat.py",
    "gizmo_units.py", "gizmo_ic_compat.py",
)
IMPORT_CODE = (
    "import importlib, importlib.metadata, json, sys; "
    "importlib.import_module(sys.argv[1]); "
    "print(json.dumps({'version': importlib.metadata.version(sys.argv[2])}))"
)


def record(rows, status, name, detail):
    rows.append({"status": status, "name": name, "detail": detail})


def check_dependency(rows, module, distribution, timeout, temporary):
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(temporary / "matplotlib"),
               XDG_CACHE_HOME=str(temporary / "cache"),
               NUMBA_CACHE_DIR=str(temporary / "numba"),
               PYTHONDONTWRITEBYTECODE="1")
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-B", "-c", IMPORT_CODE, module, distribution],
            cwd=temporary, env=env, capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        record(rows, "FAIL", distribution, str(exc))
        return
    if process.returncode:
        detail = (process.stderr or process.stdout).strip()[-1600:]
        record(rows, "FAIL", distribution,
               f"Import failed (exit {process.returncode}): {detail}")
        return
    try:
        version = json.loads(process.stdout.strip().splitlines()[-1])["version"]
    except (ValueError, IndexError, KeyError):
        record(rows, "WARN", distribution, "Imported, but version output was unexpected.")
    else:
        record(rows, "PASS", distribution, f"Import OK; version {version}")


def check_root_modules(rows):
    for name in ROOT_MODULES:
        path = REPO / name
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            record(rows, "FAIL", name, str(exc))
        else:
            record(rows, "PASS", name, "Present and syntax parses; not imported.")


def check_command(rows, name, purpose):
    path = shutil.which(name)
    if path:
        record(rows, "PASS", name, f"Found {path}; not executed.")
    else:
        record(rows, "WARN", name, f"Not on PATH; needed for {purpose}.")


def check_apec(rows, path):
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Missing or empty file: {path}")
        # Check the HDF5 signature only, including valid HDF5 user-block offsets.
        with path.open("rb") as stream:
            found = False
            for offset in (0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
                stream.seek(offset)
                if stream.read(8) == b"\x89HDF\r\n\x1a\n":
                    found = True
                    break
        if not found:
            raise OSError("HDF5 signature not found in the inspected header offsets.")
    except OSError as exc:
        record(rows, "FAIL", "APEC", str(exc))
        return
    status = "PASS" if path.name == "apec_emissivity_v3.h5" else "WARN"
    record(rows, status, "APEC",
           f"{path} ({path.stat().st_size} bytes), HDF5 signature present. "
           "No emissivity arrays read; schema/version/content not scientifically validated.")


def check_gizmo(rows, argument):
    if argument is None:
        record(rows, "SKIP", "GIZMO",
               "No --gizmo supplied; Python postprocessing can be checked independently.")
        return
    path = argument.expanduser().resolve()
    try:
        if not path.is_file():
            raise OSError(f"Not a regular file: {path}")
        with path.open("rb") as stream:
            magic = stream.read(4)
    except OSError as exc:
        record(rows, "FAIL", "GIZMO", str(exc))
        return
    if not os.access(path, os.X_OK):
        record(rows, "FAIL", "GIZMO", f"Not executable by this user: {path}")
        return
    macho = {
        bytes.fromhex(value) for value in
        ("feedface", "cefaedfe", "feedfacf", "cffaedfe",
         "cafebabe", "bebafeca", "cafebabf", "bfbafeca")
    }
    kind = ("ELF" if magic == b"\x7fELF" else "Mach-O" if magic in macho else
            "PE/Windows" if magic.startswith(b"MZ") else
            "script" if magic.startswith(b"#!") else "unrecognized")
    system = platform.system()
    if system == "Darwin" and kind in ("ELF", "PE/Windows"):
        record(rows, "FAIL", "GIZMO",
               f"{path}: {kind} cannot run natively on macOS. Rebuild GIZMO for this Mac.")
    elif system == "Linux" and kind in ("Mach-O", "PE/Windows"):
        record(rows, "FAIL", "GIZMO", f"{path}: {kind} is not a native Linux binary.")
    else:
        record(rows, "WARN", "GIZMO",
               f"{path}: {kind}; host {system}/{platform.machine()}. "
               "File inspected only; CPU architecture, linked libraries, MPI and "
               "scientific build configuration are not validated. Never executed.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apec-table", type=Path,
                        default=REPO / "erosita_xray" / "apec_emissivity_v3.h5",
                        help="APEC HDF5 file; default is the table included in this repository.")
    parser.add_argument("--gizmo", type=Path,
                        help="Optional GIZMO executable path to inspect, never execute.")
    parser.add_argument("--timeout", type=float, default=45.0,
                        help="Maximum seconds per third-party import (default: 45).")
    parser.add_argument("--json", action="store_true",
                        help="Print the check report as JSON to stdout.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main():
    args = parse_args()
    rows = []
    version = sys.version_info[:2]
    status = "PASS" if version == (3, 12) else "FAIL" if version < (3, 12) else "WARN"
    record(rows, status, "Python",
           f"{platform.python_version()} at {sys.executable}; setup target is Python 3.12.")
    record(rows, "PASS", "Platform", f"{platform.system()} / {platform.machine()}")
    check_root_modules(rows)
    with tempfile.TemporaryDirectory(prefix="erosita-doctor-") as folder:
        for module, distribution in DEPENDENCIES:
            check_dependency(rows, module, distribution, args.timeout, Path(folder))
    check_command(rows, "ffmpeg", "movie encoding; still-image analysis does not require it")
    check_command(rows, "sha256sum", "legacy GNU checksum commands")
    check_apec(rows, args.apec_table.expanduser().resolve())
    check_gizmo(rows, args.gizmo)
    failed = sum(row["status"] == "FAIL" for row in rows)
    report = {
        "repository": str(REPO), "checks": rows, "failed_checks": failed,
        "scope": "Environment only; no scientific scripts imported and no simulation started.",
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Repository: {REPO}")
        for row in rows:
            print(f"[{row['status']}] {row['name']}: {row['detail']}")
        print(f"\n{failed} failed check(s). {report['scope']}")
        print("Passing these checks does not establish macOS runtime or scientific equivalence.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
