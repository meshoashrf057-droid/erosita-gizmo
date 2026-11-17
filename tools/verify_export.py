"""Verify source copies and parse Python without executing scientific scripts."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    records = json.loads((root / "provenance/source_manifest.json").read_text(encoding="utf-8"))
    problems = []
    for rec in records:
        path = root / rec["path"]
        if not path.is_file():
            problems.append({"path": rec["path"], "error": "missing"})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != rec["sha256"]:
            problems.append({"path": rec["path"], "error": "SHA256 differs"})
    count = 0
    for path in root.rglob("*.py"):
        if any(part in {".venv", ".git", "__pycache__"} for part in path.parts):
            continue
        count += 1
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            problems.append({"path": str(path.relative_to(root)), "error": str(exc)})
    print(json.dumps({"copied_files": len(records), "python_files_parsed": count,
                      "problems": problems, "passed": not problems}, indent=2))
    return 1 if problems else 0

if __name__ == "__main__":
    sys.exit(main())
