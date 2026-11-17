#!/usr/bin/env python3
"""Find a t10 executable without executing or replacing any file.

A different SHA does not diagnose a different physical configuration. It does
mean that the old endpoint cannot by itself establish this executable's local
response. Selection defaults to an exact old-build copy at any path. A separate,
explicit review may authorize the single documented same-config rebuild;
numerical equivalence is not implied by that review.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

from local_live_build_review import validate_review, assert_build_pair


def _expected(anchor):
    value = anchor.get("gizmo_sha256")
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("trial10 缺少有效的 GIZMO SHA256 记录")
    return value.lower()


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = Path(path).stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ValueError("文件在计算 SHA256 期间发生变化")
    return digest.hexdigest()


def select_binary(anchor, override=None, build_review=None):
    """Select exact bytes by default, or the explicitly reviewed rebuild pair."""
    expected = _expected(anchor)
    checked_review = validate_review(build_review, anchor)
    supplied = override if override is not None else anchor.get("gizmo_path")
    if not supplied:
        raise ValueError("缺少 GIZMO 路径；请用 binary-check 查找 t10 旧程序备份")
    path = Path(supplied).expanduser().absolute()
    actual = "<无法读取>"
    try:
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("不是普通文件")
        actual = _hash_file(path)
        assert_build_pair(expected, actual, checked_review)
        if not os.access(path, os.X_OK):
            raise ValueError("内容匹配，但文件没有执行权限")
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"GIZMO 检查失败：{exc}\n  路径：{path}\n"
            f"  t10 期望 SHA256：{expected}\n  当前 SHA256：{actual}\n"
            "请运行 binary-check 查找旧程序备份，再用 --gizmo-executable 指向匹配副本。"
            "已有逐项复核记录时可使用配套 --reviewed-build；该记录仅允许指定的构建组合。"
            "不能仅因参数相同就跳过构建检查；不同 SHA 也不能单独证明物理模块不同。"
        ) from exc
    return dict(path=str(path), sha256=actual)


def audit_binaries(anchor, search_dirs=(), candidates=()):
    """Read-only ELF search in explicitly provided directories.

    Directory symlinks are not followed. Recursive candidates must have
    'gizmo' in their basename; explicitly supplied files may have any name.
    Only ELF candidates are fully hashed, so configuration/log files and large
    data files are not treated as executable backups. No subprocess is run.
    """
    expected = _expected(anchor)
    recorded = anchor.get("gizmo_path")
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("trial10 缺少 GIZMO 原路径记录")
    records, errors, seen, roots = [], [], set(), []

    def inspect(raw, origin):
        unresolved = Path(raw).expanduser().absolute()
        try:
            path = unresolved.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            errors.append(dict(path=str(unresolved), error=str(exc), operation="resolve"))
            return
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        record = dict(path=key, requested_path=str(unresolved), origin=origin,
                      sha256=None, executable=False, size_bytes=None,
                      sha_matches=False, status="unreadable")
        records.append(record)
        try:
            if not path.exists():
                record["status"] = "missing"
                raise FileNotFoundError(f"文件不存在：{path}")
            if not path.is_file():
                record["status"] = "not_file"
                return
            record["size_bytes"] = path.stat().st_size
            record["executable"] = bool(os.access(path, os.X_OK))
            with path.open("rb") as stream:
                magic = stream.read(4)
            if magic != b"\x7fELF":
                record["status"] = "not_elf"
                return
            actual = _hash_file(path)
            record["sha256"] = actual
            record["sha_matches"] = actual == expected
            record["status"] = ("exact_match" if record["executable"] else "exact_match_not_executable") if actual == expected else "different_build"
        except (OSError, ValueError) as exc:
            record["error"] = str(exc)
            errors.append(dict(path=key, error=str(exc), operation="read_candidate"))

    inspect(recorded, "anchor_recorded_path")
    for candidate in candidates:
        inspect(candidate, "explicit_candidate")
    visited_roots = set()
    for raw in search_dirs:
        unresolved = Path(raw).expanduser().absolute()
        if unresolved.is_symlink():
            errors.append(dict(path=str(unresolved), error="搜索目录是符号链接，未跟随", operation="search_directory"))
            continue
        try:
            root = unresolved.resolve(strict=True)
            if not root.is_dir():
                raise ValueError("搜索路径不是目录")
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(dict(path=str(unresolved), error=str(exc), operation="search_directory"))
            continue
        if str(root) in visited_roots:
            continue
        visited_roots.add(str(root)); roots.append(str(root))

        def walk_error(exc):
            errors.append(dict(path=str(getattr(exc, "filename", root)), error=str(exc), operation="walk"))

        for directory, dirnames, filenames in os.walk(root, followlinks=False, onerror=walk_error):
            dirnames[:] = sorted(d for d in dirnames if not (Path(directory) / d).is_symlink())
            for name in sorted(filenames):
                if "gizmo" in name.lower():
                    inspect(Path(directory) / name, "recursive_name_match")

    return dict(
        schema="local-live-binary-audit-v1", anchor_expected_sha256=expected,
        anchor_recorded_path=recorded, searched_directories=roots,
        recursive_name_filter="basename contains gizmo (case insensitive); ELF magic required for hashing",
        follows_directory_symlinks=False, candidates=records, errors=errors,
        exact_match_paths=[r["path"] for r in records if r["status"] == "exact_match"],
        exact_match_nonexecutable_paths=[r["path"] for r in records if r["status"] == "exact_match_not_executable"],
        scan_complete=not errors,
        scan_complete_scope="Only the named directories and explicit candidates; directory symlinks excluded.",
        notes=[
            "只读检查；没有执行 GIZMO、修改权限、替换文件或改写 t10 元数据。",
            "不同 SHA 可能来自物理配置、源码、编译器或构建过程差异；本检查不据此判断原因。",
            "未找到匹配不证明旧程序不存在；可增加搜索目录或用 --candidate 指定任意名称的 ELF 文件。",
            "若无旧构建，先核对实际源码、启动选项和读入兼容性。本版可用明确的 reviewed-build 记录接受已审查的指定重新编译，但不宣称终点数值等价。",
            "未经复核或已知存在物理差异的新构建应重新建立未扰动基准；复用旧基准时必须明确构建引起的终点偏移尚未测量，并验证实际候选。",
        ])
