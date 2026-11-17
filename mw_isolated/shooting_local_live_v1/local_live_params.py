#!/usr/bin/env python3
"""Compare the parameters actually generated for production with the t10 run.

A base parameter file is a template. Its bytes and its overridden time/path
entries need not equal the historical template. Every resulting non-path
production value must, however, equal the frozen t10 production parameters.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import tempfile

from local_live_evidence import normalized_params, paths, read_params, sha256

IGNORED_PATH_KEYS = {"InitCondFile", "OutputDir", "RestartFile", "SnapshotFileBase"}


def _normalize(values):
    result = {}
    for key, value in values.items():
        if key in IGNORED_PATH_KEYS:
            continue
        try:
            number = float(value)
        except (ValueError, TypeError):
            result[key] = value
        else:
            if not math.isfinite(number):
                raise ValueError(f"param 含非有限数值：{key}={value}")
            result[key] = number
    return result


def production_overrides(anchor):
    """Mirror create_stage2_files production overrides, excluding IO paths."""
    start = float(anchor["stage2_start_code"])
    target = float(anchor["target_time_code"])
    if not math.isfinite(start) or not math.isfinite(target) or target <= start:
        raise ValueError("基准起止时间无效")
    return dict(ICFormat=3, SnapFormat=3, OutputListOn=0,
                NumFilesPerSnapshot=1, NumFilesWrittenInParallel=1,
                SnapshotFileBase="snapshot", RestartFile="restart",
                TimeBegin=format(start, ".16g"), InitGasTemp=0,
                TimeMax=format(target, ".16g"),
                TimeOfFirstSnapshot=format(start, ".16g"), TimeBetSnapshot="0.1")


def _anchor_runtime_path(anchor):
    explicit = anchor.get("runtime_param_path")
    if explicit:
        return Path(explicit).expanduser().resolve()
    if anchor.get("work_dir") and anchor.get("trial"):
        return paths(anchor["work_dir"], anchor["trial"])["runtime_param"]
    return None


def assess_base_param(anchor, path):
    """Return a serializable semantic comparison without altering any file."""
    candidate = Path(path).expanduser().resolve(strict=True)
    historical = Path(anchor["base_param_path"]).expanduser().resolve()
    if candidate.parent != historical.parent:
        raise ValueError("base-param 必须与 trial10 原参数位于同一目录，保持相对冷却表等路径的运行含义")
    values = read_params(candidate)
    # Detect malformed numerical values even when the generator overrides them.
    _normalize(values)
    overrides = production_overrides(anchor)
    effective = _normalize(dict(values, **overrides))
    expected = _normalize(anchor["runtime_param_normalized"])
    differences = [dict(key=key, anchor=expected.get(key), candidate=effective.get(key),
                        missing_from_anchor=key not in expected,
                        missing_from_candidate=key not in effective)
                   for key in sorted(set(expected) | set(effective))
                   if key not in expected or key not in effective or expected[key] != effective[key]]
    runtime = _anchor_runtime_path(anchor)
    actual_hash = sha256(candidate)
    return dict(schema="local-live-production-param-comparison-v1",
                compatible=not differences, differences=differences,
                actual_sha256=actual_hash, original_base_sha256=anchor["base_param_sha256"],
                byte_identical=actual_hash == anchor["base_param_sha256"],
                effective_normalized=effective, source_path=str(candidate),
                anchor_runtime_path=None if runtime is None else str(runtime),
                anchor_runtime_sha256=anchor.get("runtime_param_sha256"),
                generator_overrides=overrides,
                comparison="all_effective_production_values_except_generated_io_paths")


def assert_base_param_compatible(anchor, path):
    report = assess_base_param(anchor, path)
    if not report["compatible"]:
        lines = []
        for item in report["differences"]:
            before = "<缺失>" if item["missing_from_anchor"] else repr(item["anchor"])
            after = "<缺失>" if item["missing_from_candidate"] else repr(item["candidate"])
            lines.append(f"  {item['key']}: trial10={before}；当前生成后={after}")
        raise ValueError("base-param 生成的 production 参数与 trial10 不同：\n" + "\n".join(lines)
                         + "\n请使用 t10 配置，或用 recover-param 从保留的 t10 production/stage2.param 生成独立副本。")
    return report


def recover_base_param(anchor, output):
    """Copy the frozen t10 production file into a new local base template.

    The source is revalidated against its recorded bytes and parsed values.
    No old isolated base file is required. Existing output is only reused when
    byte-identical; original files and symlinks are never overwritten.
    """
    runtime = _anchor_runtime_path(anchor)
    if runtime is None or not runtime.is_file():
        raise ValueError("缺少 t10 production/stage2.param，无法恢复参数副本")
    data = runtime.read_bytes()
    expected_hash = anchor.get("runtime_param_sha256")
    if not expected_hash or hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("t10 production/stage2.param 指纹已变化；不能据此恢复旧运行配置")
    if normalized_params(runtime) != anchor["runtime_param_normalized"]:
        raise ValueError("t10 production/stage2.param 有效参数与冻结基准不一致")
    original = Path(anchor["base_param_path"]).expanduser().resolve()
    destination_raw = Path(output).expanduser().absolute()
    if destination_raw.is_symlink():
        raise ValueError("参数恢复输出不能是符号链接")
    destination = destination_raw.resolve()
    if destination.parent != original.parent:
        raise ValueError("恢复副本必须放在 trial10 原 base-param 同一目录")
    if destination in (original, runtime):
        raise ValueError("参数恢复必须使用新的副本路径，不得覆盖原 base-param 或 t10 production 参数")
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".t10_param_check_", suffix=".param") as stream:
        stream.write(data)
        stream.flush()
        report = assert_base_param_compatible(anchor, Path(stream.name))
    reused = destination.exists()
    if reused:
        if not destination.is_file() or destination.read_bytes() != data:
            raise ValueError(f"拒绝覆盖已有参数文件：{destination}")
    else:
        # Exclusive creation also protects against a file appearing after validation.
        with destination.open("xb") as stream:
            stream.write(data)
    report.update(source_path=str(destination), actual_sha256=sha256(destination),
                  byte_identical=sha256(destination) == anchor["base_param_sha256"],
                  recovered_from=str(runtime), reused_existing=reused)
    return report
