#!/usr/bin/env python3
"""One explicitly reviewed t10 rebuild, not a general binary-hash override.

The review is evidence of matching reported source/options, not a measurement
of numerical equivalence. Reusing t10's old endpoint remains an assumption;
all new probes/candidates must bind their actual execution binary separately.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ANCHOR_IC_SHA256 = "914ca086c68e1e5d04d6aff53ee3284cd67092241e330bd5ee3dd6c9224a9579"
ANCHOR_BINARY_SHA256 = "ce77566c66921818f46099edb29ac8674e315fdc0f22cd1dc1ea24b9e7c7361c"
EXECUTION_BINARY_SHA256 = "d53fddf37f5cf6218195e95e9e9c646e28d3712a64ceaad3bba6fc948ab3b827"
SOURCE_COMMIT = "d0e0d19354b5ba6c8169c2051a7e7806a7e5a529"

REVIEW_DATA = {
    "schema": "local-live-reviewed-build-v1",
    "review_id": "t10-rebuild-20261023",
    "anchor_trial": "trial10",
    "anchor_ic_sha256": ANCHOR_IC_SHA256,
    "anchor_binary_sha256": ANCHOR_BINARY_SHA256,
    "execution_binary_sha256": EXECUTION_BINARY_SHA256,
    "source_commit": SOURCE_COMMIT,
    "anchor_reported_build_time": "Oct 8 2026 19:52:45",
    "execution_reported_build_time": "Oct 23 2026 18:30:55",
    "observed_compiled_options": [
        "HYDRO_MESHLESS_FINITE_MASS", "GRAVITY_NOT_PERIODIC", "EVALPOTENTIAL",
        "ADAPTIVE_GRAVSOFT_FORGAS", "COOLING", "METALS",
        "COOL_METAL_LINES_BY_SPECIES", "COOL_UVB_SELFSHIELD_RAHMATI",
        "COOL_LOW_TEMPERATURES", "COOL_MOLECFRAC=5",
        "INIT_STELLAR_METALS_AGES_DEFINED", "BLACK_HOLES",
        "BH_REPOSITION_ON_POTMIN=2", "GALSF", "GALSF_SFR_CRITERION=1",
        "GALSF_FB_MECHANICAL", "OUTPUT_IN_DOUBLEPRECISION",
        "INPUT_IN_DOUBLEPRECISION", "OUTPUT_TEMPERATURE", "OUTPUT_POTENTIAL",
        "OUTPUT_ACCELERATION", "OUTPUT_VORTICITY", "OUTPUT_TIMESTEP",
        "OUTPUT_CHANGEOFENERGY", "OUTPUT_ADDITIONAL_RUNINFO",
        "STOP_WHEN_BELOW_MINTIMESTEP", "OUTPUT_COOLRATE", "OUTPUT_COOLRATE_DETAIL",
    ],
    "evidence": {
        "anchor_startup": "User-supplied t10_build_context.txt: t10 production startup reports master:d0e0d19 and the 28 options above.",
        "execution_startup": "User-supplied current_gizmo_startup.txt transcript: current executable reports master:d0e0d19 and the same 28 options.",
        "current_git_status": "User-supplied rev-parse gives the full source_commit above; git status --short reports untracked files only.",
        "binary_hashes": "User-supplied gizmo_binary_audit_t10.json records the old audit SHA and current executable SHA; selection re-hashes the execution file.",
        "configuration_files": "Current Config.sh active options and GIZMO_config.h agree with both reported startup option lists.",
    },
    "baseline_reuse": "assumed_equivalent_not_measured",
    "endpoint_numerical_equivalence_measured": False,
    "live_candidate_validation_required": True,
    "limitations": [
        "Matching source identifier and printed options do not prove identical compiler flags, compiler versions, linked libraries, or floating-point results.",
        "The historical working-tree dirty state and historical build environment were not recorded; the current source tree alone cannot reconstruct them.",
        "The supplied startup transcript and binary audit are observational evidence; this review does not recreate or execute the historical binary.",
        "Reusing the old baseline assumes any build-induced endpoint offset is negligible for these local probes; that offset has not been measured.",
        "All new runs must use the reviewed execution SHA consistently. Predictions remain provisional until the actual candidate endpoint is validated.",
    ],
}


def _hash_value(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"构建复核缺少有效 {label}")
    return value.lower()


def _check_anchor(anchor):
    if not isinstance(anchor, dict):
        raise ValueError("构建复核缺少 trial10 锚点")
    if _hash_value(anchor.get("ic_sha256"), "IC SHA256") != ANCHOR_IC_SHA256:
        raise ValueError("该构建复核仅适用于已经审计的 trial10 IC")
    if _hash_value(anchor.get("gizmo_sha256"), "旧 GIZMO SHA256") != ANCHOR_BINARY_SHA256:
        raise ValueError("该构建复核与 trial10 原 GIZMO 指纹不符")


def load_review(path, anchor):
    """Load the exact reviewed record, binding it to the original t10 IC/SHA."""
    _check_anchor(anchor)
    try:
        actual = Path(path).expanduser().resolve(strict=True)
        if not actual.is_file() or actual.stat().st_size > 1024 * 1024:
            raise ValueError("不是有效的小型 JSON 文件")
        raw = actual.read_bytes()
        data = json.loads(raw)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"无法读取构建复核文件：{exc}") from exc
    if data != REVIEW_DATA:
        raise ValueError("构建复核内容不是本版已审查的 t10 重新编译记录；不能编辑 SHA 或扩大适用范围")
    return {"path": str(actual), "sha256": hashlib.sha256(raw).hexdigest(), "data": data}


def validate_review(record, anchor):
    """Re-read the frozen review file; never silently refresh a changed record."""
    if record is None:
        return None
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "data"}:
        raise ValueError("构建复核记录结构不完整")
    fresh = load_review(record["path"], anchor)
    if fresh != record:
        raise ValueError("构建复核文件或记录在冻结后已改变")
    return fresh


def assert_build_pair(old_sha, new_sha, review=None):
    """Accept identical bytes, or this one directional reviewed rebuild pair."""
    old_sha = _hash_value(old_sha, "来源 GIZMO SHA256")
    new_sha = _hash_value(new_sha, "执行 GIZMO SHA256")
    if old_sha == new_sha:
        return
    if review is None:
        raise ValueError("GIZMO 内容不是 t10 记录的同一构建，且未指定已核对的构建复核记录")
    checked = validate_review(review, {"ic_sha256": ANCHOR_IC_SHA256,
                                       "gizmo_sha256": old_sha})
    if (old_sha != ANCHOR_BINARY_SHA256 or new_sha != EXECUTION_BINARY_SHA256
            or checked["data"]["execution_binary_sha256"] != new_sha):
        raise ValueError("当前 GIZMO 不属于本次已审查的旧构建到新构建组合")

