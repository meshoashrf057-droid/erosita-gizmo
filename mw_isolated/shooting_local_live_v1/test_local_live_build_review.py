"""Tests for the one reviewed rebuild; no GIZMO executable is ever run."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import local_live_build_review as br
from local_live_binary import select_binary


class ReviewedBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.review_path = self.root / "review.json"
        self.review_path.write_text(json.dumps(br.REVIEW_DATA, indent=2) + "\n")
        self.binary = self.root / "GIZMO"
        self.binary.write_bytes(b"\x7fELF synthetic; never execute")
        self.binary.chmod(0o700)
        self.anchor = {
            "ic_sha256": br.ANCHOR_IC_SHA256,
            "gizmo_sha256": br.ANCHOR_BINARY_SHA256,
            "gizmo_path": str(self.binary),
        }

    def record(self):
        return br.load_review(self.review_path, self.anchor)

    def test_canonical_review_is_explicit_about_numerical_uncertainty(self):
        data = self.record()["data"]
        self.assertEqual(data["baseline_reuse"], "assumed_equivalent_not_measured")
        self.assertFalse(data["endpoint_numerical_equivalence_measured"])
        self.assertTrue(data["live_candidate_validation_required"])
        self.assertEqual(len(data["observed_compiled_options"]), 28)
        self.assertIn("historical working-tree dirty", " ".join(data["limitations"]))
        self.assertIn("compiler flags", " ".join(data["limitations"]))

    def test_load_and_recheck_preserve_anchor_and_content(self):
        before = copy.deepcopy(self.anchor)
        raw = self.review_path.read_bytes()
        record = self.record()
        self.assertEqual(record["path"], str(self.review_path.resolve()))
        self.assertEqual(record["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(br.validate_review(record, self.anchor), record)
        self.assertEqual(self.anchor, before)
        self.assertEqual(self.review_path.read_bytes(), raw)

    def test_no_review_is_default_and_different_build_rejected(self):
        self.assertIsNone(br.validate_review(None, self.anchor))
        with self.assertRaisesRegex(ValueError, "未指定"):
            br.assert_build_pair(br.ANCHOR_BINARY_SHA256, br.EXECUTION_BINARY_SHA256)
        with patch("local_live_binary._hash_file", return_value=br.EXECUTION_BINARY_SHA256):
            with self.assertRaises(ValueError):
                select_binary(self.anchor)

    def test_reviewed_pair_is_directional_and_scoped(self):
        record = self.record()
        br.assert_build_pair(br.ANCHOR_BINARY_SHA256, br.EXECUTION_BINARY_SHA256, record)
        for old, new in ((br.EXECUTION_BINARY_SHA256, br.ANCHOR_BINARY_SHA256),
                         (br.ANCHOR_BINARY_SHA256, "1" * 64),
                         ("2" * 64, br.EXECUTION_BINARY_SHA256)):
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                br.assert_build_pair(old, new, record)

    def test_exact_same_hash_needs_no_review(self):
        br.assert_build_pair("3" * 64, "3" * 64)

    def test_wrong_anchor_ic_or_old_binary_rejected(self):
        for key in ("ic_sha256", "gizmo_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                br.load_review(self.review_path, dict(self.anchor, **{key: "1" * 64}))

    def test_content_edits_and_extra_claims_rejected(self):
        for key, value in (("execution_binary_sha256", "1" * 64),
                           ("endpoint_numerical_equivalence_measured", True),
                           ("unreviewed_extra", "accepted")):
            data = copy.deepcopy(br.REVIEW_DATA)
            data[key] = value
            self.review_path.write_text(json.dumps(data))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "复核内容"):
                br.load_review(self.review_path, self.anchor)

    def test_format_change_after_freeze_rejected(self):
        record = self.record()
        self.review_path.write_text(json.dumps(br.REVIEW_DATA))
        with self.assertRaisesRegex(ValueError, "冻结后"):
            br.validate_review(record, self.anchor)
        with self.assertRaises(ValueError):
            br.assert_build_pair(br.ANCHOR_BINARY_SHA256, br.EXECUTION_BINARY_SHA256, record)

    def test_record_edit_and_deleted_review_rejected(self):
        record = self.record()
        bad = copy.deepcopy(record)
        bad["data"]["live_candidate_validation_required"] = False
        with self.assertRaises(ValueError):
            br.validate_review(bad, self.anchor)
        self.review_path.unlink()
        with self.assertRaises(ValueError):
            br.validate_review(record, self.anchor)

    def test_select_records_actual_binary_and_preserves_original_hash(self):
        before = copy.deepcopy(self.anchor)
        with patch("local_live_binary._hash_file", return_value=br.EXECUTION_BINARY_SHA256):
            selected = select_binary(self.anchor, build_review=self.record())
        self.assertEqual(selected, {"path": str(self.binary), "sha256": br.EXECUTION_BINARY_SHA256})
        self.assertEqual(before, self.anchor)

    def test_select_rejects_third_binary_or_missing_execute_permission(self):
        record = self.record()
        with patch("local_live_binary._hash_file", return_value="5" * 64):
            with self.assertRaises(ValueError):
                select_binary(self.anchor, build_review=record)
        self.binary.chmod(0o600)
        with patch("local_live_binary._hash_file", return_value=br.EXECUTION_BINARY_SHA256):
            with self.assertRaisesRegex(ValueError, "执行权限"):
                select_binary(self.anchor, build_review=record)

    def test_select_keeps_toctou_hash_failure(self):
        with patch("local_live_binary._hash_file", side_effect=ValueError("文件在计算 SHA256 期间发生变化")):
            with self.assertRaisesRegex(ValueError, "发生变化"):
                select_binary(self.anchor, build_review=self.record())


if __name__ == "__main__":
    unittest.main()
