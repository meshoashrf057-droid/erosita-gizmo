"""Synthetic binary discovery tests; never execute any candidate."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_live_binary import audit_binaries, select_binary


class BinaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = b"\x7fELF old synthetic program; do not execute"
        self.current = self.root / "GIZMO"
        self.current.write_bytes(self.old); self.current.chmod(0o700)
        self.anchor = dict(gizmo_path=str(self.current), gizmo_sha256=hashlib.sha256(self.old).hexdigest())

    def backup(self, name="GIZMO_t10", executable=True):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(self.old);p.chmod(0o700 if executable else 0o600)
        return p

    def test_original_and_override_preserve_anchor(self):
        before=copy.deepcopy(self.anchor)
        self.assertEqual(select_binary(self.anchor)["path"],str(self.current))
        saved=self.backup();self.current.write_bytes(b"\x7fELF changed")
        result=select_binary(self.anchor,saved)
        self.assertEqual(result,dict(path=str(saved),sha256=self.anchor["gizmo_sha256"]))
        self.assertEqual(self.anchor,before)

    def test_changed_binary_error_lists_fingerprints(self):
        self.current.write_bytes(b"\x7fELF changed")
        with self.assertRaises(ValueError) as cm:select_binary(self.anchor)
        error=str(cm.exception)
        self.assertIn(self.anchor["gizmo_sha256"],error)
        self.assertIn(hashlib.sha256(self.current.read_bytes()).hexdigest(),error)
        self.assertIn("binary-check",error)

    def test_nonexecutable_and_missing_cannot_select(self):
        saved=self.backup(executable=False)
        with self.assertRaisesRegex(ValueError,"执行权限"):select_binary(self.anchor,saved)
        with self.assertRaisesRegex(ValueError,"无法读取"):select_binary(self.anchor,self.root/"absent")

    def test_scan_finds_exact_deduplicates_symlink_files_and_is_readonly(self):
        saved=self.backup("old/gizmo_t10")
        alias=self.root/"GIZMO_alias";alias.symlink_to(saved)
        self.current.write_bytes(b"\x7fELF changed")
        before={p:(p.read_bytes(),p.stat().st_mode) for p in (self.current,saved)}
        report=audit_binaries(self.anchor,[self.root],[saved,alias])
        self.assertEqual(report["exact_match_paths"],[str(saved)])
        self.assertEqual(len([r for r in report["candidates"] if r["path"]==str(saved)]),1)
        self.assertTrue(report["scan_complete"])
        self.assertEqual(before,{p:(p.read_bytes(),p.stat().st_mode) for p in before})

    def test_explicit_arbitrary_name_nonexecutable_match(self):
        saved=self.backup("old-program",False)
        report=audit_binaries(self.anchor,[],[saved])
        self.assertIn(str(saved),report["exact_match_nonexecutable_paths"])
        self.assertNotIn(str(saved),report["exact_match_paths"])

    def test_not_elf_not_fully_hashed(self):
        self.current.write_bytes(b"#!/bin/sh\nexit 42\n")
        with patch("local_live_binary._hash_file",side_effect=AssertionError("must not hash non-ELF")):
            report=audit_binaries(self.anchor)
        self.assertEqual(report["candidates"][0]["status"],"not_elf")
        self.assertIsNone(report["candidates"][0]["sha256"])

    def test_directory_symlink_not_followed(self):
        outside=self.root/"outside";outside.mkdir()
        saved=self.backup("outside/GIZMO_saved")
        scan=self.root/"scan";scan.mkdir();(scan/"link").symlink_to(outside,target_is_directory=True)
        self.current.write_bytes(b"\x7fELF different")
        report=audit_binaries(self.anchor,[scan])
        self.assertNotIn(str(saved),[r["path"] for r in report["candidates"]])
        self.assertFalse(report["follows_directory_symlinks"])

    def test_missing_root_and_original_reported_not_proof_of_absence(self):
        self.current.unlink()
        report=audit_binaries(self.anchor,[self.root/"missing"])
        self.assertFalse(report["scan_complete"])
        self.assertEqual(report["exact_match_paths"],[])
        self.assertEqual(len(report["errors"]),2)
        self.assertIn("未找到匹配不证明",str(report["notes"]))

    def test_nonmatching_basename_not_scanned_without_explicit_candidate(self):
        saved=self.backup("snapshot_011.hdf5")
        self.current.write_bytes(b"\x7fELF different")
        report=audit_binaries(self.anchor,[self.root])
        self.assertNotIn(str(saved),[r["path"] for r in report["candidates"]])
        report=audit_binaries(self.anchor,[self.root],[saved])
        self.assertEqual(report["exact_match_paths"],[str(saved)])

    def test_invalid_expected_hash_rejected(self):
        bad=dict(self.anchor,gizmo_sha256="guess")
        with self.assertRaisesRegex(ValueError,"有效"):select_binary(bad)
        with self.assertRaisesRegex(ValueError,"有效"):audit_binaries(bad)


if __name__=="__main__":unittest.main()
