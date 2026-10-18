"""Protect original inference artifacts and verify saved-score provenance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.project2_reanalyze import reanalyze


class ReanalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "inference"
        self.output = Path(self.temporary.name) / "reanalysis"
        self.source.mkdir()
        self.scores = b'{"synthetic": "provenance fixture; analyzer is mocked"}\n'
        (self.source / "pair_scores.jsonl").write_bytes(self.scores)
        self.manifest = {"status": "completed", "output_sha256": {
            "pair_scores.jsonl": hashlib.sha256(self.scores).hexdigest()}}
        self.write_manifest()

    def write_manifest(self):
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))

    def test_success_preserves_source_and_records_hashes_of_new_outputs(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}

        def fake_analyze(directory):
            (directory / "evaluation.json").write_text('{"test": true}\n')

        with patch("scripts.project2_reanalyze.analyze", side_effect=fake_analyze) as analyze:
            result = reanalyze(self.source, self.output)
        analyze.assert_called_once_with(self.output.resolve())
        self.assertEqual(result["status"], "completed")
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, before)
        self.assertEqual((self.output / "pair_scores.jsonl").read_bytes(), self.scores)
        self.assertEqual((self.output / "source_manifest.json").read_bytes(), before["manifest.json"])
        self.assertEqual(result["source_manifest_sha256"], hashlib.sha256(before["manifest.json"]).hexdigest())
        self.assertIn("src/project2_evaluation.py", result["analysis_source_sha256"])
        for name, digest in result["output_sha256"].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), digest)

    def test_rejects_modified_scores_before_creating_output_or_analyzing(self):
        (self.source / "pair_scores.jsonl").write_bytes(self.scores + b"tampered")
        with patch("scripts.project2_reanalyze.analyze") as analyze, self.assertRaisesRegex(ValueError, "does not match"):
            reanalyze(self.source, self.output)
        analyze.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_rejects_incomplete_inference_and_missing_hash(self):
        for manifest in ({"status": "started", "output_sha256": self.manifest["output_sha256"]},
                         {"status": "completed", "output_sha256": {}}):
            self.manifest = manifest
            self.write_manifest()
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                reanalyze(self.source, self.output)
            self.assertFalse(self.output.exists())

    def test_rejects_source_or_nested_output_and_preserves_existing_outputs(self):
        self.output.mkdir()
        (self.output / "keep.txt").write_text("previous result")
        for output in (self.source, self.source / "nested", self.output):
            with self.subTest(output=output), self.assertRaises(ValueError):
                reanalyze(self.source, output)
        self.assertEqual((self.output / "keep.txt").read_text(), "previous result")
        self.assertFalse((self.source / "nested").exists())

    def test_analysis_failure_gets_failed_manifest_without_modifying_source(self):
        with patch("scripts.project2_reanalyze.analyze", side_effect=ValueError("invalid pair metadata")):
            with self.assertRaisesRegex(ValueError, "invalid pair metadata"):
                reanalyze(self.source, self.output)
        manifest = json.loads((self.output / "analysis_manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("invalid pair metadata", manifest["error"])
        self.assertEqual((self.source / "pair_scores.jsonl").read_bytes(), self.scores)
        self.assertNotIn("output_sha256", manifest)


if __name__ == "__main__":
    unittest.main()
