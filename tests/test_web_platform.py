import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from applications.web_platform.engine import PlatformEngine, build_result_analysis


class TestWebPlatform(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.directory = TemporaryDirectory()
        self.engine = PlatformEngine(self.project_root, jobs_root=self.directory.name)

    def tearDown(self):
        self.engine.close()
        self.directory.cleanup()

    def test_example_prepares_real_profile_without_touching_repository_profile(self):
        repository_profile = self.project_root / "profiles" / "repository" / "XJTU_Measured_Memristor.json"
        before = repository_profile.read_bytes()
        prepared = self.engine.prepare_example()
        self.assertEqual(prepared["status"], "prepared")
        self.assertEqual(prepared["profile"]["dynamic_parameters"]["discrete_states_count"], 50)
        self.assertEqual(len(prepared["measured"]["devices"]), 12)
        self.assertEqual(len(prepared["measured"]["mean"]), 100)
        self.assertEqual(repository_profile.read_bytes(), before)
        self.assertFalse((self.engine.job_directory(prepared["id"]) / "report.json").exists())

    def test_invalid_upload_and_training_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.prepare_upload("bad.exe", b"not a workbook")
        prepared = self.engine.prepare_example()
        with self.assertRaises(ValueError):
            self.engine.start_training(prepared["id"], epochs=0)
        with self.assertRaises(ValueError):
            self.engine.start_training(prepared["id"], validation="unknown")

    def test_analysis_separates_fold_mean_from_pooled_predictions(self):
        report = json.loads((self.project_root / "reports" / "xjtu_measured_ecg_patient_group_results.json").read_text())
        analysis = build_result_analysis(report)
        self.assertTrue(any("五折均值对各折等权" in statement for statement in analysis))
        self.assertTrue(any("75.53%" in statement for statement in analysis))
        self.assertTrue(any("异常心搏召回率" in statement for statement in analysis))


if __name__ == "__main__":
    unittest.main()
