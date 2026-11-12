import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from applications.ecg_cardio.train import (
    export_prediction_evidence,
    make_validation_splits,
    train_and_evaluate_ecg,
)
from profiles.device_profile import DeviceProfile


class TestECGValidation(unittest.TestCase):
    def setUp(self):
        self.groups = np.repeat([f"record-{index}" for index in range(15)], 10)
        self.y = np.tile([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 15)
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(len(self.y), 8))

    def test_grouped_folds_never_share_records(self):
        splits = make_validation_splits(self.X, self.y, self.groups, "patient-group-kfold")
        self.assertEqual(len(splits), 5)
        for train_idx, test_idx in splits:
            self.assertTrue(set(self.groups[train_idx]).isdisjoint(set(self.groups[test_idx])))
            self.assertEqual(set(np.unique(self.y[test_idx])), {0, 1})

    def test_grouped_training_saves_fold_evidence(self):
        profile = DeviceProfile(
            name="ValidationDevice", is_volatile=False, discrete_states_count=16,
            conductance_min=1.0, conductance_max=2.0, noise_std_ratio=0.0,
        )
        result = train_and_evaluate_ecg(
            self.X, self.y, out_classes=2, device_profile=profile, hidden_dim=8,
            epochs=1, groups=self.groups, validation="patient-group-kfold",
        )
        self.assertEqual(len(result["folds"]), 5)
        for fold in result["folds"]:
            self.assertEqual(fold["train_record_count"], 12)
            self.assertEqual(fold["test_record_count"], 3)
            self.assertEqual(fold["test_class_counts"], {"0": 24, "1": 6})
            self.assertEqual(len(fold["test_records"]), 3)

    def test_prediction_records_cover_each_held_out_sample_once(self):
        records = []
        train_and_evaluate_ecg(
            self.X, self.y, out_classes=2, hidden_dim=8, epochs=1,
            groups=self.groups, validation="patient-group-kfold",
            prediction_records=records, model_name="float_baseline",
        )
        self.assertEqual(len(records), len(self.y))
        self.assertEqual({record["sample_index"] for record in records}, set(range(len(self.y))))
        for record in records:
            self.assertAlmostEqual(record["normal_score"] + record["abnormal_score"], 1.0, places=6)
            self.assertEqual(record["record_id"], self.groups[record["sample_index"]])

    def test_evidence_export_contains_confusion_and_real_inputs(self):
        records = []
        for sample_index, class_index in enumerate([0, 0, 1, 1]):
            for model_name in ["float_baseline", "measured_device_simulation"]:
                records.append({
                    "sample_index": sample_index,
                    "fold": 1,
                    "record_id": f"record-{sample_index}",
                    "model": model_name,
                    "y_true": class_index,
                    "y_pred": class_index,
                    "normal_score": 0.9 if class_index == 0 else 0.1,
                    "abnormal_score": 0.1 if class_index == 0 else 0.9,
                })
        waveforms = np.arange(4 * 180).reshape(4, 180)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            evidence = export_prediction_evidence(output, records, waveforms)
            self.assertEqual(evidence["models"]["measured_device_simulation"]["support"], [2, 2])
            self.assertEqual(evidence["models"]["measured_device_simulation"]["accuracy_percent"], 100.0)
            self.assertEqual(len(evidence["examples"]), 2)
            self.assertEqual(len(evidence["examples"][0]["waveform"]), 180)
            self.assertTrue((output.parent / "predictions.csv").is_file())
            saved = json.loads((output.parent / "prediction_examples.json").read_text())
            self.assertEqual(saved["selection_rule"], evidence["selection_rule"])


if __name__ == "__main__":
    unittest.main()
