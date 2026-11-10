import unittest

import numpy as np

from applications.ecg_cardio.train import make_validation_splits, train_and_evaluate_ecg
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


if __name__ == "__main__":
    unittest.main()
