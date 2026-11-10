import json
import os
import tempfile
import unittest
from unittest.mock import patch

from profiles.device_profile import DeviceProfile


class TestDeviceProfileIsolation(unittest.TestCase):
    def test_benchmark_overlay_does_not_modify_source_profile(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        source_path = os.path.join(project_root, "profiles", "repository", "FingerMemristor.json")
        with open(source_path, encoding="utf-8") as handle:
            original = json.load(handle)

        with tempfile.TemporaryDirectory() as overlay_dir:
            overlay = original.copy()
            overlay["device_name"] = "OverlayDevice"
            overlay["fitting_parameters"] = original["fitting_parameters"].copy()
            overlay["fitting_parameters"]["conductance_min"] = 3.0
            overlay_path = os.path.join(overlay_dir, "FingerMemristor.json")
            with open(overlay_path, "w", encoding="utf-8") as handle:
                json.dump(overlay, handle)

            with patch.dict(os.environ, {"OCIMKIT_PROFILE_OVERLAY_DIR": overlay_dir}):
                profile = DeviceProfile.from_json(source_path)
                self.assertEqual(profile.device_name, "OverlayDevice")
                self.assertEqual(profile.conductance_min, 3.0)

            profile = DeviceProfile.from_json(source_path)
            self.assertEqual(profile.device_name, original["device_name"])
            self.assertEqual(profile.conductance_min, original["fitting_parameters"]["conductance_min"])

        with open(source_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), original)


if __name__ == "__main__":
    unittest.main()
