import json
import os
import subprocess
import sys
import tempfile
import unittest

from openpyxl import Workbook

from profiles.device_profile import DeviceProfile


class TestDeviceProfileIngestion(unittest.TestCase):
    def test_multi_sheet_workbook_aggregation_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = os.path.join(temp_dir, "measured.xlsx")
            output_path = os.path.join(temp_dir, "profile.json")
            workbook = Workbook()
            workbook.remove(workbook.active)
            for sheet_name, scale in [("单个1", 1.0), ("单个2", 1.1)]:
                sheet = workbook.create_sheet(sheet_name)
                curve = [1.0, 1.5, 2.0, 2.5, 2.5, 2.0, 1.5, 1.0]
                for pulse, value in enumerate(curve, start=1):
                    sheet.append([pulse, value * scale])
            workbook.save(source_path)

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            subprocess.run([
                sys.executable, os.path.join(project_root, "profiles", "parser.py"),
                "--file", source_path,
                "--name", "MeasuredTest",
                "--type", "nonvolatile",
                "--measurement-type", "current",
                "--output", output_path,
            ], cwd=project_root, check=True, capture_output=True, text=True)

            with open(output_path, encoding="utf-8") as handle:
                data = json.load(handle)
            self.assertEqual(data["dynamic_parameters"]["discrete_states_count"], 4)
            self.assertEqual(data["source_metadata"]["device_count"], 2)
            self.assertEqual(data["source_metadata"]["pulses_per_device"], 8)
            self.assertEqual(data["source_metadata"]["measurement_type"], "current")
            self.assertAlmostEqual(data["fitting_parameters"]["conductance_min"], 1.05)
            self.assertAlmostEqual(data["fitting_parameters"]["conductance_max"], 2.625)
            self.assertGreaterEqual(data["fitting_parameters"]["noise_std_ratio"], 0.0)

            profile = DeviceProfile.from_json(output_path)
            self.assertEqual(profile.source_metadata["source_sheets"], ["单个1", "单个2"])


if __name__ == "__main__":
    unittest.main()
