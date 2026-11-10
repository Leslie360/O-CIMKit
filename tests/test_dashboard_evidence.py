import unittest
from pathlib import Path

from scripts.export_dashboard_data import read_reliability_report


class TestDashboardEvidence(unittest.TestCase):
    def test_reliability_values_come_from_source_report(self):
        project_root = Path(__file__).resolve().parents[1]
        evidence = read_reliability_report(project_root / "reports" / "top_journal_benchmark_report.md")
        self.assertEqual(evidence["labels"], ["全新器件", "1 个月", "10 年"])
        self.assertEqual(evidence["methods"][1]["values"][-1], 86.67)
        self.assertEqual(evidence["methods"][3]["values"][-1], 100.0)
        self.assertIn("not physical ten-year operation", evidence["scope"])


if __name__ == "__main__":
    unittest.main()
