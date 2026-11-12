import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.download_ecg_data import RECORD_IDS, REQUIRED_SUFFIXES, prepare_ecg_data


class TestECGDataPreparation(unittest.TestCase):
    def test_downloader_must_produce_all_records_and_annotations(self):
        with TemporaryDirectory() as directory:
            calls = []

            def download(name, target, **kwargs):
                calls.append((name, target, kwargs))
                for record in RECORD_IDS:
                    for suffix in REQUIRED_SUFFIXES:
                        Path(target, record + suffix).touch()

            prepare_ecg_data(directory, downloader=download)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], "mitdb")
            self.assertEqual(calls[0][2]["records"], RECORD_IDS)
            self.assertEqual(calls[0][2]["annotators"], ["atr"])
            self.assertFalse(calls[0][2]["keep_subdirs"])

    def test_retry_and_incomplete_download_are_explicit(self):
        with TemporaryDirectory() as directory, patch("scripts.download_ecg_data.time.sleep"):
            calls = []

            def download(*args, **kwargs):
                calls.append(1)

            with self.assertRaises(RuntimeError):
                prepare_ecg_data(directory, downloader=download)
            self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
