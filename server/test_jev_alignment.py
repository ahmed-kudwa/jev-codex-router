from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import jev_server


class AlignmentSnapshotTest(unittest.TestCase):
    def test_snapshot_reports_bounded_dataset_metadata(self):
        with tempfile.TemporaryDirectory(prefix="jev-align-status-") as directory:
            path = os.path.join(directory, "routing-dataset.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "task_summary"])
                writer.writerow(["trace-1", "safe summary"])
            previous = jev_server.ALIGNMENT_DATASET_PATH
            try:
                jev_server.ALIGNMENT_DATASET_PATH = path
                snapshot = jev_server.alignment_snapshot()
            finally:
                jev_server.ALIGNMENT_DATASET_PATH = previous
        self.assertTrue(snapshot["datasetExists"])
        self.assertEqual(snapshot["datasetRows"], 1)
        self.assertGreater(snapshot["datasetBytes"], 0)
        self.assertEqual(snapshot["mode"], "offline_human_labeling")


if __name__ == "__main__":
    unittest.main()
