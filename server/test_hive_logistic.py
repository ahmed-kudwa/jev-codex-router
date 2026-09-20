import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import hive_logistic


class HiveLogisticTest(unittest.TestCase):
    def test_shadow_model_updates_and_persists(self):
        with tempfile.TemporaryDirectory(prefix="hive-logistic-test-") as state:
            old_state = hive_logistic.STATE
            old_path = hive_logistic.MODEL_PATH
            try:
                hive_logistic.STATE = state
                hive_logistic.MODEL_PATH = os.path.join(state, "hive-logistic.json")
                features = {
                    "route_class": "routine",
                    "model": "gpt-5.6-luna",
                    "effort": "medium",
                    "confidence": 0.8,
                    "step_type": "user_turn",
                    "context_chars": 5000,
                    "context_items": 12,
                    "errored": False,
                    "deepseek_replay": False,
                    "phase_cacheable": True,
                    "reason": "mechanical_keyword",
                }
                before = hive_logistic.predict(features)
                for _ in range(20):
                    hive_logistic.update(features, 200)
                after = hive_logistic.predict(features)
                snapshot = hive_logistic.snapshot()
                self.assertGreater(after, before)
                self.assertEqual(snapshot["examples"], 20)
                self.assertEqual(snapshot["positive"], 20)
                self.assertTrue(os.path.exists(hive_logistic.MODEL_PATH))
            finally:
                hive_logistic.STATE = old_state
                hive_logistic.MODEL_PATH = old_path


if __name__ == "__main__":
    unittest.main()
