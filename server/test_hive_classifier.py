import os
import tempfile
import unittest

import sys

sys.path.insert(0, os.path.dirname(__file__))
import hive_classifier


class HiveClassifierTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hive-classifier-test-")
        self.old = {
            name: getattr(hive_classifier, name)
            for name in ("STATE", "POLICY_PATH", "EVENTS_PATH", "CACHE_PATH", "CACHE_MAINTENANCE_PATH")
        }
        hive_classifier.STATE = self.tmp.name
        hive_classifier.POLICY_PATH = os.path.join(self.tmp.name, "hive-policy.json")
        hive_classifier.EVENTS_PATH = os.path.join(self.tmp.name, "hive-events.jsonl")
        hive_classifier.CACHE_PATH = os.path.join(self.tmp.name, "hive-jev-cache.json")
        hive_classifier.CACHE_MAINTENANCE_PATH = os.path.join(self.tmp.name, "hive-maintenance.json")

    def tearDown(self):
        for name, value in self.old.items():
            setattr(hive_classifier, name, value)
        self.tmp.cleanup()

    def test_mechanical_route_keeps_luna_until_deepseek_is_promoted(self):
        catalog = {
            hive_classifier.LUNA: {"efforts": ["low", "medium"]},
            hive_classifier.DEEPSEEK: {"efforts": ["low", "medium"]},
        }
        step = {
            "step_type": "tool_step",
            "errored": False,
            "digest": "short output",
            "deepseek_replay": True,
        }
        first = hive_classifier.classify("run the prepared check", step, catalog)
        self.assertEqual(first["model"], hive_classifier.LUNA)

        policy = hive_classifier._load_policy()
        policy["adaptive"]["recommendations"]["mechanical"] = {
            "model": hive_classifier.DEEPSEEK,
            "promoted": True,
            "attempts": 8,
            "successRate": 1.0,
        }
        hive_classifier._save_policy(policy)
        promoted = hive_classifier.classify("run the prepared check", step, catalog)
        self.assertEqual(promoted["model"], hive_classifier.DEEPSEEK)

    def test_degraded_recommended_route_is_not_selected(self):
        catalog = {
            hive_classifier.LUNA: {"efforts": ["low", "medium"]},
            hive_classifier.DEEPSEEK: {"efforts": ["low", "medium"]},
        }
        policy = hive_classifier._load_policy()
        policy["adaptive"]["recommendations"]["routine"] = {
            "model": hive_classifier.DEEPSEEK,
            "promoted": True,
            "attempts": 8,
            "successRate": 1.0,
        }
        policy["models"][hive_classifier.DEEPSEEK] = {
            "success": 1,
            "failure": 4,
            "quarantinedUntil": 0,
        }
        hive_classifier._save_policy(policy)
        decision = hive_classifier.classify(
            "run the prepared check",
            {"step_type": "user_turn", "errored": False, "digest": ""},
            catalog,
        )
        self.assertEqual(decision["model"], hive_classifier.LUNA)


if __name__ == "__main__":
    unittest.main()
