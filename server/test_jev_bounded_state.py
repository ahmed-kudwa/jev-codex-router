import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import jev_server


class JevBoundedStateTest(unittest.TestCase):
    def test_advisor_packet_keeps_only_bounded_recent_state(self):
        old_task = "old task history " * 1000
        current_task = ("current task " * 100) + " TASK_END"
        previous = ("previous assistant history " * 100) + " PREVIOUS_END"
        tool_output = ("tool output history " * 1000) + " TOOL_END"
        signals = {
            "context_chars": 999999,
            "context_items": 999,
            "step_type": "tool_step",
        }
        step = {
            "step_type": "tool_step",
            "digest": tool_output,
            "errored": True,
            "deepseek_replay": False,
        }
        catalog = {
            jev_server.LUNA: {"efforts": ["low", "medium", "high"]},
            jev_server.SOL: {"efforts": ["low", "medium", "high"]},
        }

        packet = jev_server.build_jev_state(
            old_task + current_task,
            previous,
            signals,
            step,
            catalog,
            "",
            "phase-cache-key",
        )

        self.assertEqual(packet["task"], (old_task + current_task)[:500])
        self.assertEqual(packet["previous_assistant"], previous[-240:])
        self.assertEqual(packet["step"]["last_tool_output_tail"], tool_output[-jev_server.DIGEST_CHARS:])
        self.assertTrue(packet["step"]["contains_error"])
        self.assertNotIn("old task history", json.dumps(packet["step"]))
        self.assertNotIn("tool output history", json.dumps(packet["signals"]))
        self.assertLessEqual(len(packet["task"]), 500)
        self.assertLessEqual(len(packet["previous_assistant"]), 240)
        self.assertLessEqual(
            len(packet["step"]["last_tool_output_tail"]),
            jev_server.DIGEST_CHARS,
        )


if __name__ == "__main__":
    unittest.main()
