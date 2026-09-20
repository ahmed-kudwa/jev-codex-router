import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import jev_server


class LocalRouterRetryableTest(unittest.TestCase):
    def test_overlay_cost_cap_is_retryable(self):
        self.assertEqual(
            jev_server.local_router_retryable(
                402,
                '{"error":{"message":"Monthly total cost cap reached."}}',
            ),
            "cost_cap",
        )

    def test_overlay_missing_policy_502_is_retryable(self):
        self.assertEqual(
            jev_server.local_router_retryable(
                502,
                "The local router could not complete the request.",
            ),
            "gateway",
        )

    def test_missing_cost_policy_body_is_retryable(self):
        self.assertEqual(
            jev_server.local_router_retryable(
                500,
                "Error: Default cost-policy.json is missing from the overlay.",
            ),
            "gateway",
        )

    def test_usage_limit_is_quota(self):
        self.assertEqual(
            jev_server.local_router_retryable(429, "hit your usage limit"),
            "quota",
        )

    def test_ordinary_400_is_not_retryable(self):
        self.assertIsNone(
            jev_server.local_router_retryable(400, "unknown model gpt-nope"),
        )


if __name__ == "__main__":
    unittest.main()
