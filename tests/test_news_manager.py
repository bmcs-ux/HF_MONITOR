import sys
import types
import unittest
from datetime import datetime, timedelta, timezone

if "requests" not in sys.modules:
    sys.modules["requests"] = types.ModuleType("requests")

from news_manager import NewsManager


class TestNewsManager(unittest.TestCase):
    def setUp(self):
        self.logs = []
        self.manager = NewsManager(data_dir='.', logger=self.logs.append)

    def test_get_news_status_exposes_countdown_for_next_high_impact_event(self):
        future_time = (datetime.now(timezone.utc) + timedelta(minutes=7)).isoformat()
        self.manager.high_impact_events = [
            {"title": "NFP", "country": "USD", "date": future_time, "impact": "High"}
        ]

        status = self.manager.get_news_status()

        self.assertFalse(status["is_restricted"])
        self.assertEqual(status["next_event"]["title"], "NFP")
        self.assertGreater(status["seconds_to_next_event"], 0)

    def test_is_currently_restricted_true_when_inside_news_window(self):
        now_time = datetime.now(timezone.utc).isoformat()
        self.manager.high_impact_events = [
            {"title": "CPI", "country": "USD", "date": now_time, "impact": "High"}
        ]

        self.assertTrue(self.manager.is_currently_restricted())


if __name__ == '__main__':
    unittest.main()
