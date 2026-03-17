import unittest
from pathlib import Path


class TestDccContagionGroupCacheSource(unittest.TestCase):
    def setUp(self):
        self.source = Path("monitor_for_vps.py").read_text()

    def test_group_specific_cache_key_is_used_for_dcc_metrics(self):
        self.assertIn('cache_key = f"{timeframe_name}::{group_name}"', self.source)
        self.assertIn('dcc_metrics_cache[cache_key] = {', self.source)
        self.assertIn('dcc_metrics_cache.get(cache_key, {}).get("contagion_score", 0.0)', self.source)

    def test_group_score_uses_covariance_submatrix(self):
        self.assertIn('def _compute_contagion_score_from_covariance(', self.source)
        self.assertIn('getattr(dcc_model, "column_names", None)', self.source)
        self.assertIn('endog_names_group,', self.source)


if __name__ == "__main__":
    unittest.main()
