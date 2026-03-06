import unittest
from pathlib import Path


class TestMonitorDeviationLogicSource(unittest.TestCase):
    def setUp(self):
        self.source = Path('monitor_for_vps.py').read_text()

    def test_new_model_based_deviation_function_exists(self):
        self.assertIn('def _estimate_forecast_std(', self.source)
        self.assertIn('No model forecast found', self.source)
        self.assertIn('model forecast (RLS + Kalman)', self.source)

    def test_exogenous_alias_fallback_exists(self):
        self.assertIn("normalized_exog_name = str(exog_name).replace('_Transformed', '')", self.source)
        self.assertIn('alias: {normalized_exog_name}', self.source)


if __name__ == '__main__':
    unittest.main()
