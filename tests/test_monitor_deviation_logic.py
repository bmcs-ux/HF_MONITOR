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

    def test_dynamic_position_tp_guard_exists(self):
        self.assertIn('def _compute_dynamic_position_tp(', self.source)
        self.assertIn('entry_price: float', self.source)
        self.assertIn('min_target_dist = max(abs(float(sl_dist)) * tp_rr_adj, 1e-6)', self.source)
        self.assertIn('tp_floor = max(float(latest_actual_price), float(entry_price)) + min_target_dist', self.source)
        self.assertIn('tp_ceiling = min(float(latest_actual_price), float(entry_price)) - min_target_dist', self.source)

    def test_position_modification_uses_dynamic_tp_guard(self):
        self.assertIn('new_tp = _compute_dynamic_position_tp(', self.source)
        self.assertIn('entry_price = float(getattr(pos, "price_open", latest_actual_price) or latest_actual_price)', self.source)


if __name__ == '__main__':
    unittest.main()
