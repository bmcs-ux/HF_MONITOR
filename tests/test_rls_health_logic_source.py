import unittest
from pathlib import Path


class TestRLSHealthLogicSource(unittest.TestCase):
    def setUp(self):
        self.source = Path('monitor_for_vps.py').read_text()

    def test_confidence_helper_exists_and_uses_warmup_factor(self):
        self.assertIn('def _compute_rls_confidence(', self.source)
        self.assertIn('warmup_factor = 0.25 + (0.75 * maturity)', self.source)
        self.assertIn('normalized_uncertainty = pred_variance / variance_ref', self.source)

    def test_global_confidence_is_maturity_weighted(self):
        self.assertIn('def _summarize_rls_global_metrics(', self.source)
        self.assertIn('weighted_conf_sum += float(confidence) * maturity', self.source)
        self.assertIn('global_confidence = weighted_conf_sum / weight_sum', self.source)

    def test_deviation_gate_typo_is_fixed(self):
        self.assertIn('parameter._RLS_DEVIATION_THRESHOLD', self.source)
        self.assertNotIn('_RLS_DEVIATION_TRESHOLD', self.source)

    def test_timeframe_marker_uses_estimator_specific_key(self):
        self.assertIn('marker_key: Optional[str] = None', self.source)
        self.assertIn('key = marker_key or tf_name', self.source)
        self.assertIn('marker_key=estimator_label', self.source)

    def test_kalman_trigger_and_rls_confirmation_gate_exists(self):
        self.assertIn('kalman_velocity_threshold = float(getattr(parameter, "KALMAN_VELOCITY_THRESHOLD", 1e-6))', self.source)
        self.assertIn('kalman_entry_zscore = float(getattr(parameter, "KALMAN_ENTRY_ZSCORE", 0.25))', self.source)
        self.assertIn('RLS confirmation failed (ret=', self.source)

    def test_position_modification_uses_kalman_flip_not_rls_flip(self):
        self.assertIn('close_due_to_kalman_flip', self.source)
        self.assertIn('Kalman Flip z=', self.source)
        self.assertNotIn('close_due_to_rls_flip', self.source)

    def test_position_modification_new_tp_is_kalman_based(self):
        self.assertIn('kalman_projected_tp = float(kalman_result.get("filtered_price", latest_actual_price)) + float(kalman_result.get("velocity", 0.0))', self.source)
        self.assertIn('Target profit dinamis dari Kalman', self.source)


if __name__ == '__main__':
    unittest.main()
