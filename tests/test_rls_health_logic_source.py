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


if __name__ == '__main__':
    unittest.main()
