import unittest
from pathlib import Path


class TestRLSPredVarGateSource(unittest.TestCase):
    def test_pair_pred_var_resolver_prefers_h1_group_and_fallback(self):
        source = Path('monitor_for_vps.py').read_text()
        self.assertIn('def _resolve_pair_pred_variance(', source)
        self.assertIn('tf_group_key = f"{str(timeframe).upper()}::{pair_group}"', source)
        self.assertIn('for key in (pair_group, tf_group_key):', source)
        self.assertIn('return float("inf")', source)

    def test_entry_gate_uses_resolver_and_parameterized_pred_var_limit(self):
        source = Path('monitor_for_vps.py').read_text()
        self.assertIn('pair_pred_var = _resolve_pair_pred_variance(rls_metrics, pair_group, timeframe="H1")', source)
        self.assertIn('pred_var_gate = float(getattr(parameter, "RLS_MAX_PRED_VARIANCE_FOR_ENTRY", 25.0))', source)

    def test_block_signal_typo_alias_kept_backward_compatible(self):
        parameter_source = Path('parameter.py').read_text()
        self.assertIn('BLOCK_SIGNAL_FOR =', parameter_source)
        self.assertIn('BLOK_SIGNAL_FOR = BLOCK_SIGNAL_FOR', parameter_source)

        engine_source = Path('trade_engine.py').read_text()
        self.assertIn("getattr(parameter, 'BLOCK_SIGNAL_FOR', getattr(parameter, 'BLOK_SIGNAL_FOR', set()))", engine_source)
        self.assertIn('disabled in BLOCK_SIGNAL_FOR', engine_source)


if __name__ == '__main__':
    unittest.main()
