import ast
import unittest
from pathlib import Path


class TestConfigWiring(unittest.TestCase):
    def _load_ast(self, rel_path: str):
        return ast.parse(Path(rel_path).read_text(), filename=rel_path)

    def test_trade_engine_uses_parameter_credentials_and_keys(self):
        tree = self._load_ast('trade_engine.py')
        assigns = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                assigns[node.targets[0].id] = node.value

        required = {
            'MT5_LOGIN': 'parameter.MT5_LOGIN',
            'MT5_PASSWORD': 'parameter.MT5_PASSWORD',
            'MT5_SERVER': 'parameter.MT5_SERVER',
            'TRADE_ENGINE_API_KEY': 'parameter.TRADE_ENGINE_API_KEY',
            'COLAB_API_KEY_FOR_TRADE_ENGINE': 'parameter.COLAB_API_KEY_FOR_TRADE_ENGINE',
        }
        for var, dotted in required.items():
            self.assertIn(var, assigns)
            value = assigns[var]
            self.assertIsInstance(value, ast.Attribute)
            self.assertIsInstance(value.value, ast.Name)
            self.assertEqual(f"{value.value.id}.{value.attr}", dotted)

    def test_monitor_uses_parameter_signal_entry_config(self):
        source = Path('monitor_for_vps.py').read_text()
        self.assertIn('TE_API_KEY = parameter.TRADE_ENGINE_API_KEY', source)
        self.assertIn('TRADE_ENGINE_API_URL = parameter.TRADE_ENGINE_API_URL', source)

    def test_parameter_model_paths_portable(self):
        source = Path('parameter.py').read_text()
        self.assertIn("FORECAST_OUTPUT_PATH = os.path.join(ROOT_DIR, 'restored_forecasts.pkl')", source)
        self.assertIn("FRED_DATA_PATH = os.path.join(ROOT_DIR, 'final_fred_data.pkl')", source)
        self.assertIn("FITTED_MODELS_PATH = os.path.join(ROOT_DIR, 'fitted_ensemble.pkl')", source)


if __name__ == '__main__':
    unittest.main()
