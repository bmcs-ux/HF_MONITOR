import ast
import types
import unittest
from pathlib import Path


class _NP:
    @staticmethod
    def clip(value, min_value, max_value):
        return max(min_value, min(max_value, value))


class TestRLSConfirmationBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path("monitor_for_vps.py").read_text()
        tree = ast.parse(source)
        wanted = {"_stabilize_expected_return", "_passes_rls_directional_confirmation"}
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        module = ast.Module(body=selected, type_ignores=[])
        code = compile(module, filename="monitor_for_vps.py", mode="exec")

        env = {
            "np": _NP,
            "parameter": types.SimpleNamespace(
                RLS_RETURN_EMA_ALPHA=0.35,
                RLS_RETURN_DEADBAND=5e-5,
                RLS_RETURN_DIRECTION_EPSILON=1e-5,
            ),
        }
        exec(code, env)
        cls.stabilize = staticmethod(env["_stabilize_expected_return"])
        cls.direction_ok = staticmethod(env["_passes_rls_directional_confirmation"])

    def test_directional_confirmation_is_side_aware(self):
        self.assertTrue(self.direction_ok("BUY", 1e-4))
        self.assertFalse(self.direction_ok("BUY", -1e-4))
        self.assertTrue(self.direction_ok("SELL", -1e-4))
        self.assertFalse(self.direction_ok("SELL", 1e-4))

    def test_stabilize_expected_return_applies_deadband(self):
        stabilized = self.stabilize(1e-5, 0.0)
        self.assertEqual(stabilized, 0.0)

    def test_stabilize_expected_return_keeps_direction_when_signal_is_clear(self):
        stabilized = self.stabilize(-4e-4, -3e-4)
        self.assertLess(stabilized, 0.0)


if __name__ == "__main__":
    unittest.main()
