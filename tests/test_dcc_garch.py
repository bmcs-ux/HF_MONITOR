import unittest
import numpy as np

from fitted_models.dcc_garch import DCCGARCH


class TestDCCGARCH(unittest.TestCase):
    def test_fit_and_forecast_shape(self):
        np.random.seed(7)
        T, N = 120, 3
        eps = np.random.normal(0.0, 0.01, size=(T, N))

        model = DCCGARCH()
        out = model.fit(eps, column_names=["A", "B", "C"], disp=False)

        self.assertIn("H", out)
        self.assertEqual(out["H"].shape, (T, N, N))

        H_next = model.forecast(horizon=1)
        self.assertEqual(H_next.shape, (N, N))
        self.assertTrue(np.all(np.diag(H_next) > 0))


if __name__ == "__main__":
    unittest.main()
