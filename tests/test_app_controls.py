import unittest
from pathlib import Path


class TestAppDashboardWiring(unittest.TestCase):
    def test_dashboard_split_files_exist(self):
        self.assertTrue(Path('templates/dashboard.html').exists())
        self.assertTrue(Path('static/dashboard.css').exists())
        self.assertTrue(Path('static/dashboard.js').exists())

    def test_app_has_control_endpoints_and_monitor_trade_routes(self):
        src = Path('app.py').read_text()
        self.assertIn("@app.route('/api/control/simulate_monitor', methods=['POST'])", src)
        self.assertIn("@app.route('/api/control/simulate_trade', methods=['POST'])", src)
        self.assertIn("@app.route('/api/control/reset', methods=['POST'])", src)
        self.assertIn("@app.route('/update_monitor_data', methods=['POST'])", src)
        self.assertIn("@app.route('/update_trade_data', methods=['POST'])", src)


if __name__ == '__main__':
    unittest.main()
