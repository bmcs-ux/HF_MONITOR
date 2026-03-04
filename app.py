from flask import Flask, jsonify, render_template, request
import datetime

app = Flask(__name__)

# Penyimpanan data
vps_data_store = {
    "logs": [],
    "assets": {},
    "signals": {},
    "rls_forecast": {},
    "rls_health": {},
    "latest_actual_prices": {},
    "trade_signals": {},
    "parameter_deviations": {},
    "global_metrics": {},
    "dcc_metrics": {},
    "kalman_metrics": {},
    "consensus_metrics": {},
    "mean_reversion_candidates": {},
    "open_trades_summary": [],
    "summary": {"status": "WAITING", "time": "-"},
    "equity_history": [],
    "financials": {
        "equity": 0,
        "equity_peak": 0,
        "peak": 0,
        "drawdown": 0,
        "daily_loss_pct": 0,
        "trading_enabled": True,
    },
}


def _append_log(data: dict):
    vps_data_store["logs"].insert(0, data)
    if len(vps_data_store["logs"]) > 50:
        vps_data_store["logs"].pop()


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/get_data')
def get_data():
    return jsonify(vps_data_store)


@app.route('/api/control/reset', methods=['POST'])
def reset_data():
    vps_data_store["logs"].clear()
    vps_data_store["assets"].clear()
    vps_data_store["signals"].clear()
    vps_data_store["rls_forecast"].clear()
    vps_data_store["rls_health"].clear()
    vps_data_store["latest_actual_prices"].clear()
    vps_data_store["trade_signals"].clear()
    vps_data_store["parameter_deviations"].clear()
    vps_data_store["global_metrics"].clear()
    vps_data_store["dcc_metrics"].clear()
    vps_data_store["kalman_metrics"].clear()
    vps_data_store["consensus_metrics"].clear()
    vps_data_store["mean_reversion_candidates"].clear()
    vps_data_store["open_trades_summary"].clear()
    vps_data_store["summary"] = {"status": "WAITING", "time": "-"}
    return jsonify({"status": "success"}), 200


@app.route('/api/control/simulate_monitor', methods=['POST'])
def simulate_monitor_data():
    now = datetime.datetime.now().isoformat()
    sample = {
        "timestamp": now,
        "cycle_number": 999,
        "latest_actual_prices": {"XAU/USD": 2345.11, "USD/JPY": 154.22},
        "rls_forecast": {"XAU/USD": 2349.21, "USD/JPY": 154.01},
        "rls_health": {
            "H1::Commodities": {"confidence": 0.74, "maturity": 0.91, "deviation": 0.22, "pred_var": 0.11},
            "H1::FX_Majors": {"confidence": 0.61, "maturity": 0.82, "deviation": 0.31, "pred_var": 0.17},
        },
        "trade_signals": {
            "XAU/USD": {"signal": "BUY", "entry_price": 2345.11, "stop_loss": 2339.0, "take_profit": 2355.0, "snr": 1.9},
            "USD/JPY": {"signal": "SELL", "entry_price": 154.22, "stop_loss": 154.8, "take_profit": 153.5, "snr": 1.4},
        },
        "parameter_deviations": {"H1::Commodities": 0.22, "H1::FX_Majors": 0.31},
        "global_metrics": {"global_confidence": 0.67, "global_deviation": 0.26, "cycle_duration": 1.22},
        "dcc_metrics": {"H1::Commodities": {"contagion_score": 0.44}},
        "kalman_metrics": {"XAU/USD": {"trend": "UP", "innovation_zscore": 1.2}},
        "consensus_metrics": {"XAU/USD": {"consensus_score": 0.72}},
        "mean_reversion_candidates": {},
    }
    _append_log(sample)
    receive_data_internal(sample)
    return jsonify({"status": "success"}), 200


@app.route('/api/control/simulate_trade', methods=['POST'])
def simulate_trade_data():
    payload = {
        "equity_data": {
            "equity": 1085.20,
            "equity_peak": 1120.0,
            "drawdown": 3.11,
            "daily_loss_pct": 1.22,
            "trading_enabled": True,
        },
        "open_trades_summary": [
            {"symbol": "XAUUSDm", "type": "BUY", "volume": 0.01, "profit": 4.2},
            {"symbol": "USDJPYm", "type": "SELL", "volume": 0.01, "profit": -1.3},
        ],
    }
    receive_trade_data_internal(payload)
    return jsonify({"status": "success"}), 200


def receive_data_internal(data: dict):
    new_health = data.get('rls_health')
    if new_health:
        vps_data_store['rls_health'].update(new_health)

    if 'deviation_results' in data:
        for sym, val in data['deviation_results'].items():
            if isinstance(val, dict) and 'actual_price' in val:
                vps_data_store['latest_actual_prices'][sym] = val['actual_price']

    new_prices = data.get('latest_actual_prices')
    if new_prices:
        vps_data_store['latest_actual_prices'].update(new_prices)

    incoming_rls = data.get('rls_forecast') or data.get('rls_forecasts')
    if incoming_rls:
        for sym, val in incoming_rls.items():
            if isinstance(val, dict):
                p = val.get('rls_predicted_price') or val.get('predicted_price')
                if p is not None:
                    vps_data_store['rls_forecast'][sym] = p
            else:
                vps_data_store['rls_forecast'][sym] = val

    for key in [
        'trade_signals', 'parameter_deviations', 'global_metrics', 'dcc_metrics',
        'kalman_metrics', 'consensus_metrics', 'mean_reversion_candidates'
    ]:
        new_val = data.get(key)
        if new_val:
            vps_data_store[key] = new_val

    if data.get('trade_signals'):
        vps_data_store['signals'] = data['trade_signals']

    vps_data_store['summary'] = {
        "status": "RUNNING",
        "time": data.get('timestamp', datetime.datetime.now().isoformat())
    }


@app.route('/update_monitor_data', methods=['POST'])
def receive_data():
    data = request.json
    if not data:
        return jsonify({"status": "error"}), 400

    _append_log(data)
    receive_data_internal(data)
    return jsonify({"status": "success"}), 200


def receive_trade_data_internal(data: dict):
    eq_data = data.get('equity_data')

    if eq_data:
        vps_data_store['financials'].update({
            "equity": eq_data.get('equity', vps_data_store['financials']['equity']),
            "equity_peak": eq_data.get('equity_peak', vps_data_store['financials']['equity_peak']),
            "peak": eq_data.get('equity_peak', vps_data_store['financials']['peak']),
            "drawdown": eq_data.get('drawdown', vps_data_store['financials']['drawdown']),
            "daily_loss_pct": eq_data.get('daily_loss_pct', vps_data_store['financials']['daily_loss_pct']),
            "trading_enabled": eq_data.get('trading_enabled', vps_data_store['financials']['trading_enabled'])
        })
        vps_data_store['equity_history'].append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "equity": vps_data_store['financials']['equity']
        })
    elif 'equity' in data:
        vps_data_store['financials']['equity'] = data.get('equity')
        vps_data_store['financials']['equity_peak'] = data.get('equity_peak')

    if 'open_trades_summary' in data:
        vps_data_store['open_trades_summary'] = data.get('open_trades_summary', [])


@app.route('/update_trade_data', methods=['POST'])
def receive_trade_data():
    data = request.json or {}
    receive_trade_data_internal(data)
    return jsonify({"status": "success"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
