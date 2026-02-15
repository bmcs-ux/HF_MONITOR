from flask import Flask, jsonify, render_template_string, request
import os
import datetime

app = Flask(__name__)

# Penyimpanan data
vps_data_store = {
    "logs": [],
    "assets": {}, # Tempat menyimpan harga & prediksi terbaru
    "signals": {},
    "rls_forecast": {},
    "rls_health": {},
    "latest_actual_prices": {},
    "summary": {"status": "WAITING", "time": "-"},
    "equity_history": [], # Tambahkan ini untuk menampung data grafik
    "financials": {
        "equity": 0,
        "equity_peak": 0,  # Gunakan nama yang sama dengan payload Python
        "peak": 0,
        "drawdown": 0,
        "daily_loss_pct": 0,
        "trading_enabled": True
    }
}

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
    <title>CASSANDRA Project</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #0f111a; color: #e0e0e0; margin: 0; padding: 20px; }
        h1 { color: #4fc3f7; border-bottom: 1px solid #333; padding-bottom: 10px; font-size: 20px; }
        .grid { display: grid; grid-template-columns: 1fr; gap: 20px; }
        .card { background: #1a1d29; border: 1px solid #333; border-radius: 10px; padding: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        h3 { margin-top: 0; color: #e0e0e0; font-size: 14px; text-transform: uppercase; }

        /* Equity Text Style */
        .equity-box { background: #252836; padding: 20px; border-radius: 10px; border-left: 5px solid #00e676; margin-top: 10px; }
        .eq-val { font-size: 32px; font-weight: bold; font-family: monospace; color: #00e676; display: block; }
        .fin-row { display: flex; justify-content: space-between; margin-top: 10px; font-family: monospace; font-size: 14px; }

        /* Gauge Style - Versi Lengkap */
        .gauge-container { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
        .health-card { background: #252836; padding: 8px; border-radius: 10px; width: 110px; text-align: center; border: 1px solid #333; display: inline-block; margin: 5px; }
        .gauge-wrapper { position: relative; width: 80px; height: 40px; margin: auto; overflow: hidden; }
        .gauge-bg { width: 80px; height: 80px; border-radius: 50%; background: conic-gradient(from 270deg, #ff4d4d 0%, #ffeb3b 25%, #00e676 50%, #252836 50%); }
        .gauge-cover { position: absolute; top: 10px; left: 10px; width: 60px; height: 60px; background: #252836; border-radius: 50%; z-index: 2; }
        .gauge-needle { position: absolute; bottom: 0; left: 50%; width: 2px; height: 32px; background: #fff; transform-origin: bottom center; transition: transform 1s ease; z-index: 3; }
        .group-label { font-size: 9px; font-weight: bold; color: #4fc3f7; text-transform: uppercase; display: block; margin-bottom: 4px; }
        .conf-val { font-size: 14px; font-weight: bold; display: block; margin-top: 2px; }

        /* Table & Console */
        table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
        th { text-align: left; color: #555; padding: 10px; border-bottom: 2px solid #333; }
        td { padding: 12px 10px; border-bottom: 1px solid #252836; }
        .price-up { color: #00e676; }
        .price-down { color: #ff5252; }
        .log-container { background: #000; padding: 10px; height: 180px; overflow-y: auto; font-family: monospace; font-size: 11px; border: 1px solid #333; color: #888; }
    </style>
</head>
<body>
    <h1>CASSANDRA PROJECT DASHBOARD</h1>
    
    <div class="grid">
        <div class="card">
            <h3>Equity Performance</h3>
            <div class="equity-box">
                <span style="font-size: 12px; color: #888;">ACCOUNT EQUITY</span>
                <span id="curr_eq" class="eq-val">$0.00</span>
                <div class="fin-row">
                    <div style="color: #ffcc00;">PEAK: <span id="peak_val">$0</span></div>
                    <div style="color: #ff5252;">DRAWDOWN: <span id="dd_val">0%</span></div>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>Model Health</h3>
            <div id="healthStatus" class="gauge-container">Menunggu dtaa...</div>
        </div>

        <div class="card">
            <h3>Live Market Analysis</h3>
            <table>
                <thead><tr><th>ASSET</th><th>ACTUAL</th><th>PREDIKSI</th><th>EXPECTED</th></tr></thead>
                <tbody id="assetTable"></tbody>
            </table>
        </div>

        <div class="card" style="margin-top: 20px;"> <div class="d-flex justify-content-between align-items-center mb-3">
                <h3 style="margin: 0;"><i class="fas fa-satellite-dish me-2"></i>Live Trade Signals</h3>
                <span id="last-signal-update" style="font-size: 12px; color: #888;">Waiting for update...</span>
            </div>

            <div id="signal-grid" class="row row-cols-1 row-cols-md-3 g-4">
                </div>
        </div>

        <div class="card">
            <h3>System Console</h3>
            <div id="logBox" class="log-container"></div>
        </div>
    </div>

    <script>
        function renderSignals(signals) {
            const grid = document.getElementById('signal-grid');
            grid.innerHTML = ''; // Bersihkan grid lama

            for (const [symbol, data] of Object.entries(signals)) {
                // Tentukan warna berdasarkan status signal
                let badgeClass = "bg-secondary";
                let cardBorder = "border-secondary";
        
                if (data.signal === "BUY") {
                    badgeClass = "bg-success";
                    cardBorder = "border-success shadow-sm";
                } else if (data.signal === "SELL") {
                    badgeClass = "bg-danger";
                    cardBorder = "border-danger shadow-sm";
                }

                const cardHtml = `
                    <div class="col">
                        <div class="card h-100 bg-black text-white ${cardBorder}" style="border-width: 2px; border-radius: 12px; overflow: hidden;">
                            <div class="card-body" style="padding: 10px 15px 20px 15px;">
                
                                <div class="d-flex justify-content-between align-items-start mb-3">
                                    <h3 class="card-title mb-0 fw-bold" style="font-size: 1.6rem; letter-spacing: -0.5px; margin-top: -2px;">
                                        ${symbol}
                                    </h3>
                                    <span class="badge ${badgeClass}" style="margin-right: 15px; padding: 0px 10px; font-size: 0.8rem; min-width: 60px;">
                                        ${data.signal}
                                    </span>
                                </div>
                
                                <div class="signal-info-rows" style="margin-bottom: 15px;">
                                    <div class="d-flex justify-content-between align-items-center mb-2">
                                        <span style="color: #666; font-size: 0.85rem;">ENTRY</span>
                                        <span class="text-info fw-bold" style="font-family: 'Courier New', monospace;">${data.entry_price || '-'}</span>
                                    </div>
                                    <div class="d-flex justify-content-between align-items-center mb-2">
                                        <span style="color: #666; font-size: 0.85rem;">TAKE PROFIT</span>
                                        <span class="text-success fw-bold" style="font-family: 'Courier New', monospace;">${data.take_profit ? data.take_profit.toFixed(5) : '-'}</span>
                                    </div>
                                    <div class="d-flex justify-content-between align-items-center mb-2">
                                        <span style="color: #666; font-size: 0.85rem;">STOP LOSS</span>
                                        <span class="text-warning fw-bold" style="font-family: 'Courier New', monospace;">${data.stop_loss ? data.stop_loss.toFixed(5) : '-'}</span>
                                    </div>
                                </div>

                                <hr style="border-color: #222; margin: 12px 0;">

                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <span style="font-size: 0.7rem; color: #666; font-weight: bold; letter-spacing: 0.5px;">SNR STRENGTH</span>
    
                                    <div style="
                                        background: rgba(${data.snr > 1.5 ? '0, 255, 136' : '255, 82, 82'}, 0.1); 
                                        border: 1px solid rgba(${data.snr > 1.5 ? '0, 255, 136' : '255, 82, 82'}, 0.3);
                                        padding: 2px 10px;
                                        border-radius: 20px;
                                        display: flex;
                                        align-items: center;
                                        gap: 6px;
                                    ">
                                        <div style="
                                            width: 6px; 
                                            height: 6px; 
                                            background: ${data.snr > 1.5 ? '#00ff88' : '#ff5252'}; 
                                            border-radius: 50%;
                                            box-shadow: 0 0 5px ${data.snr > 1.5 ? '#00ff88' : '#ff5252'};
                                        "></div>
        
                                        <span style="
                                            color: ${data.snr > 1.5 ? '#00ff88' : '#ff5252'}; 
                                            font-weight: 800; 
                                            font-size: 0.9rem;
                                            font-family: 'Segoe UI', sans-serif;
                                        ">
                                            ${data.snr ? data.snr.toFixed(2) : '-'}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                grid.insertAdjacentHTML('beforeend', cardHtml);
            }
        }

        let currentStoredEquity = 0;
        let currentStoredPeak = 0;

        function updateUI() {
            fetch('/api/get_data')
                .then(res => res.json())
                .then(data => {
                    if (!data) return;
                    const lastLog = (data.logs && data.logs.length > 0) ? data.logs[data.logs.length - 1] : {};
                    const latestLog = data.logs[0];
                    if (latestLog && latestLog.trade_signals) {
                        renderSignals(latestLog.trade_signals);
                        document.getElementById('last-signal-update').innerText = "Cycle: " + (latestLog.cycle_number || '-');
                    }

                    // 1. Update Financials
                    const eqData = data.equity_data || {};

                    // Ambil Equity: Cek di equity_data, lalu fallback ke root (data.equity), lalu simpan di memori
                    const incomingEquity = eqData.equity || data.equity || (data.financials && data.financials.equity);
                    if (incomingEquity && incomingEquity !== 0) {
                        currentStoredEquity = incomingEquity;
                    }

                    // Ambil Peak
                    const incomingPeak = eqData.equity_peak || data.equity_peak || (data.financials && data.financials.peak);
                    if (incomingPeak && incomingPeak !== 0) {
                        currentStoredPeak = incomingPeak;
                    }

                    // Ambil Drawdown & Daily Loss (Jika ingin ditampilkan juga)
                    const drawdown = eqData.drawdown || data.drawdown || 0;

                    // Render ke UI
                    document.getElementById('curr_eq').innerText = '$' + parseFloat(currentStoredEquity).toLocaleString(undefined, {minimumFractionDigits: 2});
                    document.getElementById('peak_val').innerText = '$' + parseFloat(currentStoredPeak).toLocaleString(undefined, {minimumFractionDigits: 2});
                    document.getElementById('dd_val').innerText = parseFloat(drawdown).toFixed(2) + '%';

                    // Tambahan: Update warna border berdasarkan trading_enabled yang baru
                    const eqBox = document.querySelector('.equity-box');
                    if (eqBox) {
                        // Ambil status dari eqData
                        const isEnabled = eqData.trading_enabled;
                        if (isEnabled !== undefined) {
                            eqBox.style.borderLeftColor = isEnabled ? "#00e676" : "#ff5252";
                        }
                    }

                    // 2. Update Gauge (Model Health)
                    const healthSource = data.rls_health || lastLog.rls_health;
                    const healthBox = document.getElementById('healthStatus');
                    if (healthSource) {
                        let hHtml = "";
                        for (const group in healthSource) {
                            const metrics = healthSource[group];
                            hHtml += `<div style="width:100%; border-bottom:1px solid #333; margin:10px 0 5px; color:#4fc3f7; font-weight:bold; font-size:12px;">${group}</div>`;

                            const configs = [
                                { key: 'confidence', label: 'CONFIDENCE', colorMode: 'standard' },
                                { key: 'maturity',   label: 'MATURITY',   colorMode: 'standard' },
                                { key: 'deviation',  label: 'DEVIATION',  colorMode: 'reverse' },
                                { key: 'pred_var',   label: 'PRED VAR',   colorMode: 'reverse' }
                            ];

                            configs.forEach(cfg => {
                                const val = metrics[cfg.key] || 0;
                                const displayVal = val > 1 ? 1 : (val < 0 ? 0 : val); 
                                const rot = (displayVal * 180) - 90;
                                let clr = (cfg.colorMode === 'standard') ? 
                                    (val > 0.7 ? '#00e676' : (val > 0.4 ? '#ffeb3b' : '#ff4d4d')) :
                                    (val < 0.3 ? '#00e676' : (val < 0.6 ? '#ffeb3b' : '#ff4d4d'));

                                hHtml += `
                                <div class="health-card">
                                    <span class="group-label">${cfg.label}</span>
                                    <div class="gauge-wrapper">
                                        <div class="gauge-bg"></div><div class="gauge-cover"></div>
                                        <div class="gauge-needle" style="transform: rotate(${rot}deg); background:${clr}"></div>
                                    </div>
                                    <span class="conf-val" style="color:${clr}">${Math.round(displayVal * 100)}%</span>
                                </div>`;
                            });
                        }
                        healthBox.innerHTML = hHtml;
                    }

                    // 3. Update Table
                    const actuals = data.latest_actual_prices || lastLog.latest_actual_prices || {};
                    const forecasts = data.rls_forecast || lastLog.rls_forecast || {};
                    let tHtml = "";
                    Object.keys(actuals).forEach(sym => {
                        const act = parseFloat(actuals[sym] || 0);
                        const prd = parseFloat(forecasts[sym] || 0);
                        const diff = act !== 0 ? ((prd - act) / act) * 100 : 0;
                        const cls = diff >= 0 ? 'price-up' : 'price-down';
                        tHtml += `<tr>
                            <td><b>${sym}</b></td>
                            <td>${act.toLocaleString(undefined, {minimumFractionDigits: 4})}</td>
                            <td>${prd !== 0 ? prd.toLocaleString(undefined, {minimumFractionDigits: 4}) : 'WAIT'}</td>
                            <td class="${cls}">${diff.toFixed(4)}%</td>
                        </tr>`;
                    });
                    if(tHtml) document.getElementById('assetTable').innerHTML = tHtml;

                    // 4. Update Logs
                    if (data.logs) {
                        document.getElementById('logBox').innerHTML = data.logs.slice().reverse().map(l => 
                            `<div style="margin-bottom:5px; border-bottom: 1px solid #222;">> ${JSON.stringify(l).substring(0, 120)}...</div>`
                        ).join('');
                    }
                })
                .catch(err => console.error("Update Error:", err));
        }

        setInterval(updateUI, 3000);
        updateUI();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/get_data')
def get_data():
    return jsonify(vps_data_store)

@app.route('/update_monitor_data', methods=['POST'])
def receive_data():
    global vps_data_store
    data = request.json
    if not data:
        return jsonify({"status": "error"}), 400

    # 1. Simpan logs (Gunakan insert agar data terbaru di atas)
    vps_data_store['logs'].insert(0, data)
    if len(vps_data_store['logs']) > 20:
        vps_data_store['logs'].pop()

    # 2. Update Health (Gauge)
    new_health = data.get('rls_health')
    if new_health:
        vps_data_store['rls_health'].update(new_health)

    # 3. Update Harga Aktual (dari deviation_results atau root)
    if 'deviation_results' in data:
        for sym, val in data['deviation_results'].items():
            if isinstance(val, dict) and 'actual_price' in val:
                vps_data_store['latest_actual_prices'][sym] = val['actual_price']

    new_prices = data.get('latest_actual_prices')
    if new_prices:
        vps_data_store['latest_actual_prices'].update(new_prices)

    # 4. Update Prediksi (RLS Forecast)
    incoming_rls = data.get('rls_forecast') or data.get('rls_forecasts')
    if incoming_rls:
        for sym, val in incoming_rls.items():
            if isinstance(val, dict):
                p = val.get('rls_predicted_price') or val.get('predicted_price')
                if p: vps_data_store['rls_forecast'][sym] = p
            else:
                vps_data_store['rls_forecast'][sym] = val


    return jsonify({"status": "success"}), 200

@app.route('/update_trade_data', methods=['POST'])
def receive_trade_data():
    global vps_data_store
    data = request.json
    #print("TRADE DATA RECEIVED:", data.keys())
    eq_data = data.get('equity_data')
    
    if eq_data:
        # Update hanya jika paket ini membawa data equity_data
        vps_data_store['financials'].update({
            "equity": eq_data.get('equity', vps_data_store['financials']['equity']),
            "equity_peak": eq_data.get('equity_peak', vps_data_store['financials']['equity_peak']),
            "peak": eq_data.get('equity_peak', vps_data_store['financials']['peak']),
            "drawdown": eq_data.get('drawdown', vps_data_store['financials']['drawdown']),
            "daily_loss_pct": eq_data.get('daily_loss_pct', vps_data_store['financials']['daily_loss_pct']),
            "trading_enabled": eq_data.get('trading_enabled', vps_data_store['financials']['trading_enabled'])
        })
        
        # Tambahkan ke history untuk grafik hanya jika ada update equity
        vps_data_store['equity_history'].append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "equity": vps_data_store['financials']['equity']
        })
    elif 'equity' in data:
        # Fallback jika bot mengirim data di level root (bukan di dalam equity_data)
        vps_data_store['financials']['equity'] = data.get('equity')
        vps_data_store['financials']['equity_peak'] = data.get('equity_peak')

    # 6. Update Open Trades
    if 'open_trades_summary' in data:
        vps_data_store['open_trades_summary'] = data.get('open_trades_summary', [])

    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
