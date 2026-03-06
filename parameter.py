# --------------------------------
# PARAMETERS (ubah sesuai kebutuhan)
# --------------------------------
import os
import pandas as pd

# Runtime and deployment configuration
ROOT_DIR = os.getenv("HF_ROOT_DIR", "/home/bimachasin86/VARX_REGRESION")
VPS_PARAM_DIR = ROOT_DIR
VPS_DATA_DIR = ROOT_DIR

# Storage Paths
FORECAST_OUTPUT_PATH = os.path.join(ROOT_DIR, 'vps_sync', 'restored_forecasts.pkl')
FRED_DATA_PATH = os.path.join(ROOT_DIR, 'vps_sync', 'final_fred_data.pkl')
FITTED_MODELS_PATH = os.path.join(ROOT_DIR, 'vps_sync', 'fitted_models.pkl') # Path for fitted ensemble models

# Credentials and API keys (set via environment variables in production)
MT5_LOGIN = os.getenv("MT5_LOGIN", 206905748)
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "Bima12345#")
MT5_SERVER = os.getenv("MT5_SERVER", "Exness-MT5Trial7")

TRADE_ENGINE_API_KEY = os.getenv("TRADE_ENGINE_API_KEY", "bima_12345678")
COLAB_API_KEY_FOR_TRADE_ENGINE = os.getenv("COLAB_API_KEY_FOR_TRADE_ENGINE", TRADE_ENGINE_API_KEY)
COLAB_API_KEY_FOR_MONITOR = os.getenv("COLAB_API_KEY_FOR_MONITOR", TRADE_ENGINE_API_KEY)
COLAB_URL_FILE_PATH = os.path.join(VPS_DATA_DIR, "colab_ngrok_url.txt")
TRADE_ENGINE_API_URL = os.getenv("TRADE_ENGINE_API_URL", "http://127.0.0.1:8081/receive_signal")


PAIRS = {
    'GBPUSD': 'GBPUSDm',
    'AUDUSD': 'AUDUSDm',
    'USDCAD': 'USDCADm',
    'USDCHF': 'USDCHFm',
    'USDJPY': 'USDJPYm',
    'NZDUSD': 'NZDUSDm',
    'XAUUSD': 'XAUUSDm',   # Gold
    'XAGUSD': 'XAGUSDm',   # Silver
    #'USOIL': 'USOILm',     # Crude Oil
    'US500': 'US500m',
    'DXY': 'DXYm',
}


# NEW: Data window & base interval to download for HIGH-FREQUENCY MONITORING
HF_LOOKBACK_DAYS = 3         # e.g., last 6 days for high-frequency data
HF_BASE_INTERVAL = "1m"     # e.g., 15-minute interval for high-frequency data

# Timeframes we will analyse (mapping ke faktor resample)
TF_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "60min",
    "H4": "240min"   # will be created by resampling the base data
}

VARX_ENDOG_GROUPS = {
    'FX_Majors': ['GBPUSD_Close_Log_Return', 'AUDUSD_Close_Log_Return', 'USDJPY_Close_Log_Return'],
    'Commodities': ['XAUUSD_Close_Log_Return', 'XAGUSD_Close_Log_Return'],
    'Risk': ['US500_Close_Log_Return', 'DXY_Close_Log_Return'],
}

BLOK_SIGNAL_FOR = {"US500", "US30","USD/DXY", "USD/CAD","EUR/USD","XCU/USD","XPB/USD"}  # Masukkan nama pair yang ingin diblokir
# Note: yfinance intervals use strings like '1m','5m','15m','60m'. We will resample using pandas offsets.

# Granger params
maxlag_test = 5              # max lag (jumlah observation lag) untuk uji Granger
alpha = 0.05                 # signifikansi Granger
min_obs_for_granger = 50     # minimal observasi supaya Granger bermakna

# cell_id: HYcYVva61v1i
#id series_parameter
FRED_SERIES = {
    #"Secured Overnight Financing Rate": "SOFR",     # CPI AS (index, bisa dipakai hitung inflasi YoY)
    "S&P 500": "SP500",    # Federal Funds Rate
    #"Currency Conversions: US Dollar Exchange Rate": "CYPCCUSMA02STQ",              # 10-year Treasury yield
    #"US Corporate Index Semi-Annual Yi": "BAMLC0A4CBBBSYTW",
    "Index Semi-Annual" :  "BAMLH0A0HYM2SYTW",
    #"Emerging Markets Corporate": "BAMLEM5BCOCRPISYTW",
    "Markets Corporate Plus" : "BAMLEMRECRPIEMEASYTW",
    #"Financial US Emerging Markets Liquiq" : "BAMLEMFLFLCRPIUSSYTW",
    #"Daily Sterling Overnight Index Average (SONIA)...": "IUDSOIA",
    "Effective Federal Funds Volume" : "EFFRVOL",
    "Overnight Reverse Repurchase Agreements: Treas.." : "RRPONTSYD",
    #"30-Day Moving Average AMERIBOR BENCHMARK Inter" : "AMBOR30",
    "Market Yield on U.S. Treasury Securities at 10" : "DGS10",
    "5-Year Breakeven Inflation Rate: Treas..": "T5YIE",
    "Effective Federal Funds Rate" : "DFF"
}
FRED_API_KEY = "YOUR_FRED_API_KEY" #if you don't have fred api key, go to fred website follow instruction for get your api key

forecast_horizon = 2

NEWS = True #News gate

# RLS parameters for high-frequency monitoring
FORGETTING_FACTOR = 0.999 # Typically between 0.9 and 1.0. Lower for faster adaptation.
RLS_INITIAL_P_DIAG = 1e2 # Large initial value for P (covariance matrix inverse)
RLS_INITIAL_THETA = 0.0 # Initial guess for parameters (usually 0 or small random)

_RLS_DEVIATION_THRESHOLD = True #Deviation gate
RLS_DEVIATION_THRESHOLD = 6.90 # NEW: Threshold for RLS parameter deviation.
RLS_DEVIATION_CLOSE_ALL_THRESHOLD = 7.1 # NEW: Threshold to trigger closing all positions

# NEW parameters for dynamic SL/TP adjustment based on RLS parameter deviation
RLS_SCALING_FACTOR_SL = 0.15 # Scales the increase in k_atr_stop and k_model_stop
RLS_SCALING_FACTOR_TP = 0.35 # Scales the reduction in tp_rr_ratio
RLS_SNR_INCREASE_FACTOR = 0.05 # Scales the increase in snr_threshold
RLS_TP_RR_MIN = 0.5 # Minimum acceptable tp_rr_ratio
RLS_SL_MAX_MULTIPLIER = 2.2 # Maximum allowed multiplier for k_atr_stop and k_model_stop


_RLS_CONFIDENCE = False
# Core maturity gate
RLS_MIN_UPDATES_FOR_CONFIDENCE = 40
# Confidence decay sensitivity
RLS_CONFIDENCE_ALPHA = 0.4
# Entry gate
RLS_CONFIDENCE_ENTRY_THRESHOLD = 0.40

MAGIC_NUMBER = 202401

# Trade management
EQUITY = 1000
RISK_PER_TRADE_PCT = 0.1
K_ATR_STOP = 0.6
K_MODEL_STOP = 1.0
SNR_THRESHOLD = 0.1
TP_RR_RATIO = 1.0

# Adaptive safeguards for mixed-timeframe RLS monitoring
RLS_VOLATILITY_WINDOW = 96
RLS_MIN_INNOVATION_SCALE = 0.5
RLS_DEVIATION_ADAPTIVE_STD_MULTIPLIER = 0.5


# Multi-timeframe consensus & execution tuning
CONSENSUS_WEIGHT_D1 = 0.4
CONSENSUS_WEIGHT_H1 = 0.5
CONSENSUS_WEIGHT_M15 = 0.2
CONSENSUS_THRESHOLD = 0.15

# DCC proxy controls (contagion -> wider risk envelope)
DCC_RISK_MULTIPLIER = 0.5
DCC_FLIP_EPS_MULTIPLIER = 0.5

# Mean reversion monitor gates
MEAN_REVERSION_HIGH_Z = 2.5
MEAN_REVERSION_LOW_VOL_PREDVAR = 0.002

# Kalman execution filter defaults (M1)
KALMAN_F = [[1, 1], [0, 1]]
KALMAN_H = [[1, 0]]
KALMAN_Q = [[0.0001, 0.0], [0.0, 0.0001]]
KALMAN_R = [[0.000001]]
KALMAN_INITIAL_STATE = [1.0, 0.0]
KALMAN_INITIAL_P = [[0.1, 0.0], [0.0, 0.1]]
KALMAN_ZSCORE_WINDOW = 120
KALMAN_FLIP_ZSCORE = 3.0
