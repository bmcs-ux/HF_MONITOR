#
# cell_id: JGemPprTvBcw
# --------------------------------
# PARAMETERS (ubah sesuai kebutuhan)
# --------------------------------
import pandas as pd

PAIRS = {
    "USD/CAD": "USDCADm",
    "USD/DXY": "DXYm",
    "XAU/USD": "XAUUSDm",
    "XAG/USD": "XAGUSDm",
    "XCU/USD": "XCUUSDm",
    "USD/JPY": "USDJPYm",
    "EUR/USD": "EURUSDm",
    "US500": "US500m"
#    "US30": "US30m"
}


# NEW: Data window & base interval to download for HIGH-FREQUENCY MONITORING
HF_LOOKBACK_DAYS = 30         # e.g., last 6 days for high-frequency data
HF_BASE_INTERVAL = "1h"     # e.g., 15-minute interval for high-frequency data

# Timeframes we will analyse (mapping ke faktor resample)
TF_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "60min",
    "H4": "240min"   # will be created by resampling the base data
}
VARX_ENDOG_GROUPS = {
    'METALS_CLOSE': [
        'XAU/USD_Close_Log_Return',
        'XAG/USD_Close_Log_Return',
        'XCU/USD_Close_Log_Return',
#        'XPB/USD_Close_Log_Return'
    ],
    'CURRENCIES_OHLC': [
        'USD/CAD_Close_Log_Return',
        'USD/DXY_Close_Log_Return',
        'EUR/USD_Close_Log_Return', # Added EUR/USD Close Log Return
        'USD/JPY_Close_Log_Return'
    ]
#    'index_OHLC': [
#        'US500_Close_Log_Return',
#        'US30_Close_Log_Return'
#    ]
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

# RLS parameters for high-frequency monitoring
FORGETTING_FACTOR = 0.999 # Typically between 0.9 and 1.0. Lower for faster adaptation.
RLS_INITIAL_P_DIAG = 1e2 # Large initial value for P (covariance matrix inverse)
RLS_INITIAL_THETA = 0.0 # Initial guess for parameters (usually 0 or small random)
RLS_DEVIATION_THRESHOLD = 6.90 # NEW: Threshold for RLS parameter deviation.
RLS_DEVIATION_CLOSE_ALL_THRESHOLD = 7.1 # NEW: Threshold to trigger closing all positions

# NEW parameters for dynamic SL/TP adjustment based on RLS parameter deviation
RLS_SCALING_FACTOR_SL = 0.15 # Scales the increase in k_atr_stop and k_model_stop
RLS_SCALING_FACTOR_TP = 0.35 # Scales the reduction in tp_rr_ratio
RLS_SNR_INCREASE_FACTOR = 0.05 # Scales the increase in snr_threshold
RLS_TP_RR_MIN = 0.9 # Minimum acceptable tp_rr_ratio
RLS_SL_MAX_MULTIPLIER = 2.2 # Maximum allowed multiplier for k_atr_stop and k_model_stop

# Core maturity gate
RLS_MIN_UPDATES_FOR_CONFIDENCE = 60
# Confidence decay sensitivity
RLS_CONFIDENCE_ALPHA = 0.4
# Entry gate
RLS_CONFIDENCE_ENTRY_THRESHOLD = 0.40

MAGIC_NUMBER = 202401

# Trade management
EQUITY = 1000
RISK_PER_TRADE_PCT = 0.1
K_ATR_STOP = 0.5
K_MODEL_STOP = 1.0
SNR_THRESHOLD = 0.1
TP_RR_RATIO = 1.0
# Storage Paths
FORECAST_OUTPUT_PATH = '/content/drive/MyDrive/books/VARX_REGRESION/restored_forecasts.pkl'
FRED_DATA_PATH = '/content/drive/MyDrive/books/VARX_REGRESION/final_fred_data.pkl'
FITTED_MODELS_PATH = '/content/drive/MyDrive/books/VARX_REGRESION/fitted_models.pkl' # NEW: Path for fitted VARX/ARX models
