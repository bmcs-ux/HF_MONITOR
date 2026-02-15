
import sys
import os
import importlib
import time
import random
from datetime import datetime
import shutil

def display_kassandra_opening():
    """
    Advanced cinematic startup for KASSANDRA Cognitive Trading System
    """

    project_name = "CASSANDRA"
    subtitle = "Automated Market Analysis and Risk Management System"
    owner_nick = "BIMACHASIN86"

    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#@&%$"
    width = shutil.get_terminal_size((80, 20)).columns

    # ANSI Colors
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    GRAY   = "\033[38;5;109m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    def slow_print(text, delay=0.02, color=RESET):
        for ch in text:
            sys.stdout.write(f"{color}{ch}{RESET}")
            sys.stdout.flush()
            time.sleep(delay)
        print()

    def glitch_reveal(text):
        buffer = [" "] * len(text)
        for i in range(len(text)):
            for _ in range(5):
                buffer[i] = random.choice(chars)
                sys.stdout.write(
                    "\r" + " " * ((width - len(text)) // 2) +
                    f"{CYAN}{BOLD}{''.join(buffer)}{RESET}"
                )
                sys.stdout.flush()
                time.sleep(0.015)
            buffer[i] = text[i]
        sys.stdout.write(
            "\r" + " " * ((width - len(text)) // 2) +
            f"{GREEN}{BOLD}{''.join(buffer)}{RESET}\n"
        )
        sys.stdout.flush()

    # ===============================
    # SYSTEM BOOT SEQUENCE
    # ===============================
    print(f"\n{GRAY}{'-' * width}{RESET}")
    slow_print(" INITIALIZING SYSTEM ...", 0.03, GRAY)
    print(f"{GRAY}{'-' * width}{RESET}\n")
    time.sleep(0.3)

    # Project Name Reveal
    glitch_reveal(project_name)
    time.sleep(0.1)

    slow_print(
        f"{' ' * ((width - len(subtitle)) // 2)}{subtitle}",
        0.01,
        GRAY
    )
    time.sleep(0.4)

    print()

    # ===============================
    # DIAGNOSTIC PHASE
    # ===============================
    diagnostics = [
        ("Processing Data", GREEN),
        ("Mengadaptasi Parameter", GREEN),
        ("Loading Volatility Intelligence", GREEN),
        ("Risk Containment Protocol", YELLOW),
        ("Execution & Fail-Safe Module", GREEN),
    ]

    for label, color in diagnostics:
        slow_print(f" [OK] {label} synchronized", 0.015, color)
        time.sleep(0.05)

    print()

    # ===============================
    # OWNER IDENTITY
    # ===============================
    slow_print(" SYSTEM AUTHORITY", 0.03, CYAN)
    sys.stdout.write(f"{YELLOW}{BOLD} >> ")
    for ch in owner_nick:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(0.05)
    print(f" <<{RESET}")

    time.sleep(0.3)

    # ===============================
    # FINAL STATUS
    # ===============================
    print(f"\n{CYAN}{'-' * width}{RESET}")
    slow_print(
        " STATUS : COGNITIVE ENGINE ONLINE | AUTONOMOUS MODE ENABLED",
        0.02,
        GREEN
    )
    slow_print(
        " MODE   : REAL-TIME MARKET INFERENCE | VPS DEPLOYMENT",
        0.02,
        GREEN
    )
    print(f"{CYAN}{'-' * width}{RESET}\n")

    time.sleep(0.4)


# Run opening
display_kassandra_opening()


# --- Configuration for VPS --- 
# Set ROOT_DIR_VPS to the absolute path where your project is cloned on the VPS
ROOT_DIR_VPS = "/home/bimachasin86/VARX_REGRESION/"

# Add the root directory to sys.path
if ROOT_DIR_VPS not in sys.path:
    sys.path.append(ROOT_DIR_VPS)

# Import necessary modules
try:
    from mt5linux import MetaTrader5
    mt5 = MetaTrader5()
    print("Kassandra menggunakan mt5linux bridge...")
except ImportError:
    import MetaTrader5 as mt5
    print("kassandra menggunakan MetaTrader5 native...")

# Reload modules to ensure they pick up the latest changes and dummy MT5 if applicable
modules_to_reload = [
    'parameter',
    'vps_colab_connector',
    'news_manager',
    'trade_engine',
    'monitor_for_vps',
    'preprocesing.log_return',
    'preprocesing.combine_data',
    'preprocesing.stationarity_test'
]

print("[RUN.PY] Reloading essential modules...")
for module_name in modules_to_reload:
    try:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)
        print(f"  [RUN.PY] Successfully reloaded/imported: {module_name}")
    except Exception as e:
        print(f"  [RUN.PY ERROR] Failed to reload/import {module_name}: {e}")
        # If a critical module fails to load, exit
        if module_name in ['parameter', 'trade_engine', 'monitor_for_vps']:
            sys.exit(1)

import parameter
import trade_engine
import monitor_for_vps

# --- Set VPS Paths within the modules --- 
# This ensures the modules load data from the correct locations on the VPS
parameter.VPS_PARAM_DIR = ROOT_DIR_VPS
parameter.VPS_DATA_DIR = ROOT_DIR_VPS
trade_engine.VPS_PARAM_DIR = ROOT_DIR_VPS
trade_engine.VPS_DATA_DIR = ROOT_DIR_VPS
monitor_for_vps.VPS_PARAM_DIR = ROOT_DIR_VPS
monitor_for_vps.VPS_DATA_DIR = ROOT_DIR_VPS

# --- Set Colab URL file path for both modules --- 
colab_url_file = os.path.join(ROOT_DIR_VPS, 'colab_ngrok_url.txt')
monitor_for_vps.COLAB_URL_FILE_PATH = colab_url_file
trade_engine.COLAB_URL_FILE_PATH = colab_url_file

# --- Temporary hack to ensure TRADE_ENGINE_API_KEY is available in parameter module ---
# This is needed because monitor_for_vps needs to import it, 
# but it's defined in trade_engine.py. On VPS, a shared config might be better.
if not hasattr(parameter, 'TRADE_ENGINE_API_KEY'):
    parameter.TRADE_ENGINE_API_KEY = trade_engine.TRADE_ENGINE_API_KEY

print("[RUN.PY] Starting VPS orchestration...")

VPS_DATA_DIR = ROOT_DIR_VPS

# Define local log paths within the VPS_DATA_DIR
TRADE_ENGINE_LOG_FILE_VPS = os.path.join(VPS_DATA_DIR, "trade_engine_log_vps.txt")
MONITOR_LOG_FILE_VPS = os.path.join(VPS_DATA_DIR, "monitor_log_vps.txt")

# 1. Start Trade Engine's Flask API and MT5 connection/monitoring
print("[RUN.PY] Starting Trade Engine's Flask API and MT5 connection...")
trade_engine_started = trade_engine.start_trade_engine_flask_and_monitor(
    log_output_path=TRADE_ENGINE_LOG_FILE_VPS
)

if trade_engine_started:
    print("[RUN.PY] Trade Engine Flask API and MT5 connection successfully started.")
    time.sleep(2) # Give Flask app a moment to fully start

    # 2. Start the Monitor's real-time monitoring loop
    print("[RUN.PY] Starting Monitor's real-time monitoring...")
    # Configure monitoring parameters for VPS run
    VPS_TOTAL_DURATION_MINUTES = 60 * 24 * 365 # Example: 1 year of monitoring (run indefinitely)
    VPS_INTERVAL_SECONDS = 60 * 1            # Example: check every 1 minute
    VPS_CONFIDENCE_LEVEL = 0.95

    monitoring_results, full_log = monitor_for_vps.start_realtime_monitoring(
        total_duration_minutes=VPS_TOTAL_DURATION_MINUTES,
        interval_seconds=VPS_INTERVAL_SECONDS,
        confidence_level=VPS_CONFIDENCE_LEVEL,
        pipeline_run_id="VPS_ORCHESTRATION_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        log_output_path=MONITOR_LOG_FILE_VPS
    )

    print("[RUN.PY] VPS Orchestration completed its main monitoring loop.")
    print(f"[RUN.PY] See logs: {MONITOR_LOG_FILE_VPS} and {TRADE_ENGINE_LOG_FILE_VPS}")

else:
    print("[RUN.PY ERROR] Failed to start Trade Engine. Aborting monitoring.")
    sys.exit(1)

print("[RUN.PY] Orchestration script finished.")
