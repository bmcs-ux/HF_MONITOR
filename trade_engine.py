
import re # Import regex module
from datetime import datetime
import time
import os
import sys
import json
import numpy as np
from flask import Flask, request, jsonify
import threading
import traceback
from mt5_adapter import MT5Adapter, MT5_TIMEFRAME_MAP
from news_manager import NewsManager

current_script_dir = "/home/bimachasin86/VARX_REGRESION"
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)
import vps_colab_connector

ROOT_DIR_LOCAL = '/home/bimachasin86/VARX_REGRESION'
VPS_PARAM_DIR = ROOT_DIR_LOCAL
VPS_DATA_DIR = ROOT_DIR_LOCAL

if VPS_PARAM_DIR not in sys.path:
    sys.path.insert(0, VPS_PARAM_DIR)

import parameter

MT5_LOGIN = YOUR_MT5_USERNAME
MT5_PASSWORD = "YOUR_MT5_PASSWORD"
MT5_SERVER = "YOUR_MT5_SERVER"

DEFAULT_VOLUME = 0.01
SLIPPAGE = 20
MAGIC_NUMBER = 202401

app = Flask(__name__)
TRADE_ENGINE_API_KEY = "bima_12345678"
TRADE_ENGINE_API_PORT = 8081

COLAB_API_KEY_FOR_TRADE_ENGINE = "bima_12345678"
COLAB_URL_FILE_PATH = os.path.join(VPS_DATA_DIR, "colab_ngrok_url.txt")

import re

def sanitize_comment(comment: str, max_len: int = 31):
    # Replace any character not allowed by MT5
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', comment)
    return safe[:max_len]

def send_trade_data_to_colab(data: dict, log_func=print):
    try:
        local_url = "http://127.0.0.1:5000/update_trade_data" # Sesuaikan portnya
        import requests
        requests.post(local_url, json=data, timeout=5)
    except:
        pass

    return vps_colab_connector.send_data_to_colab(
        endpoint="update_trade_data",
        data=data,
        colab_api_key=COLAB_API_KEY_FOR_TRADE_ENGINE,
        colab_url_file_path=COLAB_URL_FILE_PATH,
        log_func=log_func
    )

#from mt5_adapter import MT5Adapter, MT5_TIMEFRAME_MAP

class TradeEngine:
    def __init__(self, log_output_path: str = None):
        self.log_output_path = log_output_path
        self.log_stream = open(log_output_path, 'a') if log_output_path else None
        self.connected = False
        self.EQUITY_DRAWDOWN_TOLERANCE = 0.10  # 10% max drawdown
        self.DAILY_LOSS_TOLERANCE = 0.045      # 4.5% Daily Loss Guard (Aturan 2.3) - Kasih buffer 0.5%
        self.equity_peak = None               # runtime reference
        self.daily_equity_start = None         # Saldo awal hari untuk hitung Daily Loss
        self.last_day_reset = None             # Tracking hari untuk reset Daily Loss
        self.trading_enabled = True            # Kill-switch jika kena limit
        self.news_manager = NewsManager(VPS_DATA_DIR, self._log)
        self.news_manager.load_local_news() # Load cache saat startup
        self.equity_peak = None               # runtime reference

        self.MT5_LOGIN = MT5_LOGIN
        self.MT5_PASSWORD = MT5_PASSWORD
        self.MT5_SERVER = MT5_SERVER
        self.MAGIC_NUMBER = MAGIC_NUMBER
        self.SLIPPAGE = SLIPPAGE

        self.mt5 = MT5Adapter(logger=self._log)

        self.mt5_initialize()
        self.current_open_trades = {}

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.log_stream:
            self.log_stream.write(f"[{timestamp}] [TradeEngine] {message}\n")
            self.log_stream.flush()
        else:
            print(f"[{timestamp}] [TradeEngine] {message}")

    def mt5_initialize(self):
        if not self.mt5.initialize():
            self._log(f"Failed to initialize MT5: {self.mt5.last_error()}")
            self.connected = False
            return False

        if not self.mt5.login(
            login=self.MT5_LOGIN,
            password=self.MT5_PASSWORD,
            server=self.MT5_SERVER
        ):
            self._log(
                f"Failed to connect to MT5 account {self.MT5_LOGIN}: "
                f"{self.mt5.last_error()}"
            )
            self.mt5.shutdown()
            self.connected = False
            return False

        self._log(f"Successfully connected to MetaTrader5 account {self.MT5_LOGIN}")
        self.connected = True
        return True

    def mt5_shutdown(self):
        if self.connected:
            self.mt5.shutdown()
            self._log("MetaTrader5 connection shutdown.")
            self.connected = False
        if self.log_output_path:
            self.log_stream.close()

    def get_symbol_info(self, symbol):
        symbol_info = self.mt5.symbol_info(symbol)
        if symbol_info is None:
            self._log(f"Failed to get symbol info for {symbol}: {self.mt5.last_error()}")
            return None
        return symbol_info

    def send_order(self, symbol: str, trade_type: str, volume: float, price: float,
                   stop_loss: float = 0.0, take_profit: float = 0.0, comment: str = ""):
        if not self.connected:
            self._log("MT5 not connected. Cannot send order.")
            return {"status": "error", "message": "MT5 not connected"}

        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            return {"status": "error", "message": f"Could not get symbol info for {symbol}"}

        point = self.mt5.symbol_info(symbol).point
        if point == 0:
            self._log(f"[WARN] Symbol {symbol} has a point value of 0. Using default point 0.0001 for calculations.")
            point = 0.0001

        deviation = self.SLIPPAGE

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": self.mt5.ORDER_TYPE_BUY if trade_type == "BUY" else self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": int(deviation),
            "magic": self.MAGIC_NUMBER,
            "comment": comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_FOK
        }

        if stop_loss > 0:
            request["sl"] = float(round(stop_loss / point) * point)
        if take_profit > 0:
            request["tp"] = float(round(take_profit / point) * point)

        self._log(f"Sending order request: {request}")

        positions_cek = self.mt5.positions_get(symbol=symbol, magic=self.MAGIC_NUMBER)
        if positions_cek and len(positions_cek) > 0:
            self._log(
                f"[BLOCK] Order blocked for {symbol}: existing position detected "
                f"(count={len(positions_cek)})"
            )
            return {
                "status": "blocked",
                "message": "Existing position already open"
            }

        result = self.mt5.order_send(request)

        if result is None:
            err = self.mt5.last_error()
            self._log(f"[ERROR] order_send returned None: {err}")
            return {
                "status": "error",
                "message": "order_send failed",
                "last_error": err
            }

        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            self._log(f"Order failed for {symbol} {trade_type} (retcode={result.retcode}): {result.comment}")
            return {"status": "error", "message": f"Order failed: {result.comment}", "retcode": result.retcode, "details": result._asdict()}
        else:
            self._log(f"Order successful for {symbol} {trade_type}. Order ID: {result.order}, Deal ID: {result.deal}")
            self.current_open_trades[result.order] = {
                "symbol": symbol,
                "type": trade_type,
                "volume": volume,
                "open_price": result.price,
                "sl": request.get("sl", 0.0),
                "tp": request.get("tp", 0.0),
                "ticket": result.order,
                "time_open": datetime.now().isoformat()
            }
            return {"status": "success", "message": "Order executed", "order_id": result.order, "details": result._asdict()}

    def update_position_sl_tp(self, ticket: int, new_sl: float, new_tp: float) -> dict:
        """
        Modifies the Stop Loss and Take Profit levels of an existing position.

        Args:
            ticket (int): The ticket number (ID) of the position to modify.
            new_sl (float): The new Stop Loss price. Use 0.0 or a very small number to remove SL.
            new_tp (float): The new Take Profit price. Use 0.0 or a very small number to remove TP.

        Returns:
            dict: A dictionary containing the modification status.
        """
        if not self.connected:
            self._log("MT5 not connected. Cannot modify position SL/TP.")
            return {"status": "error", "message": "MT5 not connected"}

        positions = self.mt5.positions_get(ticket=ticket)
        if not positions:
            self._log(f"[WARN] Position with ticket {ticket} not found. Cannot modify SL/TP.")
            return {"status": "error", "message": f"Position {ticket} not found"}
        pos = positions[0]

        symbol_info = self.mt5.symbol_info(pos.symbol)
        if symbol_info is None:
            self._log(f"[ERROR] Failed to get symbol info for {pos.symbol} for SL/TP modification.")
            return {"status": "error", "message": f"Could not get symbol info for {pos.symbol}"}

        digits = symbol_info.digits
        rounded_sl = round(new_sl, digits) if new_sl > 0 else 0.0
        rounded_tp = round(new_tp, digits) if new_tp > 0 else 0.0

        if (abs(pos.sl - rounded_sl) < 1e-9) and (abs(pos.tp - rounded_tp) < 1e-9):
            self._log(f"[INFO] SL/TP for position {ticket} (Symbol: {pos.symbol}) already match requested values. No modification needed. Current SL: {pos.sl:.{digits}f}, TP: {pos.tp:.{digits}f}. Requested SL: {rounded_sl:.{digits}f}, TP: {rounded_tp:.{digits}f}")
            return {"status": "success", "message": "SL/TP already set to requested values, no action taken."}

        self._log(f"Modifying position {ticket} (Symbol: {pos.symbol}). Old SL: {pos.sl:.{digits}f}, TP: {pos.tp:.{digits}f}. New SL: {rounded_sl:.{digits}f}, TP: {rounded_tp:.{digits}f}")

        result = self.mt5.modify_position(ticket, rounded_sl, rounded_tp)

        if result is None:
            err_msg = self.mt5.last_error()
            self._log(f"[ERROR] Failed to modify position {ticket}: {err_msg}")
            return {"status": "error", "message": f"Failed to modify position: {err_msg}"}
        elif result.retcode != self.mt5.TRADE_RETCODE_DONE:
            self._log(f"[ERROR] Position modification failed for ticket {ticket} (retcode={result.retcode}): {result.comment}")
            return {"status": "error", "message": f"Position modification failed: {result.comment}", "retcode": result.retcode, "details": result._asdict()}
        else:
            self._log(f"Position {ticket} SL/TP modified successfully.")
            if ticket in self.current_open_trades:
                self.current_open_trades[ticket]['sl'] = rounded_sl
                self.current_open_trades[ticket]['tp'] = rounded_tp
            return {"status": "success", "message": "Position SL/TP modified", "details": result._asdict()}

    def close_order(self, order_ticket: int, comment: str = "CLOSE_ALL"):
        if not self.connected:
            self._log("MT5 not connected. Cannot close order.")
            return {"status": "error", "message": "MT5 not connected"}

        positions = self.mt5.positions_get(ticket=order_ticket)
        if not positions:
            self._log(f"[WARN] Position with ticket {order_ticket} not found.")
            return {"status": "error", "message": f"Position {order_ticket} not found"}

        pos = positions[0]
        symbol = pos.symbol

        #  WAJIB
        if not self.mt5.symbol_select(symbol, True):
            self._log(f"[ERROR] Failed to select symbol {symbol}")
            return {"status": "error", "message": "symbol_select failed"}

        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None:
            self._log(f"[ERROR] Could not get tick for {symbol} (ticket {order_ticket})")
            return {"status": "error", "message": "No tick data"}

        close_type = (
            self.mt5.ORDER_TYPE_SELL
            if pos.type == self.mt5.POSITION_TYPE_BUY
            else self.mt5.ORDER_TYPE_BUY
        )

        price = tick.bid if pos.type == self.mt5.POSITION_TYPE_BUY else tick.ask

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "deviation": int(self.SLIPPAGE),
            "magic": self.MAGIC_NUMBER,
            "comment": comment[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_FOK
        }

        self._log(f"Sending close order request: {request}")
        result = self.mt5.order_send(request)

        if result is None:
            err = self.mt5.last_error()
            self._log(f"[ERROR] order_send returned None while closing {order_ticket}: {err}")
            return {"status": "error", "message": "order_send returned None", "last_error": err}

        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            self._log(f"[ERROR] Close failed for ticket {order_ticket} (retcode={result.retcode}): {result.comment}")
            return {"status": "error", "retcode": result.retcode, "details": result._asdict()}

        self._log(f"[SUCCESS] Closed position {order_ticket}")
        self.current_open_trades.pop(order_ticket, None)
        return {"status": "success", "order_id": result.order, "details": result._asdict()}

    def close_all_open_positions(self):
        self._log("[ALERT] Initiating BULLETPROOF CLOSE ALL sequence.")

        if not self.connected:
            self._log("[ERROR] MT5 not connected. Abort CLOSE ALL.")
            return {"status": "error", "message": "MT5 not connected"}

        positions = self.mt5.positions_get(magic=self.MAGIC_NUMBER)
        if not positions:
            self._log("[INFO] No open positions found to close.")
            return {"status": "success", "message": "No positions to close."}

        close_results = []

        for pos in positions:
            symbol = pos.symbol
            ticket = pos.ticket
            volume = pos.volume
            pos_type = pos.type  # POSITION_TYPE_BUY / POSITION_TYPE_SELL

            # 1. Ensure symbol is tradable
            if not self.mt5.symbol_select(symbol, True):
                self._log(f"[ERROR] symbol_select failed for {symbol}. Skipping ticket {ticket}.")
                close_results.append({"ticket": ticket, "status": "error", "message": "symbol_select failed"})
                continue

            # 2. Get tick data (MANDATORY)
            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                self._log(f"[ERROR] symbol_info_tick returned None for {symbol}. Ticket {ticket}.")
                close_results.append({"ticket": ticket, "status": "error", "message": "No tick data"})
                continue
            if pos_type == self.mt5.POSITION_TYPE_BUY:
                close_type = self.mt5.ORDER_TYPE_SELL
                price = tick.bid
                side = "BUY"
            elif pos_type == self.mt5.POSITION_TYPE_SELL:
                close_type = self.mt5.ORDER_TYPE_BUY
                price = tick.ask
                side = "SELL"
            else:
                self._log(f"[ERROR] Unknown position type for ticket {ticket}.")
                close_results.append({"ticket": ticket, "status": "error", "message": "Unknown position type"})
                continue

            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "position": ticket,
                "volume": volume,
                "type": close_type,
                "price": price,
                "deviation": int(self.SLIPPAGE),
                "magic": self.MAGIC_NUMBER,
                "comment": "GLOBAL_CLOSE_ALL",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_FOK
            }

            self._log(
                f"[CLOSE_ALL] Closing ticket {ticket} | {symbol} | {side} | vol={volume} | price={price}"
            )


            # 5. Send order
            result = self.mt5.order_send(request)

            if result is None:
                err = self.mt5.last_error()
                self._log(f"[ERROR] order_send returned None for ticket {ticket}: {err}")
                close_results.append({
                    "ticket": ticket,
                    "status": "error",
                    "message": "order_send None",
                    "last_error": err
                })
                continue

            self._log(f"[SUCCESS] Closed ticket {ticket} successfully.")
            close_results.append({
                "ticket": ticket,
                "status": "success",
                "order_id": result.order,
                "deal_id": result.deal
            })

            # Clean internal state
            self.current_open_trades.pop(ticket, None)

        self._log(f"[DONE] CLOSE ALL completed. Total attempts: {len(positions)}")
        return {
            "status": "completed",
            "attempted": len(positions),
            "results": close_results
        }


    def handle_trade_signal(self, signal: dict):
        self._log(f"Received trade signal: {signal}")
        status_update = {"signal_id": signal.get("signal_id", "N/A"), "status": "processed", "trade_result": "N/A"}

        if signal.get("action") == "CLOSE_ALL":
            self._log("[ALERT] Received signal to CLOSE ALL open positions!")
            close_all_result = self.close_all_open_positions()
            status_update["status"] = "close_all_executed"
            status_update["message"] = "Global close all positions signal processed."
            status_update["trade_result"] = close_all_result
            if send_trade_data_to_colab({"timestamp": datetime.now().isoformat(), "signal": signal, "status_update": status_update}, log_func=self._log):
                self._log("[INFO] Close all signal processing results successfully sent to Colab.")
            else:
                self._log("[WARN] Failed to send close all signal processing results to Colab.")
            return json.dumps(status_update, default=str)

        elif signal.get("action") == "MODIFY":
            ticket = signal.get("ticket")
            new_sl = signal.get("new_sl")
            new_tp = signal.get("new_tp")

            if not all([ticket, new_sl is not None, new_tp is not None]):
                self._log(f"[ERROR] Invalid MODIFY signal: missing ticket, new_sl, or new_tp. Signal: {signal}")
                status_update["status"] = "error"
                status_update["message"] = "Invalid MODIFY signal data"
            else:
                self._log(f"[INFO] Received MODIFY signal for ticket {ticket} to set SL: {new_sl}, TP: {new_tp}")
                modify_result = self.update_position_sl_tp(ticket, new_sl, new_tp)
                status_update["trade_result"] = modify_result
                if modify_result["status"] == "success":
                    status_update["status"] = "modified"
                    status_update["message"] = f"Position {ticket} SL/TP modified successfully."
                else:
                    status_update["status"] = "failed_modification"
                    status_update["message"] = f"Failed to modify position {ticket}. Reason: {modify_result.get('message', 'Unknown')}"

            if send_trade_data_to_colab({"timestamp": datetime.now().isoformat(), "signal": signal, "status_update": status_update}, log_func=self._log):
                self._log("[INFO] Position modification processing results successfully sent to Colab.")
            else:
                self._log("[WARN] Failed to send position modification processing results to Colab.")
            return json.dumps(status_update, default=str)

        elif signal.get("action") == "CLOSE":
            ticket = signal.get("ticket")
            reason = signal.get("reason", "MANUAL_CLOSE")

            if not ticket:
                status_update["status"] = "error"
                status_update["message"] = "CLOSE signal missing ticket"
                return json.dumps(status_update, default=str)

            self._log(f"[ALERT] Closing position {ticket}. Reason: {reason}")
            close_result = self.close_order(ticket, comment=reason)
            status_update["trade_result"] = close_result

            if close_result["status"] == "success":
                status_update["status"] = "closed"
                status_update["message"] = f"Position {ticket} closed successfully."
            else:
                status_update["status"] = "failed_close"
                status_update["message"] = f"Failed to close position {ticket}"
            return json.dumps(status_update, default=str)

        pair_name = signal.get('pair_name') or signal.get("symbol")
        blocked_pairs = getattr(parameter, 'BLOK_SIGNAL_FOR', set())
        if pair_name in blocked_pairs:
            self._log(f"[BLOCK] Signal for {pair_name} is BLOCKED based on parameter settings.")
            status_update["status"] = "blocked"
            status_update["message"] = f"Trading for {pair_name} is disabled in BLOK_SIGNAL_FOR"

            # Kirim info ke Colab agar monitor tahu sinyal diblokir
            send_trade_data_to_colab({
                "timestamp": datetime.now().isoformat(), 
                "signal": signal, 
                "status_update": status_update
            }, log_func=self._log)
            
            return json.dumps(status_update, default=str)

        if not self.trading_enabled and signal.get("action") not in ["CLOSE", "CLOSE_ALL"]:
            self._log("[BLOCK] Trading disabled by risk guard. Signal ignored.")
            return json.dumps({
                "status": "blocked",
                "message": "Trading disabled by equity/daily loss guard"
            })

        # Aturan News Master (Aturan 2.5.2.2)
        if self.news_manager.is_currently_restricted():
            self._log("[BLOCK] Signal blocked due to High Impact News.")
            return json.dumps({"status": "blocked", "message": "News Restricted Window (FF Rules)"})

        trade_type = signal.get('signal') or signal.get("action")
        entry_price = signal.get('entry_price')
        stop_loss = signal.get('stop_loss', 0.0)
        take_profit = signal.get('take_profit', 0.0)
        position_units = signal.get('position_units', 0)
        reason = signal.get('reason')

        if not all([pair_name, trade_type, entry_price]) or position_units <= 0:
            self._log(f"Invalid signal received, missing critical data or invalid position units: {signal}")
            status_update["status"] = "error"
            status_update["message"] = "Invalid signal data or position units"
            if send_trade_data_to_colab({"timestamp": datetime.now().isoformat(), "signal": signal, "status_update": status_update}, log_func=self._log):
                self._log("[INFO] Trade signal processing results (invalid signal) successfully sent to Colab.")
            else:
                self._log("[WARN] Failed to send trade signal processing results (invalid signal) to Colab.")
            return json.dumps(status_update, default=str)

        mt5_symbol = parameter.PAIRS.get(pair_name, pair_name)
        if mt5_symbol == pair_name:
            self._log(f"[WARN] Symbol mapping not found for {pair_name}. Attempting to use default name: {mt5_symbol}")

        existing_positions = self.mt5.positions_get(symbol=mt5_symbol, magic=self.MAGIC_NUMBER)
        if existing_positions and len(existing_positions) > 0:
            self._log(f"[WARN] Found existing open position for {pair_name}. Ticket: {existing_positions[0].ticket}. Not acting on new signal yet. Implement closing logic if desired.")
            return

        if trade_type == "BUY" or trade_type == "SELL":
            order_result = self.send_order(
                symbol=mt5_symbol,
                trade_type=trade_type,
                volume=round(position_units, 2),
                price=entry_price,
                stop_loss=stop_loss if not np.isnan(stop_loss) else 0.0,
                take_profit=take_profit if not np.isnan(take_profit) else 0.0,
                comment = sanitize_comment(signal["signal_id"])
            )
            status_update["trade_result"] = order_result
            if order_result["status"] == "success":
                status_update["status"] = "executed"
                status_update["message"] = f"Trade {trade_type} executed for {pair_name}"
            else:
                status_update["status"] = "failed_execution"
                status_update["message"] = f"Failed to execute trade for {pair_name}. Reason: {order_result.get('message', 'Unknown')}"

        elif trade_type == "HOLD":
            self._log(f"Signal for {pair_name} is HOLD. No action taken.")
            status_update["status"] = "held"
            status_update["message"] = "No trade action for HOLD signal"
        else:
            self._log(f"Unknown trade type received: {trade_type}")
            status_update["status"] = "error"
            status_update["message"] = "Unknown trade type"

        if send_trade_data_to_colab({"timestamp": datetime.now().isoformat(), "signal": signal, "status_update": status_update}, log_func=self._log):
            self._log("[INFO] Trade signal processing results successfully sent to Colab.")
        else:
            self._log("[WARN] Failed to send trade signal processing results (invalid signal) to Colab.")

        return json.dumps(status_update, default=str)

    def monitor_open_trades(self):
        """
        Periodically checks open trades initiated by this engine (using MAGIC_NUMBER)
        and performs emergency exits if necessary. Also updates self.current_open_trades.
        """
        if not self.connected:
            self._log("MT5 not connected. Cannot monitor trades.")
            return

        self._log("Monitoring open trades...")

        account_info = self.mt5.account_info()
        if account_info is None:
            self._log("[ERROR] Failed to retrieve account info. Skipping equity guard.")
            return

        open_trades_list = []
        current_equity = account_info.equity

        current_balance = account_info.balance
        server_time = datetime.now() # Sesuaikan dengan waktu server jika perlu

        # --- LOGIKA RESET DAILY EQUITY (Aturan 2.3) ---
        # Reset setiap jam 00:00 CE(S)T
        current_day = server_time.day
        if self.last_day_reset != current_day:
            # Gunakan yang lebih tinggi antara Balance atau Equity (Aturan 2.3)
            self.daily_equity_start = max(current_equity, current_balance)
            self.last_day_reset = current_day
            self.equity_peak = current_equity # Reset peak harian
            if self.last_day_reset != current_day:
                self.trading_enabled = True
                self._log("[INFO] Trading re-enabled for new trading day.")
            self._log(f"[DAILY RESET] New Daily Start: {self.daily_equity_start:.2f}")

        # --- HITUNG DAILY LOSS ---
        daily_loss_pct = (self.daily_equity_start - current_equity) / self.daily_equity_start

        # --- HARD STOP DAILY LOSS (4.5%) ---
        if daily_loss_pct >= self.DAILY_LOSS_TOLERANCE:
            self._log(f"[CRITICAL] Daily Loss Limit reached ({daily_loss_pct:.2%}). Shutting down for today.")
            self.close_all_open_positions()
            self.trading_enabled = False # Matikan trading sampai besok
            return

        # Initialize equity peak once
        if self.equity_peak is None:
            self.equity_peak = current_equity
            self._log(f"[INFO] Equity peak initialized at {self.equity_peak:.2f}")

        # Update equity peak only if new high
        if current_equity > self.equity_peak:
            self.equity_peak = current_equity

        drawdown_pct = (self.equity_peak - current_equity) / self.equity_peak

        financial_data = {
            "timestamp": datetime.now().isoformat(),
            "equity": float(current_equity),
            "equity_peak": float(self.equity_peak),
            "drawdown": float(drawdown_pct * 100), # Dalam persen
            "daily_loss_pct": float(daily_loss_pct * 100),
            "open_trades_summary": open_trades_list, # List yang sudah Anda buat
            "trading_enabled": self.trading_enabled
        }

        self._log(
            f"[INFO] Equity Monitor | "
            f"Equity={current_equity:.2f} | "
            f"Peak={self.equity_peak:.2f} | "
            f"DD={drawdown_pct:.2%}"
        )

        final_payload = {
            "timestamp": datetime.now().isoformat(),
            "equity_data": financial_data, # Data finansial
            "open_trades_summary": open_trades_list,    # Daftar posisi
            "status": "ACTIVE",
            "current_open_trades_engine_tracked": self.current_open_trades
        }

        if send_trade_data_to_colab(final_payload, log_func=self._log):
            self._log("[INFO] Open trades summary successfully sent to Colab.")
        else:
            self._log("[WARN] Failed to send open trades summary to Colab.")

        # --- HARD STOP: Emergency equity exit
        if drawdown_pct >= self.EQUITY_DRAWDOWN_TOLERANCE:
            self._log(
                f"[EMERGENCY] EQUITY DRAWDOWN BREACHED! "
                f"{drawdown_pct:.2%} >= {self.EQUITY_DRAWDOWN_TOLERANCE:.2%} | "
                f"Triggering CLOSE ALL."
            )

            close_reulst = self.close_all_open_positions()
            self._log(f"[EMERGENCY] CLOSE ALL result: {close_result}")

            # Reset internal tracking
            self.current_open_trades.clear()
            self.trading_enabled = False # Matikan trading sampai besok
            return

        positions = self.mt5.positions_get(magic=self.MAGIC_NUMBER)

        if positions is None:
            self._log(f"No positions found or error: {self.mt5.last_error()}")
            return

        active_tickets = {pos.ticket for pos in positions}
        self.current_open_trades = {ticket: trade for ticket, trade in self.current_open_trades.items() if ticket in active_tickets}

        if len(positions) == 0 and not self.current_open_trades:
            self._log("No open positions found initiated by this engine.")
            return


        for pos in positions:
            ticket = pos.ticket
            symbol = pos.symbol
            volume = pos.volume
            open_price = pos.price_open
            sl = pos.sl
            tp = pos.tp

            # --- Ensure symbol is selected in Market Watch
            if not self.mt5.symbol_select(symbol, True):
                self._log(f"[ERROR] Failed to select symbol {symbol} (Ticket: {ticket}). Skipping.")
                continue

            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                self._log(f"[ERROR] No tick data for {symbol} (Ticket: {ticket}). Skipping.")
                continue

            # --- Determine position type & correct close price
            if pos.type == self.mt5.POSITION_TYPE_BUY:
                position_type = "BUY"
                current_price = tick.bid
            elif pos.type == self.mt5.POSITION_TYPE_SELL:
                position_type = "SELL"
                current_price = tick.ask
            else:
                self._log(f"[ERROR] Unknown position type for ticket {ticket}. Skipping.")
                continue

            if current_price is None or current_price <= 0:
                self._log(
                    f"[ERROR] Invalid market price for {symbol} "
                    f"(Ticket: {ticket}, Price: {current_price}). Skipping."
                )
                continue

            # --- Logging snapshot
            self._log(
                f"[INFO] Ticket: {ticket} | {symbol} | {position_type} | "
                f"Vol: {volume} | Open: {open_price:.5f} | "
                f"Price: {current_price:.5f} | SL: {sl} | TP: {tp}"
            )

            open_trades_list.append({
                "ticket": ticket,
                "symbol": symbol,
                "type": position_type,
                "volume": volume,
                "open_price": open_price,
                "current_price": current_price,
                "sl": sl,
                "tp": tp
            })

            # --- Emergency exit logic (hard risk guard)
            adverse_move = False

            if position_type == "BUY" and current_price <= open_price * 0.95:
                adverse_move = True
            elif position_type == "SELL" and current_price >= open_price * 1.05:
                adverse_move = True

            if adverse_move:
                self._log(
                    f"[EMERGENCY] Adverse movement detected! "
                    f"{symbol} | Ticket {ticket} | Initiating close."
                )

                close_result = self.close_order(
                    order_ticket=ticket,
                    symbol=symbol,
                    trade_type=position_type,
                    volume=volume,
                    price=current_price,
                    comment="Emergency exit: adverse price movement"
                )

                self._log(f"[EMERGENCY] Close result for {ticket}: {close_result}")


    def start_monitoring_loop(self, interval_sec=5):
        def monitor_loop():
            self._log("[MONITOR] Risk monitoring thread started.")
            while self.connected:
                try:
                    # Sync berita jika hari sudah berganti
                    current_date = datetime.now().date()
                    if self.news_manager.last_sync_date != current_date:
                        self.news_manager.sync_news()

                    self.monitor_open_trades()
                except Exception as e:
                    self._log(f"[ERROR] monitor_open_trades crashed: {e}")
                time.sleep(interval_sec)

        t = threading.Thread(
            target=monitor_loop,
            daemon=True
        )
        t.start()

@app.route("/receive_signal", methods=["POST"])
def receive_signal():
    received_api_key = request.headers.get("X-API-Key")
    if received_api_key != TRADE_ENGINE_API_KEY:
        return jsonify({"status": "error", "message": "Unauthorized: Invalid API Key"}), 401

    try:
        signal_data = request.get_json()
        if not signal_data:
            raise ValueError("No JSON signal data received.")

        global trade_engine_instance
        if trade_engine_instance is None:
            trade_engine_instance = TradeEngine(log_output_path=os.path.join(os.path.dirname(__file__), "trade_engine_log.txt"))

        response = trade_engine_instance.handle_trade_signal(signal_data)
        return jsonify(response), 200

    except Exception as e:
        if 'trade_engine_instance' in globals() and trade_engine_instance is not None:
            trade_engine_instance._log(f"[ERROR] Error processing received signal: {e}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Error processing signal: {e}"}), 500

trade_engine_instance = None
flask_trade_engine_thread = None

def start_trade_engine_flask_and_monitor(log_output_path: str):
    global trade_engine_instance, flask_trade_engine_thread

    if trade_engine_instance is None:
        trade_engine_instance = TradeEngine(log_output_path=log_output_path)

    if not trade_engine_instance.connected:
        trade_engine_instance._log("Trade Engine cannot start without MT5 connection. Exiting.")
        return False

    trade_engine_instance.start_monitoring_loop(interval_sec=5)

    if flask_trade_engine_thread is None or not flask_trade_engine_thread.is_alive():
        def run_flask_app_local():
            os.environ['WERKZEUG_RUN_MAIN'] = 'false'
            app.run(host="127.0.0.1", port=TRADE_ENGINE_API_PORT, debug=False, use_reloader=False)

        flask_trade_engine_thread = threading.Thread(target=run_flask_app_local)
        flask_trade_engine_thread.daemon = True
        flask_trade_engine_thread.start()

    TRADE_ENGINE_LOG_FILE = os.path.join(os.path.dirname(__file__), "trade_engine_log.txt")

    trade_engine_instance._log(f"Flask API started on port {TRADE_ENGINE_API_PORT} in a separate thread.")
    time.sleep(1)

    trade_engine_instance._log("Trade Engine is running. Waiting for signals or monitoring trades...")
    return True
