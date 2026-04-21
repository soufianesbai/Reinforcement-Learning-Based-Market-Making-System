RESET_MODE = False
#!/usr/bin/env python3
import json
import os
import re
import signal
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = ROOT_DIR / "dashboard"
DATA_DIR = ROOT_DIR / "data"
MARKET_STATE_FILE = DATA_DIR / "market_state.json"
MARKET_LOG_FILE = DATA_DIR / "dashboard_market.log"
AGENT_LOG_FILE = DATA_DIR / "dashboard_agent.log"

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
ORDER_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\].*?Order placed - (BUY|SELL)")
TOTAL_RE = re.compile(r"Total:\s*\$(\d+(?:\.\d+)?)")
CASH_RE = re.compile(r"Cash:\s*\$(\d+(?:\.\d+)?)")
INVENTORY_RE = re.compile(r"Inventory:\s*(\d+(?:\.\d+)?)") 

state_lock = threading.Lock()
process_state = {
    "market": None,
    "agent": None,
    "product": "SOLUSDT",
    "agent_name": "RL Agent",
    "last_error": None,
    "started_at": None,
}

perf_state = {
    "baseline_equity": None,
    "series": [],
    "ticks": [],
}

chart_state = {
    "candles": [],
    "current": None,
    "tick": 0,
    "window": 4,
    "markers": [],
    "last_mid": None,
    "symbol": None,
}


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def strategy_from_agent(agent_name: str) -> str:
    mapping = {
        "RL Agent": "mean_reversion",
        "Mean Reversion": "mean_reversion",
        "RSI": "rsi",
        "Bollinger": "bollinger",
    }
    return mapping.get(agent_name, "mean_reversion")


def process_alive(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def kill_process_group(proc: subprocess.Popen | None):
    if not process_alive(proc):
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        deadline = time.time() + 3
        while process_alive(proc) and time.time() < deadline:
            time.sleep(0.05)
        if process_alive(proc):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def tail_file(path: Path, line_limit: int = 80) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return [strip_ansi(line.rstrip("\n")) for line in lines[-line_limit:]]


def read_market_state():
    if not MARKET_STATE_FILE.exists():
        return {
            "ok": False,
            "message": "market_state.json not found",
            "symbol": process_state["product"],
        }

    try:
        with MARKET_STATE_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        mid_raw = payload.get("mid_price")
        symbol = payload.get("symbol", process_state["product"])
        if isinstance(mid_raw, (int, float)):
            mid = mid_raw / 100.0

            # --- Sanity guard: reset chart if symbol changed or price jumped >15% ---
            with state_lock:
                last_mid = chart_state["last_mid"]
                last_sym = chart_state["symbol"]
                price_jump = (
                    last_mid is not None
                    and abs(mid - last_mid) / last_mid > 0.15
                )
                symbol_changed = last_sym is not None and last_sym != symbol
                if price_jump or symbol_changed:
                    chart_state["candles"] = []
                    chart_state["current"] = None
                    chart_state["tick"] = 0
                    chart_state["markers"] = []
                    perf_state["baseline_equity"] = None
                    perf_state["series"] = []
                    perf_state["ticks"] = []
                chart_state["last_mid"] = mid
                chart_state["symbol"] = symbol
            # -----------------------------------------------------------------------
            metrics = read_runtime_metrics()
            cash = metrics.get("cash")
            inventory = metrics.get("inventory")
            if cash is not None and inventory is not None:
                equity = round(cash + inventory * mid, 2)
                with state_lock:
                    if perf_state["baseline_equity"] is None:
                        perf_state["baseline_equity"] = equity
                    delta = round(equity - perf_state["baseline_equity"], 2)
                    series = perf_state["series"]
                    ticks = perf_state["ticks"]
                    if not series or series[-1] != delta:
                        series.append(delta)
                        ticks.append(datetime.now().strftime("%H:%M:%S"))
                        if len(series) > 40:
                            perf_state["series"] = series[-40:]
                            perf_state["ticks"] = ticks[-40:]

            with state_lock:
                chart_state["tick"] += 1
            current = chart_state["current"]
            if current is None:
                chart_state["current"] = {
                    "o": round(mid, 2),
                    "h": round(mid, 2),
                    "l": round(mid, 2),
                    "c": round(mid, 2),
                    "t": datetime.now().strftime("%H:%M:%S"),
                    "v": int(abs(payload.get("spread", 1)) * 10 + 100),
                }
            else:
                current["h"] = round(max(current["h"], mid), 2)
                current["l"] = round(min(current["l"], mid), 2)
                current["c"] = round(mid, 2)

            with state_lock:
                if chart_state["tick"] % chart_state["window"] == 0 and chart_state["current"] is not None:
                    chart_state["candles"].append(chart_state["current"])
                    if len(chart_state["candles"]) > 36:
                        chart_state["candles"] = chart_state["candles"][-36:]
                    chart_state["current"] = None

        payload["ok"] = True
        return payload
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Failed to parse market state: {exc}",
            "symbol": process_state["product"],
        }


def read_trade_log(limit: int = 25):
    if not AGENT_LOG_FILE.exists():
        return []

    actions = []
    pending_action = None
    pending_time = None

    with AGENT_LOG_FILE.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = strip_ansi(raw).strip()
            order_match = ORDER_RE.search(line)
            if order_match:
                pending_time = order_match.group(1)
                pending_action = order_match.group(2).title()
                continue

            if pending_action:
                total_match = TOTAL_RE.search(line)
                if total_match:
                    amount = float(total_match.group(1))
                    signed = -amount if pending_action == "Buy" else amount
                    actions.append(
                        {
                            "time": pending_time or datetime.now().strftime("%H:%M:%S"),
                            "action": pending_action,
                            "result": round(signed, 2),
                        }
                    )
                    pending_action = None
                    pending_time = None

    dedup = []
    for row in actions:
        if not dedup or row != dedup[-1]:
            dedup.append(row)
    return dedup[-limit:]


def read_chart_payload():
    candles = list(chart_state["candles"])
    if chart_state["current"] is not None:
        candles = candles + [chart_state["current"]]

    recent_rows = read_trade_log(limit=10)
    markers = []
    for row in recent_rows[-6:]:
        markers.append(
            {
                "label": f"{row['action']} Order",
                "side": row["action"].lower(),
                "time": row["time"],
            }
        )

    if not candles:
        return {
            "candles": [],
            "axis": [],
            "markers": [],
            "tag": "Waiting for market data",
            "symbol": chart_state["symbol"] or process_state["product"]
        }

    prices = [c["h"] for c in candles] + [c["l"] for c in candles]
    lo = min(prices)
    hi = max(prices)
    pad = max(0.6, (hi - lo) * 0.15)
    axis_start = (lo - pad)
    step = max(0.5, (hi - lo + 2 * pad) / 5)
    axis = [round(axis_start + i * step, 1) for i in range(6)]

    last = candles[-1]
    return {
        "candles": candles[-24:],
        "axis": axis,
        "markers": markers,
        "tag": f"Vol: {last['v']} | ${last['c']:.2f}  {last['t']}",
    }


def read_history_summary():
    rows = read_trade_log(limit=300)
    wins = len([r for r in rows if r["result"] > 0])
    losses = len([r for r in rows if r["result"] < 0])
    total = len(rows)
    pnl = round(sum(r["result"] for r in rows), 2)
    win_rate = round((wins / total) * 100, 2) if total else 0.0
    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "net": pnl,
        "win_rate": win_rate,
    }


def read_performance(limit: int = 30):
    with state_lock:
        series = list(perf_state["series"][-limit:])
        ticks = list(perf_state["ticks"][-limit:])
    return {"series": series, "ticks": ticks}


def read_runtime_metrics():
    latest_cash = None
    latest_inventory = None
    trades = 0

    if AGENT_LOG_FILE.exists():
        with AGENT_LOG_FILE.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = strip_ansi(raw)
                cash_match = CASH_RE.search(line)
                if cash_match:
                    latest_cash = float(cash_match.group(1))

                inv_match = INVENTORY_RE.search(line)
                if inv_match:
                    latest_inventory = float(inv_match.group(1))

                if ORDER_RE.search(line):
                    trades += 1

    return {
        "cash": latest_cash,
        "inventory": latest_inventory,
        "trade_count": trades,
    }


def start_market(product: str):
    with state_lock:
        market_proc = process_state["market"]
        if process_alive(market_proc):
            return

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        market_log = MARKET_LOG_FILE.open("a", encoding="utf-8")

        process_state["market"] = subprocess.Popen(
            ["bash", "run_market.sh", product, "1", "50"],
            cwd=ROOT_DIR,
            stdout=market_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )


def start_agent(agent_name: str, product: str):
    strategy = strategy_from_agent(agent_name)
    with state_lock:
        agent_proc = process_state["agent"]
        if process_alive(agent_proc):
            return

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        agent_log = AGENT_LOG_FILE.open("a", encoding="utf-8")

        process_state["agent"] = subprocess.Popen(
            ["bash", "run_agent.sh", strategy],
            cwd=ROOT_DIR,
            stdout=agent_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "SYMBOL": product},
        )


class DashboardHandler(BaseHTTPRequestHandler):
    reset_in_progress = False
    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, rel_path: str):
        file_path = (DASHBOARD_DIR / rel_path).resolve()
        if not str(file_path).startswith(str(DASHBOARD_DIR.resolve())) or not file_path.exists():
            self.send_error(404, "Not found")
            return

        mime = "text/plain"
        if file_path.suffix == ".html":
            mime = "text/html; charset=utf-8"
        elif file_path.suffix == ".css":
            mime = "text/css; charset=utf-8"
        elif file_path.suffix == ".js":
            mime = "application/javascript; charset=utf-8"

        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        # Endpoint pour vérifier si le reset est en cours
        if path == "/api/reset_status":
            self._send_json({"reset": self.reset_in_progress})
            return

        if path == "/api/market_state":
            # Si reset en cours, renvoyer un état vide/tag Waiting
            if self.reset_in_progress:
                self._send_json({
                    "ok": False,
                    "message": "market_state.json not ready",
                    "symbol": process_state["product"]
                })
                return
            payload = read_market_state()
            payload["symbol"] = process_state["product"]
            self._send_json(payload)
            return

        if path == "/api/trade_log":
            self._send_json({"rows": read_trade_log()})
            return

        if path == "/api/performance":
            self._send_json(read_performance())
            return

        if path == "/api/chart":
            # Si reset en cours, renvoyer un chart vide/tag Waiting
            if self.reset_in_progress:
                self._send_json({
                    "candles": [],
                    "axis": [],
                    "markers": [],
                    "tag": "Waiting for market data",
                    "symbol": process_state["product"]
                })
                return
            payload = read_chart_payload()
            payload["symbol"] = process_state["product"]
            if "tag" not in payload or not payload["tag"]:
                payload["tag"] = "Waiting for market data"
            self._send_json(payload)
            return

        if path == "/api/history":
            self._send_json(read_history_summary())
            return

        if path == "/api/status":
            with state_lock:
                self._send_json(
                    {
                        "market_running": process_alive(process_state["market"]),
                        "agent_running": process_alive(process_state["agent"]),
                        "product": process_state["product"],
                        "agent": process_state["agent_name"],
                        "last_error": process_state["last_error"],
                        "started_at": process_state["started_at"],
                    }
                )
            return

        if path == "/api/metrics":
            self._send_json(read_runtime_metrics())
            return

        if path == "/api/recent_logs":
            self._send_json(
                {
                    "agent": tail_file(AGENT_LOG_FILE, 60),
                    "market": tail_file(MARKET_LOG_FILE, 60),
                }
            )
            return

        if path == "/api/clear_logs":
            RESET_MODE = True
            cleared = []
            # Stopper le process market (LiveMarketData) pour éviter toute écriture résiduelle
            with state_lock:
                if process_alive(process_state["market"]):
                    kill_process_group(process_state["market"])
                    process_state["market"] = None
            for log_path in [AGENT_LOG_FILE, MARKET_LOG_FILE]:
                try:
                    if log_path.exists():
                        log_path.write_text("", encoding="utf-8")
                        cleared.append(log_path.name)
                except OSError as e:
                    self._send_json({"ok": False, "message": str(e)}, status=500)
                    return
            # Supprimer le fichier market_state.json pour forcer un état vide
            try:
                if MARKET_STATE_FILE.exists():
                    MARKET_STATE_FILE.unlink()
            except Exception:
                pass
            with state_lock:
                chart_state["candles"] = []
                chart_state["current"] = None
                chart_state["tick"] = 0
                chart_state["markers"] = []
                chart_state["last_mid"] = None
                chart_state["symbol"] = process_state["product"]
                perf_state["baseline_equity"] = None
                perf_state["series"] = []
                perf_state["ticks"] = []
            self._send_json({"ok": True, "cleared": cleared})
            return

        if path in ["/", "/index.html"]:
            self._serve_file("index.html")
            return

        self._serve_file(path.lstrip("/"))

    def do_POST(self):
        global RESET_MODE
        parsed = urlparse(self.path)

        if parsed.path == "/api/clear_logs":
            self.reset_in_progress = True
            cleared = []
            # Stopper le process market (LiveMarketData) pour éviter toute écriture résiduelle
            with state_lock:
                if process_alive(process_state["market"]):
                    kill_process_group(process_state["market"])
                    process_state["market"] = None
            for log_path in [AGENT_LOG_FILE, MARKET_LOG_FILE]:
                try:
                    if log_path.exists():
                        log_path.write_text("", encoding="utf-8")
                        cleared.append(log_path.name)
                except OSError as e:
                    self._send_json({"ok": False, "message": str(e)}, status=500)
                    self.reset_in_progress = False
                    return
            # Supprimer le fichier market_state.json pour forcer un état vide
            try:
                if MARKET_STATE_FILE.exists():
                    MARKET_STATE_FILE.unlink()
            except Exception:
                pass
            with state_lock:
                chart_state["candles"] = []
                chart_state["current"] = None
                chart_state["tick"] = 0
                chart_state["markers"] = []
                chart_state["last_mid"] = None
                chart_state["symbol"] = process_state["product"]
                perf_state["baseline_equity"] = None
                perf_state["series"] = []
                perf_state["ticks"] = []
            # Attente active : vérifier que les fichiers sont bien vides/supprimés
            import time
            max_wait = 5.0
            waited = 0.0
            while waited < max_wait:
                if (not MARKET_STATE_FILE.exists() or MARKET_STATE_FILE.stat().st_size == 0):
                    break
                time.sleep(0.1)
                waited += 0.1
            self.reset_in_progress = False
            self._send_json({"ok": True, "cleared": cleared})
            return

        if parsed.path == "/api/start_simulation":
            RESET_MODE = True
            try:
                content_len = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(content_len).decode("utf-8") if content_len else "{}"
                payload = json.loads(body)

                product = payload.get("product", "SOLUSDT")
                agent_name = payload.get("agent", "RL Agent")

                with state_lock:
                    process_state["product"] = product
                    process_state["agent_name"] = agent_name
                    process_state["last_error"] = None

                    if process_alive(process_state["market"]):
                        kill_process_group(process_state["market"])
                        process_state["market"] = None

                    if process_alive(process_state["agent"]):
                        kill_process_group(process_state["agent"])
                        process_state["agent"] = None

                    chart_state["candles"] = []
                    chart_state["current"] = None
                    chart_state["tick"] = 0
                    chart_state["markers"] = []
                    perf_state["baseline_equity"] = None
                    perf_state["series"] = []
                    perf_state["ticks"] = []
                    chart_state["last_mid"] = None
                    chart_state["symbol"] = product

                # Truncate log files so new session starts clean
                for log_path in [AGENT_LOG_FILE, MARKET_LOG_FILE]:
                    try:
                        if log_path.exists():
                            log_path.write_text("", encoding="utf-8")
                    except OSError:
                        pass
                # Supprimer le fichier market_state.json pour forcer un état vide
                try:
                    if MARKET_STATE_FILE.exists():
                        MARKET_STATE_FILE.unlink()
                except Exception:
                    pass

                start_market(product)
                start_agent(agent_name, product)

                with state_lock:
                    process_state["started_at"] = datetime.now().isoformat(timespec="seconds")

                self._send_json({"ok": True, "message": "Simulation started"})
            except Exception as exc:
                with state_lock:
                    process_state["last_error"] = str(exc)
                self._send_json({"ok": False, "message": f"Failed to start simulation: {exc}"}, status=500)
            return

        if parsed.path == "/api/stop_simulation":
            try:
                with state_lock:
                    kill_process_group(process_state["market"])
                    kill_process_group(process_state["agent"])
                    process_state["market"] = None
                    process_state["agent"] = None
                self._send_json({"ok": True, "message": "Simulation stopped"})
            except Exception as exc:
                with state_lock:
                    process_state["last_error"] = str(exc)
                self._send_json({"ok": False, "message": f"Failed to stop simulation: {exc}"}, status=500)
            return

        self.send_error(404, "Not found")


def run_server(port: int = 8000):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    print(f"MarketMind dashboard server running: http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
    finally:
        with state_lock:
            kill_process_group(process_state["market"])
            kill_process_group(process_state["agent"])
        server.server_close()


if __name__ == "__main__":
    run_server()
