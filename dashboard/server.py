#!/usr/bin/env python3
import hashlib
import json
import os
import re
import secrets
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
USERS_FILE = DATA_DIR / "users.json"
INVENTORY_FILE = DATA_DIR / "inventory.json"

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
ORDER_RE = re.compile(r"\[(\d{2}:\d{2}:\d{2})\].*?Order placed - (BUY|SELL)")
TOTAL_RE = re.compile(r"Total:\s*\$(\d+(?:\.\d+)?)")
UNIT_PRICE_RE = re.compile(r"Unit Price:\s*\$(\d+(?:\.\d+)?)")
QTY_RE = re.compile(r"Quantity:\s*(\d+(?:\.\d+)?)")
CASH_RE = re.compile(r"Cash:\s*\$(\d+(?:\.\d+)?)")
INVENTORY_RE = re.compile(r"Inventory:\s*(\d+(?:\.\d+)?)") 

state_lock = threading.Lock()
process_state = {
    "market": None,
    "agent": None,
    "product": "SOLUSDT",
    "agent_name": "Mean Reversion",
    "last_error": None,
    "started_at": None,
    "agent_log_offset": 0,  # byte offset in agent log when current agent started
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

# Inventory tracked per product across session switches
inventory_per_product: dict = {}

# Auth state
_users: dict = {}
sessions: dict = {}  # token → username


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def load_users() -> None:
    global _users
    if USERS_FILE.exists():
        try:
            with USERS_FILE.open("r") as f:
                _users = json.load(f)
            print(f"[Auth] Loaded {len(_users)} user(s) from {USERS_FILE.name}")
            return
        except Exception:
            pass
    _users = {"admin": hash_password("admin")}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(_users, indent=2), encoding="utf-8")
    print("[Auth] Default user created → admin / admin")


def check_session(handler) -> str | None:
    cookie = handler.headers.get("Cookie", "")
    for part in cookie.split(";"):
        part = part.strip()
        if part.startswith("session="):
            return sessions.get(part[8:])
    return None


def load_inventory() -> None:
    global inventory_per_product
    if INVENTORY_FILE.exists():
        try:
            with INVENTORY_FILE.open("r") as f:
                inventory_per_product = json.load(f)
            print(f"[Inventory] Restored {len(inventory_per_product)} product(s) from disk")
        except Exception:
            pass


def save_inventory() -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        INVENTORY_FILE.write_text(json.dumps(inventory_per_product, indent=2), encoding="utf-8")
    except Exception:
        pass


def strategy_from_agent(agent_name: str) -> str:
    mapping = {
        "Mean Reversion": "mean_reversion",
        "RSI": "rsi",
        "Bollinger": "bollinger",
        "SMA": "sma",
        "Random": "random",
        "A2C": "a2c",
        "SAC": "sac",
        "DQN": "dqn",
        "PPO": "ppo",
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


def kill_stale_agents():
    """Kill any lingering agents.main processes not tracked by process_state."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "agents.main"],
            capture_output=True, text=True
        )
        pids = []
        for pid_str in result.stdout.strip().splitlines():
            try:
                pid = int(pid_str)
                os.kill(pid, signal.SIGTERM)
                pids.append(pid)
            except (ValueError, ProcessLookupError):
                pass
        if pids:
            deadline = time.time() + 2
            while pids and time.time() < deadline:
                time.sleep(0.05)
                pids = [p for p in pids if _pid_alive(p)]
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    except Exception:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def tail_file(path: Path, line_limit: int = 80) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return [strip_ansi(line.rstrip("\n")) for line in lines[-line_limit:]]


def read_market_state(metrics=None):
    if not MARKET_STATE_FILE.exists():
        return {
            "ok": False,
            "message": "market_state.json not found",
            "symbol": process_state["product"],
        }

    try:
        with MARKET_STATE_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        actual_symbol = payload.get("symbol", "")
        if actual_symbol and actual_symbol != process_state["product"]:
            return {
                "ok": False,
                "message": f"Waiting for {process_state['product']} data",
                "symbol": process_state["product"],
            }

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
            if metrics is None:
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
    pending_qty = None
    pending_price = None

    with AGENT_LOG_FILE.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 65536))
        session_text = f.read().decode("utf-8", errors="ignore")

    for raw in session_text.splitlines():
        line = strip_ansi(raw).strip()
        order_match = ORDER_RE.search(line)
        if order_match:
            pending_time = order_match.group(1)
            pending_action = order_match.group(2).title()
            pending_qty = None
            pending_price = None
            continue

        if pending_action:
            qty_match = QTY_RE.search(line)
            if qty_match and pending_qty is None:
                pending_qty = float(qty_match.group(1))
                continue
            price_match = UNIT_PRICE_RE.search(line)
            if price_match and pending_price is None:
                pending_price = float(price_match.group(1))
                continue
            total_match = TOTAL_RE.search(line)
            if total_match:
                amount = float(total_match.group(1))
                signed = -amount if pending_action == "Buy" else amount
                actions.append({
                    "time": pending_time or datetime.now().strftime("%H:%M:%S"),
                    "action": pending_action,
                    "result": round(signed, 2),
                    "qty": pending_qty,
                    "price": pending_price,
                })
                pending_action = None
                pending_time = None
                pending_qty = None
                pending_price = None

    dedup = []
    for row in actions:
        if not dedup or row != dedup[-1]:
            dedup.append(row)
    return dedup[-limit:]


def read_chart_payload():
    with state_lock:
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
        "tag": f"${last['c']:.2f}  {last['t']}",
    }


def read_performance(limit: int = 30):
    with state_lock:
        series = list(perf_state["series"][-limit:])
        ticks = list(perf_state["ticks"][-limit:])
    return {"series": series, "ticks": ticks}


def read_last_decision():
    if not AGENT_LOG_FILE.exists():
        return {"action": "—", "time": "—"}
    offset = process_state.get("agent_log_offset", 0)
    with AGENT_LOG_FILE.open("rb") as f:
        full = f.read()
    tail = full[offset:][-4096:].decode("utf-8", errors="ignore")
    last_action, last_time = None, None
    for line in tail.splitlines():
        m = ORDER_RE.search(strip_ansi(line))
        if m:
            last_time = m.group(1)
            last_action = m.group(2)
    return {"action": last_action or "HOLD", "time": last_time or "—"}


def read_snapshot():
    """Single endpoint returning all dashboard data — replaces 8 parallel requests."""
    if _reset_in_progress:
        empty_chart = {"candles": [], "axis": [], "markers": [],
                       "tag": "Waiting for market data", "symbol": process_state["product"]}
        empty_market = {"ok": False, "message": "reset in progress", "symbol": process_state["product"]}
        with state_lock:
            status = {
                "market_running": process_alive(process_state["market"]),
                "agent_running": process_alive(process_state["agent"]),
                "product": process_state["product"],
                "agent": process_state["agent_name"],
                "last_error": process_state["last_error"],
                "started_at": process_state["started_at"],
            }
        return {"market": empty_market, "trades": [], "performance": {"series": [], "ticks": []},
                "status": status, "metrics": {"cash": None, "inventory": None, "trade_count": 0},
                "logs": {"agent": [], "market": []}, "chart": empty_chart,
                "history": {"trades": 0, "buys": 0, "sells": 0, "net": 0.0, "sell_ratio": 0.0}}

    metrics = read_runtime_metrics()
    market = read_market_state(metrics=metrics)
    market["symbol"] = process_state["product"]

    all_trades = read_trade_log(limit=300)
    trade_log = all_trades[-25:]

    buys = len([r for r in all_trades if r["action"] == "Buy"])
    sells = len([r for r in all_trades if r["action"] == "Sell"])
    total = len(all_trades)
    profitable = [r for r in all_trades if r["result"] > 0]
    lossy = [r for r in all_trades if r["result"] < 0]
    history = {
        "trades": total,
        "buys": buys,
        "sells": sells,
        "net": round(sum(r["result"] for r in all_trades), 2),
        "sell_ratio": round((sells / total) * 100, 2) if total else 0.0,
        "win_rate": round(len(profitable) / total * 100, 1) if total else 0.0,
        "avg_profit": round(sum(r["result"] for r in profitable) / len(profitable), 2) if profitable else 0.0,
        "avg_loss": round(sum(r["result"] for r in lossy) / len(lossy), 2) if lossy else 0.0,
    }

    with state_lock:
        candles = list(chart_state["candles"])
        if chart_state["current"] is not None:
            candles = candles + [chart_state["current"]]
    markers = [
        {"label": f"{r['action']} Order", "side": r["action"].lower(), "time": r["time"]}
        for r in all_trades[-6:]
    ]
    if candles:
        prices = [c["h"] for c in candles] + [c["l"] for c in candles]
        lo, hi = min(prices), max(prices)
        pad = max(0.6, (hi - lo) * 0.15)
        step = max(0.5, (hi - lo + 2 * pad) / 5)
        axis = [round((lo - pad) + i * step, 1) for i in range(6)]
        last = candles[-1]
        chart = {
            "candles": candles[-24:], "axis": axis, "markers": markers,
            "tag": f"${last['c']:.2f}  {last['t']}",
            "symbol": process_state["product"],
        }
    else:
        chart = {"candles": [], "axis": [], "markers": [],
                 "tag": "Waiting for market data", "symbol": process_state["product"]}

    with state_lock:
        perf = {"series": list(perf_state["series"][-30:]), "ticks": list(perf_state["ticks"][-30:])}
        status = {
            "market_running": process_alive(process_state["market"]),
            "agent_running": process_alive(process_state["agent"]),
            "product": process_state["product"],
            "agent": process_state["agent_name"],
            "last_error": process_state["last_error"],
            "started_at": process_state["started_at"],
        }

    return {
        "market": market,
        "trades": trade_log,
        "performance": perf,
        "status": status,
        "metrics": metrics,
        "logs": {"agent": tail_file(AGENT_LOG_FILE, 60), "market": tail_file(MARKET_LOG_FILE, 60)},
        "chart": chart,
        "history": history,
        "last_decision": read_last_decision(),
    }


def read_runtime_metrics():
    latest_cash = None
    latest_inventory = None
    trades = 0

    if not AGENT_LOG_FILE.exists():
        return {"cash": None, "inventory": None, "trade_count": 0}

    offset = process_state.get("agent_log_offset", 0)
    with AGENT_LOG_FILE.open("rb") as f:
        full_content = f.read()

    # Count trades only from current agent's portion
    current_content = full_content[offset:]
    text = current_content.decode("utf-8", errors="ignore")
    trades = len(ORDER_RE.findall(text))

    # Only scan last 8 KB of current agent's log for cash/inventory
    tail = current_content[-8192:].decode("utf-8", errors="ignore")
    for line in tail.splitlines():
        line = strip_ansi(line)
        m = CASH_RE.search(line)
        if m:
            latest_cash = float(m.group(1))
        m = INVENTORY_RE.search(line)
        if m:
            latest_inventory = float(m.group(1))

    # Persist inventory per product
    current_product = process_state["product"]
    if latest_inventory is not None:
        inventory_per_product[current_product] = latest_inventory
        save_inventory()

    return {
        "cash": latest_cash,
        "inventory": inventory_per_product.get(current_product),
        "trade_count": trades,
        "inventory_per_product": dict(inventory_per_product),
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
        # Record log offset before the new agent starts writing
        offset = AGENT_LOG_FILE.stat().st_size if AGENT_LOG_FILE.exists() else 0
        process_state["agent_log_offset"] = offset
        # Initialize inventory to 0 for this product if never seen before
        if product not in inventory_per_product:
            inventory_per_product[product] = 0.0

        agent_log = AGENT_LOG_FILE.open("a", encoding="utf-8")

        process_state["agent"] = subprocess.Popen(
            ["bash", "run_agent.sh", strategy],
            cwd=ROOT_DIR,
            stdout=agent_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "SYMBOL": product},
        )


_reset_in_progress = False


class DashboardHandler(BaseHTTPRequestHandler):
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
        try:
            self._do_GET_inner()
        except Exception as exc:
            try:
                self._send_json({"error": str(exc)}, status=500)
            except Exception:
                pass

    def _do_GET_inner(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Public paths — no auth required
        _PUBLIC = {"/login", "/login.html"}
        if path not in _PUBLIC:
            if check_session(self) is None:
                self.send_response(302)
                self.send_header("Location", "/login")
                self.end_headers()
                return

        if path in ("/login", "/login.html"):
            self._serve_file("login.html")
            return

        # Endpoint pour vérifier si le reset est en cours
        if path == "/api/snapshot":
            snap = read_snapshot()
            snap["username"] = check_session(self) or "anonymous"
            self._send_json(snap)
            return

        if path == "/api/reset_status":
            self._send_json({"reset": _reset_in_progress})
            return

        if path == "/api/market_state":
            # Si reset en cours, renvoyer un état vide/tag Waiting
            if _reset_in_progress:
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
            if _reset_in_progress:
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

        if path in ["/", "/index.html"]:
            self._serve_file("index.html")
            return

        self._serve_file(path.lstrip("/"))

    def do_POST(self):
        global _reset_in_progress
        parsed = urlparse(self.path)

        if parsed.path == "/api/login":
            content_len = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(content_len).decode("utf-8") if content_len else "{}"
            try:
                payload = json.loads(body)
            except Exception:
                payload = {}
            username = payload.get("username", "").strip()
            password = payload.get("password", "")
            if username in _users and _users[username] == hash_password(password):
                token = secrets.token_hex(32)
                sessions[token] = username
                resp_body = json.dumps({"ok": True, "username": username}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Set-Cookie", f"session={token}; HttpOnly; Path=/; SameSite=Strict")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
            else:
                self._send_json({"ok": False, "message": "Invalid credentials"}, status=401)
            return

        if parsed.path == "/api/logout":
            cookie = self.headers.get("Cookie", "")
            for part in cookie.split(";"):
                part = part.strip()
                if part.startswith("session="):
                    sessions.pop(part[8:], None)
            self._send_json({"ok": True})
            return

        # Auth check for all other POST endpoints
        if check_session(self) is None:
            self._send_json({"error": "Unauthorized"}, status=401)
            return

        if parsed.path == "/api/clear_logs":
            _reset_in_progress = True
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
                    _reset_in_progress = False
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
            max_wait = 5.0
            waited = 0.0
            while waited < max_wait:
                if (not MARKET_STATE_FILE.exists() or MARKET_STATE_FILE.stat().st_size == 0):
                    break
                time.sleep(0.1)
                waited += 0.1
            _reset_in_progress = False
            self._send_json({"ok": True, "cleared": cleared})
            return

        if parsed.path == "/api/start_simulation":
            try:
                content_len = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(content_len).decode("utf-8") if content_len else "{}"
                payload = json.loads(body)

                product = payload.get("product", "SOLUSDT")
                agent_name = payload.get("agent", "Mean Reversion")

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

                kill_stale_agents()

                with state_lock:
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

        if parsed.path == "/api/start_market":
            try:
                content_len = int(self.headers.get("Content-Length", "0") or "0")
                body = self.rfile.read(content_len).decode("utf-8") if content_len else "{}"
                payload = json.loads(body)
                product = payload.get("product", "SOLUSDT")

                with state_lock:
                    process_state["product"] = product
                    if process_alive(process_state["market"]):
                        kill_process_group(process_state["market"])
                        process_state["market"] = None
                    chart_state["candles"] = []
                    chart_state["current"] = None
                    chart_state["tick"] = 0
                    chart_state["markers"] = []
                    chart_state["last_mid"] = None
                    chart_state["symbol"] = product
                    perf_state["baseline_equity"] = None
                    perf_state["series"] = []
                    perf_state["ticks"] = []

                try:
                    if MARKET_STATE_FILE.exists():
                        MARKET_STATE_FILE.unlink()
                except Exception:
                    pass

                if product not in inventory_per_product:
                    inventory_per_product[product] = 0.0

                start_market(product)
                self._send_json({"ok": True, "message": f"Market started for {product}"})
            except Exception as exc:
                self._send_json({"ok": False, "message": str(exc)}, status=500)
            return

        if parsed.path == "/api/stop_simulation":
            try:
                with state_lock:
                    market_proc = process_state["market"]
                    agent_proc = process_state["agent"]
                    process_state["market"] = None
                    process_state["agent"] = None
                kill_process_group(market_proc)
                kill_process_group(agent_proc)
                self._send_json({"ok": True, "message": "Simulation stopped"})
            except Exception as exc:
                with state_lock:
                    process_state["last_error"] = str(exc)
                self._send_json({"ok": False, "message": f"Failed to stop simulation: {exc}"}, status=500)
            return

        self.send_error(404, "Not found")


def run_server(port: int = 8090):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    load_users()
    load_inventory()
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
