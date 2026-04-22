import requests
import pytest

BASE_URL = "http://localhost:8000/api"

@pytest.mark.parametrize("endpoint", [
    "/market_state",
    "/trade_log",
    "/performance",
    "/status",
    "/metrics",
    "/recent_logs",
    "/chart",
    "/history",
])
def test_api_endpoints_available(endpoint):
    resp = requests.get(BASE_URL + endpoint)
    assert resp.status_code == 200, f"{endpoint} not available"
    assert resp.headers["content-type"].startswith("application/json")


def test_start_and_stop_simulation():
    # Start simulation
    resp = requests.post(BASE_URL + "/start_simulation", json={"product": "SOLUSDT", "agent": "RL Agent"})
    assert resp.status_code == 200
    # Stop simulation
    resp = requests.post(BASE_URL + "/stop_simulation")
    assert resp.status_code == 200


def test_clear_logs():
    resp = requests.post(BASE_URL + "/clear_logs")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("ok") is not None


def test_market_state_content():
    resp = requests.get(BASE_URL + "/market_state")
    data = resp.json()
    assert "ok" in data
    assert "symbol" in data
    # Optionally check for ask/bid if available
    if data["ok"]:
        assert "ask_price" in data
        assert "bid_price" in data
