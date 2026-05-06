"""
Manual sanity checks for SAC shared state builder.

Usage:
    python -m agents.tests.test_sac_state_manual
"""

import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from agents.strategies.sac_agent import STATE_DIM, build_sac_state


def test_state_shape_and_dtype():
    log_rets = np.linspace(-0.01, 0.01, 10, dtype=np.float32)
    state = build_sac_state(
        inventory=1.5,
        log_rets=log_rets,
        volatility=0.002,
        buyer_frac=0.54,
        market_spread_bps=8.0,
        vol_ratio=1.2,
        inventory_scale=5.0,
    )

    assert state.shape == (STATE_DIM,), f"Expected shape {(STATE_DIM,)}, got {state.shape}"
    assert state.dtype == np.float32, f"Expected float32, got {state.dtype}"
    print("State shape/dtype validated")


def test_state_clipping():
    log_rets = np.array([10.0] * 10, dtype=np.float32)
    state = build_sac_state(
        inventory=999.0,
        log_rets=log_rets,
        volatility=999.0,
        buyer_frac=5.0,
        market_spread_bps=5000.0,
        vol_ratio=999.0,
        inventory_scale=5.0,
    )

    assert -1.0 <= state[0] <= 1.0, f"Inventory norm out of bounds: {state[0]}"
    assert np.all(state[1:11] <= 5.0) and np.all(state[1:11] >= -5.0), "Returns clipping failed"
    assert 0.0 <= state[11] <= 5.0, f"Volatility clip failed: {state[11]}"
    assert 0.0 <= state[12] <= 1.0, f"Buyer fraction clip failed: {state[12]}"
    assert 0.0 <= state[13] <= 5.0, f"Spread clip failed: {state[13]}"
    assert 0.0 <= state[14] <= 5.0, f"Vol ratio clip failed: {state[14]}"
    print("State clipping validated")


if __name__ == "__main__":
    test_state_shape_and_dtype()
    test_state_clipping()
    print("\nAll SAC state checks passed.")
