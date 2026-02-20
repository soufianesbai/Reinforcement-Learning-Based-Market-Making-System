import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

from ..interfaces import BaseStrategy, StrategySignal
from ..types import Side
from ..market_data import MarketData

# On réimporte l'architecture depuis le fichier d'entraînement pour charger les poids
from ..train_rl import DeepEncoder 

class RLAgentStrategy(BaseStrategy):
    def __init__(self, symbol: str = "SOLUSDT", model_path: str = "models/gcrl_actor.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.market_data = MarketData(window_size=10)
        self.model_loaded = False
        
        self.state_dim, self.goal_dim, self.action_dim = 4, 1, 3
        
        # Objectif dynamique : Revenir "Flat" (inventaire à zéro)
        self.target_inventory = np.array([0.0], dtype=np.float32)
        # Hack pour tracker l'inventaire en local
        self.current_inventory = 0.0

        # /!\ Pensez à ajuster le paramètre depth selon ce que vous avez mis dans l'entrainement (ex: 16 ou 64)
        self.actor = DeepEncoder(self.state_dim + self.goal_dim, self.action_dim, depth=16).to(self.device)
        
        if os.path.exists(model_path):
            self.actor.load_state_dict(torch.load(model_path, map_location=self.device))
            self.actor.eval() # OBLIGATOIRE avec LayerNorm en inférence
            self.model_loaded = True
            print(f"[*] Modèle RL chargé depuis {model_path}")
        else:
            print(f"[!] {model_path} introuvable. Exécutez 'python3 -m agents.train_rl' d'abord.")

        self.observation = None

    def update_data(self, market_state: Dict) -> None:
        self.market_data.update(market_state)
        
        if len(self.market_data.prices) < 3:
            return
            
        # Calcul d'imbalance sur le best bid/ask
        b_qty = sum(b['quantity'] for b in self.market_data.bids[:3]) if self.market_data.bids else 0
        a_qty = sum(a['quantity'] for a in self.market_data.asks[:3]) if self.market_data.asks else 0
        imb = (b_qty - a_qty) / (b_qty + a_qty) if (b_qty + a_qty) > 0 else 0

        # Normalisation IDENTIQUE au _get_obs() de l'entraînement
        norm_imb = np.clip(imb, -1.0, 1.0)
        norm_spread = (self.market_data.spread / self.market_data.mid_price * 10000) / 10.0 if self.market_data.mid_price else 0.0
        norm_dev = (self.market_data.get_price_deviation() or 0.0) / 2.0
        norm_inv = self.current_inventory / 50.0

        self.observation = np.array([norm_imb, norm_spread, norm_dev, norm_inv], dtype=np.float32)

    def compute_signal(self) -> StrategySignal:
        if self.observation is None or not self.model_loaded:
            return StrategySignal(None, 0, 0, "Wait", {})

        state_t = torch.tensor(self.observation).unsqueeze(0).to(self.device)
        goal_t = torch.tensor(self.target_inventory).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_idx = torch.argmax(self.actor(torch.cat([state_t, goal_t], dim=-1)), dim=-1).item()

        qty = 5.0
        mid = int(self.market_data.mid_price * 100)
        offset = int(self.market_data.spread * 100 / 2)
        
        if action_idx == 1:
            self.current_inventory += qty # Mock local
            return StrategySignal(Side.BUY, mid - offset, qty, f"RL BUY (Visant: {self.target_inventory[0]})", {"RL": "BUY"})
        elif action_idx == 2:
            self.current_inventory -= qty # Mock local
            return StrategySignal(Side.SELL, mid + offset, qty, f"RL SELL (Visant: {self.target_inventory[0]})", {"RL": "SELL"})
            
        return StrategySignal(None, 0, 0, "RL HOLD", {"RL": "HOLD"})

    def get_status(self) -> Dict[str, Any]:
        return {"name": "GCRL Deep Agent", "Target Inv": self.target_inventory[0], "Current Inv": self.current_inventory}
