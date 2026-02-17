"""
Script de test unitaire manuel pour la stratégie SMA.
Ce script simule des données de marché pour vérifier que la stratégie réagit correctement.
"""
import sys
import os

# Ajoute le dossier courant au path pour pouvoir importer le package 'agents'
sys.path.append(os.getcwd())

from agents.strategies.sma import SMAStrategy
from agents.types import Side

def test_sma_reaction():
    print("🧪 DÉBUT DU TEST : Stratégie SMA isolée")
    print("=" * 50)
    
    # 1. Instancier la stratégie
    # On met un seuil de 1% (0.01)
    strategy = SMAStrategy("SOLUSDT", quantity_per_trade=10, threshold=0.01)
    print(f"🔹 Stratégie initialisée (Seuil: {strategy.threshold * 100}%)")
    
    # 2. Phase d'initialisation (remplissage du buffer SMA)
    # On envoie 3 prix stables à 100$
    # Note : Le format des prix dans le JSON est souvent en entiers (cents) ou flottants selon votre implémentation C++.
    # Votre classe MarketData divise souvent par 100, mais ici on simule des données brutes "propres" déjà converties ou non selon votre MarketData.py.
    # Supposons que MarketData.py attend des entiers (cents) et divise par 100, ou reçoit des floats. 
    # Regardons votre MarketData.py original : il fait `state_dict.get("mid_price", 0) / 100.0`.
    # Donc on doit envoyer des prix en CENTS (x100).
    
    stable_price_cents = 10000  # 100.00 $
    stable_market = {
        "mid_price": stable_price_cents,
        "bid_price": stable_price_cents - 10,
        "ask_price": stable_price_cents + 10
    }
    
    print("\n1️⃣  Injection de données stables (100$)...")
    for i in range(3):
        strategy.update_data(stable_market)
        print(f"   -> Update {i+1}/3")
    
    # Vérification de l'état interne
    status = strategy.get_status()
    print(f"   📊 État actuel : SMA = {status.get('sma')}")
    
    if status.get('sma') != 100.0:
        print(f"   ❌ ERREUR : La SMA devrait être 100.0, obtenu : {status.get('sma')}")
        return

    # 3. Simulation d'un Crash (Opportunité d'ACHAT)
    # Le prix chute à 90$ (9000 cents).
    # Déviation = (90 - 100) / 100 = -10%.
    # -10% est bien inférieur à -1% (seuil), donc on doit ACHETER.
    
    crash_price_cents = 9000 # 90.00 $
    crash_market = {
        "mid_price": crash_price_cents,
        "bid_price": crash_price_cents - 10,
        "ask_price": crash_price_cents + 10
    }
    
    print("\n2️⃣  Simulation Crash (100$ -> 90$)...")
    strategy.update_data(crash_market)
    
    # 4. Calcul du signal
    print("   🤔 Calcul de la décision...")
    signal = strategy.compute_signal()
    
    print("-" * 20)
    print(f"   RÉSULTAT : {signal.side}")
    print(f"   RAISON   : {signal.reason}")
    print(f"   MÉTADATA : {signal.metadata}")
    print("-" * 20)
    
    # 5. Validation finale
    if signal.side == Side.BUY:
        print("\n✅ SUCCÈS : L'agent a correctement décidé d'ACHETER après le crash.")
    elif signal.side == Side.SELL:
        print("\n❌ ÉCHEC : L'agent veut VENDRE alors que le prix est bas.")
    else:
        print("\n❌ ÉCHEC : L'agent a décidé de NE RIEN FAIRE (HOLD).")
        print("   Vérifiez votre seuil ou le calcul de la déviation.")

if __name__ == "__main__":
    test_sma_reaction()
