#!/usr/bin/env python3
"""
Script de test pour vérifier l'exécution d'ordres sur Binance testnet
"""

from ..exchange.binance_client import BinanceTestnetClient
from ..exchange.config import BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_API_SECRET
from ..trading_types import Side

def main():
    print("🧪 Test de connexion Binance Testnet")
    print("=" * 60)
    
    # Créer le client
    client = BinanceTestnetClient(
        api_key=BINANCE_TESTNET_API_KEY,
        api_secret=BINANCE_TESTNET_API_SECRET
    )
    
    # 1. Test de connectivité
    print("\n1️⃣ Test de connectivité...")
    if client.test_connectivity():
        print("   ✅ Connexion au testnet réussie")
    else:
        print("   ❌ Échec de connexion")
        return
    
    # 2. Récupérer les infos du compte
    print("\n2️⃣ Informations du compte...")
    account_info = client.get_account_info()
    if account_info:
        print(f"   ✅ Compte actif")
        print(f"   Can Trade: {account_info.get('canTrade', False)}")
        print(f"   Can Withdraw: {account_info.get('canWithdraw', False)}")
    else:
        print("   ❌ Impossible de récupérer les infos du compte")
        return
    
    # 3. Afficher les soldes
    print("\n3️⃣ Soldes principaux...")
    for balance in account_info.get('balances', [])[:10]:
        free = float(balance['free'])
        locked = float(balance['locked'])
        if free > 0 or locked > 0:
            print(f"   💰 {balance['asset']}: {free:.8f} (libre) + {locked:.8f} (bloqué)")
    
    # 4. Vérifier le solde USDT
    print("\n4️⃣ Solde USDT...")
    usdt_balance = client.get_balance("USDT")
    if usdt_balance is not None:
        print(f"   💵 USDT disponible: {usdt_balance:.2f}")
        
        if usdt_balance < 10:
            print("\n   ⚠️  Solde insuffisant pour tester un ordre")
            print("   💡 Rechargez votre compte testnet sur: https://testnet.binance.vision/")
            return
    else:
        print("   ❌ Impossible de récupérer le solde USDT")
        return
    
    # 5. Vérifier le solde SOL
    print("\n5️⃣ Solde SOL...")
    sol_balance = client.get_balance("SOL")
    if sol_balance is not None:
        print(f"   🪙 SOL disponible: {sol_balance:.8f}")
    
    # 6. Vérifier les ordres ouverts
    print("\n6️⃣ Ordres ouverts...")
    open_orders = client.get_open_orders("SOLUSDT")
    if open_orders is not None:
        print(f"   📋 Nombre d'ordres ouverts: {len(open_orders)}")
        if open_orders:
            for order in open_orders[:5]:
                print(f"      - {order['side']} {order['origQty']} @ {order['price']}")
    
    # 7. Proposer de placer un ordre de test
    print("\n7️⃣ Test d'ordre...")
    print("   ⚠️  Voulez-vous placer un petit ordre de TEST sur le testnet?")
    print("   Cet ordre sera exécuté avec de l'argent fictif (testnet).")
    print("   Ordre proposé: BUY 0.1 SOL (MARKET)")
    
    response = input("\n   Continuer? (oui/non): ").lower().strip()
    
    if response in ['oui', 'yes', 'y', 'o']:
        print("\n   📤 Placement de l'ordre de test...")
        order_response = client.place_order(
            symbol="SOLUSDT",
            side=Side.BUY,
            quantity=0.1,
            order_type="MARKET"
        )
        
        if order_response:
            print("\n   ✅ ORDRE EXÉCUTÉ AVEC SUCCÈS!")
            print(f"   Order ID: {order_response.get('orderId')}")
            print(f"   Status: {order_response.get('status')}")
            print(f"   Side: {order_response.get('side')}")
            print(f"   Executed Qty: {order_response.get('executedQty')}")
            print(f"   Cumulative Quote Qty: {order_response.get('cummulativeQuoteQty')}")
            
            if order_response.get('fills'):
                print("\n   📊 Détails des exécutions:")
                for fill in order_response['fills']:
                    print(f"      - Prix: ${fill['price']} | Qty: {fill['qty']} | Commission: {fill['commission']} {fill['commissionAsset']}")
            
            print("\n   🎉 L'ordre a été exécuté sur le TESTNET Binance!")
            print("   💡 Vérifiez votre historique sur: https://testnet.binance.vision/")
        else:
            print("\n   ❌ Échec de l'ordre")
    else:
        print("\n   ℹ️  Test annulé")
    
    print("\n" + "=" * 60)
    print("✅ Test terminé!")
    print("\nPour vérifier vos ordres en ligne:")
    print("   🌐 https://testnet.binance.vision/")
    print("   → Login → Orders → Order History")

if __name__ == "__main__":
    main()
