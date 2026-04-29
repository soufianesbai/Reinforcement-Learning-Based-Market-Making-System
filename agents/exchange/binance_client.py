"""
Binance Testnet Client for direct order execution
"""

import hmac
import hashlib
import time
import requests
from typing import Dict, Optional
from ..trading_types import Side


class BinanceTestnetClient:
    """Client for Binance Testnet API"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://testnet.binance.vision"
        self.recv_window = 10000
        self.time_offset_ms = None
        self.last_time_sync_ms = 0
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key
        })
        self._sync_time()
        
    def _sign_request(self, params: Dict) -> str:
        """Generate signature for authenticated requests"""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _sync_time(self) -> None:
        """Sync local time with Binance server time"""
        try:
            response = self.session.get(f"{self.base_url}/api/v3/time", timeout=5)
            response.raise_for_status()
            server_time = response.json().get("serverTime")
            if server_time is not None:
                local_time = int(time.time() * 1000)
                self.time_offset_ms = int(server_time) - local_time
                self.last_time_sync_ms = local_time
        except Exception:
            pass

    def _get_timestamp(self) -> int:
        """Get a timestamp adjusted by server time offset"""
        local_time = int(time.time() * 1000)
        if self.time_offset_ms is None or (local_time - self.last_time_sync_ms) > 60_000:
            self._sync_time()
        offset = self.time_offset_ms or 0
        return local_time + offset

    def _signed_params(self, params: Dict) -> Dict:
        """Attach timestamp, recvWindow, and signature"""
        signed = dict(params)
        signed.setdefault('recvWindow', self.recv_window)
        signed['timestamp'] = self._get_timestamp()
        signed['signature'] = self._sign_request(signed)
        return signed

    def _send_signed_request(self, method: str, endpoint: str, params: Dict) -> Optional[Dict]:
        """Send a signed request - fail fast for fresh decision on retry"""
        signed = self._signed_params(params)
        response = self.session.request(
            method,
            f"{self.base_url}{endpoint}",
            params=signed,
            timeout=10
        )

        if response.status_code == 200:
            return response.json()

        print(f"Order failed: {response.status_code} - {response.text}")
        return None
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        try:
            endpoint = "/api/v3/account"
            return self._send_signed_request("GET", endpoint, {})
        except Exception as e:
            print(f"Error getting account info: {e}")
            return None
    
    def get_balance(self, asset: str = "USDT") -> Optional[float]:
        """Get balance for a specific asset"""
        account_info = self.get_account_info()
        if not account_info:
            return None
        
        for balance in account_info.get('balances', []):
            if balance['asset'] == asset:
                return float(balance['free'])
        return None
    
    def place_order(self, symbol: str, side: Side, quantity: float, 
                   order_type: str = "MARKET") -> Optional[Dict]:
        """Place an order on Binance testnet
        
        Args:
            symbol: Trading pair (e.g., "SOLUSDT")
            side: BUY or SELL
            quantity: Quantity to trade
            order_type: MARKET, LIMIT, etc.
        
        Returns:
            Order response or None if failed
        """
        try:
            endpoint = "/api/v3/order"
            
            params = {
                'symbol': symbol,
                'side': side.value.upper(),
                'type': order_type,
                'quantity': f"{quantity:.8f}".rstrip('0').rstrip('.'),
            }

            return self._send_signed_request("POST", endpoint, params)
                
        except Exception as e:
            print(f"Error placing order: {e}")
            return None
    
    def place_limit_order(self, symbol: str, side: Side, quantity: float, 
                         price: float) -> Optional[Dict]:
        """Place a limit order
        
        Args:
            symbol: Trading pair (e.g., "SOLUSDT")
            side: BUY or SELL
            quantity: Quantity to trade
            price: Limit price
        
        Returns:
            Order response or None if failed
        """
        try:
            endpoint = "/api/v3/order"
            
            params = {
                'symbol': symbol,
                'side': side.value.upper(),
                'type': 'LIMIT',
                'timeInForce': 'GTC',  # Good Till Cancelled
                'quantity': f"{quantity:.8f}".rstrip('0').rstrip('.'),
                'price': f"{price:.8f}".rstrip('0').rstrip('.'),
            }

            return self._send_signed_request("POST", endpoint, params)
                
        except Exception as e:
            print(f"Error placing limit order: {e}")
            return None
    
    def get_order_status(self, symbol: str, order_id: int) -> Optional[Dict]:
        """Get status of a specific order"""
        try:
            endpoint = "/api/v3/order"
            params = {
                'symbol': symbol,
                'orderId': order_id,
            }
            return self._send_signed_request("GET", endpoint, params)
        except Exception as e:
            print(f"Error getting order status: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: int) -> Optional[Dict]:
        """Cancel an open order"""
        try:
            endpoint = "/api/v3/order"
            params = {
                'symbol': symbol,
                'orderId': order_id,
            }
            return self._send_signed_request("DELETE", endpoint, params)
        except Exception as e:
            print(f"Error canceling order: {e}")
            return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> Optional[list]:
        """Get all open orders"""
        try:
            endpoint = "/api/v3/openOrders"
            params = {}
            if symbol:
                params['symbol'] = symbol

            return self._send_signed_request("GET", endpoint, params)
        except Exception as e:
            print(f"Error getting open orders: {e}")
            return None
    
    def test_connectivity(self) -> bool:
        """Test connection to Binance testnet"""
        try:
            response = self.session.get(f"{self.base_url}/api/v3/ping", timeout=5)
            return response.status_code == 200
        except:
            return False
