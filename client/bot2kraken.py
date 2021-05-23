import requests
import time
import urllib.parse
import hashlib
import hmac
import base64

# Authentication signature
def get_kraken_signature(urlpath, data, secret):
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest())
    return sigdigest.decode()

# Attaches auth headers and returns results of a POST request
def kraken_request(uri_path, data, api_key, api_sec):
    headers = {}
    headers['API-Key'] = api_key
    headers['API-Sign'] = get_kraken_signature(uri_path, data, api_sec)             
    req = requests.post(("https://api.kraken.com" + uri_path), headers=headers, data=data)
    return req


class KrakenClient:
  
    def __init__(self, api_key=None, api_sec=None):
        self.api_key = api_key
        self.api_sec = api_sec
        self.public = {
            'time': '/0/public/Time',
            'status': '/0/public/SystemStatus',
            'assets': '/0/public/Assets',
            'pair_info': '/0/public/AssetPairs?pair=XXBTZUSD',
            'ticker_info': '/0/public/Ticker?pair=XBTUSD',
            'ohlc': '/0/public/OHLC?pair=XBTUSD',
            'order_book': '/0/public/Depth?pair=XBTUSD',
            'trades': '/0/public/Trades?pair=XBTUSD',
            'spread': '/0/public/Spread?pair=XBTUSD',
        }
        self.private = {
            'balance': '/0/private/Balance',
            'trade_balance': '/0/private/TradeBalance',
            'open_orders': '/0/private/OpenOrders',
            'closed_orders': '/0/private/ClosedOrders',
            'history': '/0/private/TradesHistory',
            'open_positions': '/0/private/OpenPositions',
            'ledger_info': '/0/private/Ledgers',
            'trade_volume': '/0/private/TradeVolume',
        }

    def public_query(self, query):
        resp = requests.get("https://api.kraken.com/"+self.public[query])
        return resp.json()

    def private_query(self, query):
        resp = kraken_request(self.private[query], {
            "nonce": str(int(1000*time.time()))
        }, self.api_key, self.api_sec)
        return resp.json()

    def add_order(self, type, volume, validate=True):
        nonce = str(int(1000*time.time()))
        resp = kraken_request('/0/private/AddOrder', {
            "nonce": nonce,
            "ordertype": "market",
            "type": type,
            "volume": volume,
            "pair": "XBTUSD",
            "validate": validate
        }, self.api_key, self.api_sec)
        return resp.json()
