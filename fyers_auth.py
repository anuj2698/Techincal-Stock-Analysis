#!/usr/bin/env python3
"""Fyers access token generation — supports both local (browser) and cloud (callback) flows."""
from __future__ import annotations

import hashlib
import os
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import pyotp
import requests
from dotenv import load_dotenv, set_key

load_dotenv()

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

BASE_URL = "https://api-t2.fyers.in/vagator/v2"
TOKEN_URL = "https://api-t1.fyers.in/api/v3"


_auth_code_result = {"code": None}


class _AuthCodeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        code = query.get("auth_code", [None])[0]
        if code:
            _auth_code_result["code"] = code
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<html><body style="background:#0f1117;color:#3fb950;font-family:sans-serif;
                display:flex;justify-content:center;align-items:center;height:100vh;font-size:1.5rem;">
                Token generated successfully. You can close this tab.
                </body></html>""")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"No auth_code received")

    def log_message(self, format, *args):
        pass


def _do_login():
    """Perform Fyers API login (OTP + TOTP + PIN) and return the bearer token."""
    fy_id = os.environ.get("FYERS_FY_ID")
    app_id = os.environ.get("FYERS_APP_ID")
    redirect_uri = os.environ.get("FYERS_REDIRECT_URI")
    pin = os.environ.get("FYERS_PIN")
    totp_secret = os.environ.get("FYERS_TOTP_SECRET")

    if not all([fy_id, app_id, redirect_uri, pin, totp_secret]):
        missing = [k for k, v in {
            "FYERS_FY_ID": fy_id, "FYERS_APP_ID": app_id,
            "FYERS_REDIRECT_URI": redirect_uri, "FYERS_PIN": pin, "FYERS_TOTP_SECRET": totp_secret,
        }.items() if not v]
        raise ValueError(f"Missing env vars: {', '.join(missing)}")

    print("  Step 1: Logging in (OTP + TOTP + PIN)...")
    r1 = requests.post(f"{BASE_URL}/send_login_otp", json={"fy_id": fy_id, "app_id": "2"})
    d1 = r1.json()
    if r1.status_code == 429:
        raise Exception("Fyers rate limit — too many requests. Wait 15-30 minutes and try again.")
    if d1.get("s") != "ok":
        raise Exception(f"send_login_otp failed: {d1}")

    otp_code = pyotp.TOTP(totp_secret).now()
    r2 = requests.post(f"{BASE_URL}/verify_otp", json={"request_key": d1["request_key"], "otp": otp_code})
    r2.raise_for_status()
    d2 = r2.json()
    if d2.get("s") != "ok":
        raise Exception(f"verify_otp failed: {d2}")

    r3 = requests.post(f"{BASE_URL}/verify_pin", json={
        "request_key": d2["request_key"], "identity_type": "pin", "identifier": pin,
    })
    r3.raise_for_status()
    d3 = r3.json()
    if d3.get("s") != "ok":
        raise Exception(f"verify_pin failed: {d3}")

    bearer = d3["data"]["access_token"]

    print("  Step 2: Creating auth session...")
    r4 = requests.post(f"{TOKEN_URL}/token", json={
        "fyers_id": fy_id, "app_id": app_id.split("-")[0], "redirect_uri": redirect_uri,
        "appType": "100", "code_challenge": "", "state": "None",
        "scope": "", "nonce": "", "response_type": "code", "create_cookie": True,
    }, headers={"Authorization": f"Bearer {bearer}"})

    d4 = r4.json()

    # Fyers returns 308 with Url containing auth_code directly
    auth_code_from_url = None
    url_val = d4.get("Url", "")
    if url_val:
        code = parse_qs(urlparse(url_val).query).get("auth_code", [None])[0]
        if code:
            auth_code_from_url = code
            print("  Auth code captured directly from /token response")

    if not auth_code_from_url and d4.get("s") != "ok":
        raise Exception(f"token request failed: {d4}")

    return bearer, auth_code_from_url


def get_auth_url(redirect_uri_override=None):
    """Do the login and return either (auth_code, None) if captured directly,
    or (None, auth_url) if browser redirect is needed.
    """
    bearer, auth_code = _do_login()
    if auth_code:
        return auth_code, None

    app_id = os.environ.get("FYERS_APP_ID")
    redirect_uri = redirect_uri_override or os.environ.get("FYERS_REDIRECT_URI")
    auth_url = (
        f"{TOKEN_URL}/generate-authcode"
        f"?client_id={app_id}&redirect_uri={redirect_uri}"
        f"&response_type=code&state=None"
    )
    return None, auth_url


def exchange_auth_code(auth_code: str) -> str:
    """Exchange an auth_code for an access token. Save to env."""
    app_id = os.environ.get("FYERS_APP_ID")
    secret_key = os.environ.get("FYERS_SECRET_KEY")

    print("  Exchanging auth code for access token...")
    app_id_hash = hashlib.sha256(f"{app_id}:{secret_key}".encode()).hexdigest()
    r = requests.post(f"{TOKEN_URL}/validate-authcode", json={
        "grant_type": "authorization_code", "appIdHash": app_id_hash, "code": auth_code,
    })
    r.raise_for_status()
    d = r.json()
    if d.get("s") != "ok":
        raise Exception(f"validate-authcode failed: {d}")

    access_token = d["access_token"]

    print("  Verifying token...")
    from fyers_apiv3 import fyersModel
    os.makedirs("logs", exist_ok=True)
    client = fyersModel.FyersModel(client_id=app_id, token=access_token, is_async=False, log_path="logs/")
    profile = client.get_profile()
    if profile.get("s") != "ok":
        raise Exception(f"Token verification failed: {profile}")

    name = profile["data"]["name"]
    print(f"  Authenticated as: {name}")

    os.environ["FYERS_ACCESS_TOKEN"] = access_token
    try:
        set_key(ENV_PATH, "FYERS_ACCESS_TOKEN", access_token)
        print("  Token saved to .env")
    except Exception:
        print("  Token saved to env (no .env file write on cloud)")

    return access_token


def generate_access_token() -> str:
    """Full token generation — tries headless first, falls back to browser.
    Works on both local and cloud.
    """
    bearer, auth_code = _do_login()

    if auth_code:
        print("  Token obtained headlessly (no browser needed)")
        return exchange_auth_code(auth_code)

    app_id = os.environ.get("FYERS_APP_ID")
    redirect_uri = os.environ.get("FYERS_REDIRECT_URI")

    auth_url = (
        f"{TOKEN_URL}/generate-authcode"
        f"?client_id={app_id}&redirect_uri={redirect_uri}"
        f"&response_type=code&state=None"
    )

    parsed = urlparse(redirect_uri)
    port = parsed.port or 8080
    server = HTTPServer(("127.0.0.1", port), _AuthCodeHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    webbrowser.open(auth_url)

    print("  Waiting for browser authorization...")
    for _ in range(120):
        if _auth_code_result["code"]:
            break
        time.sleep(0.5)

    server.shutdown()

    auth_code = _auth_code_result["code"]
    _auth_code_result["code"] = None

    if not auth_code:
        raise Exception("Timed out waiting for auth_code")

    return exchange_auth_code(auth_code)


def refresh_token_if_needed() -> bool:
    """Check if current token works. If not, generate new one (local only)."""
    app_id = os.environ.get("FYERS_APP_ID")
    token = os.environ.get("FYERS_ACCESS_TOKEN")

    if app_id and token:
        try:
            from fyers_apiv3 import fyersModel
            os.makedirs("logs", exist_ok=True)
            client = fyersModel.FyersModel(client_id=app_id, token=token, is_async=False, log_path="logs/")
            resp = client.get_profile()
            if resp.get("s") == "ok":
                print(f"Fyers token valid — {resp['data']['name']}")
                return False
        except Exception:
            pass

    print("Fyers token expired. Refreshing...")
    try:
        new_token = generate_access_token()
        return True
    except Exception as e:
        print(f"Token refresh failed: {e}")
        return False


if __name__ == "__main__":
    refresh_token_if_needed()
