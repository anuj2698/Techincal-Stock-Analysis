#!/usr/bin/env python3
"""Automated Fyers access token generation using TOTP + local redirect server."""
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


def generate_access_token() -> str:
    fy_id = os.environ.get("FYERS_FY_ID")
    app_id = os.environ.get("FYERS_APP_ID")
    secret_key = os.environ.get("FYERS_SECRET_KEY")
    redirect_uri = os.environ.get("FYERS_REDIRECT_URI")
    pin = os.environ.get("FYERS_PIN")
    totp_secret = os.environ.get("FYERS_TOTP_SECRET")

    if not all([fy_id, app_id, secret_key, redirect_uri, pin, totp_secret]):
        missing = [k for k, v in {
            "FYERS_FY_ID": fy_id, "FYERS_APP_ID": app_id, "FYERS_SECRET_KEY": secret_key,
            "FYERS_REDIRECT_URI": redirect_uri, "FYERS_PIN": pin, "FYERS_TOTP_SECRET": totp_secret,
        }.items() if not v]
        raise ValueError(f"Missing in .env: {', '.join(missing)}")

    # Step 1: Login via API
    print("  Step 1/4: Logging in (OTP + TOTP + PIN)...")
    r1 = requests.post(f"{BASE_URL}/send_login_otp", json={"fy_id": fy_id, "app_id": "2"})
    r1.raise_for_status()
    d1 = r1.json()
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

    # Step 2: Get auth session + start local server + open browser
    print("  Step 2/4: Opening browser for authorization...")

    r4 = requests.post(f"{TOKEN_URL}/token", json={
        "fyers_id": fy_id, "app_id": app_id.split("-")[0], "redirect_uri": redirect_uri,
        "appType": "100", "code_challenge": "", "state": "None",
        "scope": "", "nonce": "", "response_type": "code", "create_cookie": True,
    }, headers={"Authorization": f"Bearer {bearer}"})
    r4.raise_for_status()
    d4 = r4.json()
    if d4.get("s") != "ok":
        raise Exception(f"token request failed: {d4}")

    # Start local server to catch the redirect
    parsed = urlparse(redirect_uri)
    port = parsed.port or 8080
    server = HTTPServer(("127.0.0.1", port), _AuthCodeHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Open the auth URL in the default browser
    auth_url = (
        f"{TOKEN_URL}/generate-authcode"
        f"?client_id={app_id}&redirect_uri={redirect_uri}"
        f"&response_type=code&state=None"
    )
    webbrowser.open(auth_url)

    # Wait for the auth_code to come back via redirect
    print("  Waiting for browser authorization (log in if prompted)...")
    for _ in range(120):
        if _auth_code_result["code"]:
            break
        time.sleep(0.5)

    server.shutdown()

    auth_code = _auth_code_result["code"]
    _auth_code_result["code"] = None

    if not auth_code:
        raise Exception("Timed out waiting for auth_code (60s). Did you complete the login?")

    # Step 3: Exchange auth_code for access token
    print("  Step 3/4: Exchanging auth code for access token...")
    app_id_hash = hashlib.sha256(f"{app_id}:{secret_key}".encode()).hexdigest()
    r5 = requests.post(f"{TOKEN_URL}/validate-authcode", json={
        "grant_type": "authorization_code", "appIdHash": app_id_hash, "code": auth_code,
    })
    r5.raise_for_status()
    d5 = r5.json()
    if d5.get("s") != "ok":
        raise Exception(f"validate-authcode failed: {d5}")

    access_token = d5["access_token"]

    # Step 4: Verify
    print("  Step 4/4: Verifying token...")
    from fyers_apiv3 import fyersModel
    os.makedirs("logs", exist_ok=True)
    client = fyersModel.FyersModel(client_id=app_id, token=access_token, is_async=False, log_path="logs/")
    profile = client.get_profile()
    if profile.get("s") != "ok":
        raise Exception(f"Token verification failed: {profile}")

    print(f"  Authenticated as: {profile['data']['name']}")
    return access_token


def refresh_token_if_needed() -> bool:
    """Check if current token works. If not, generate new one and update .env."""
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
        os.environ["FYERS_ACCESS_TOKEN"] = new_token
        set_key(ENV_PATH, "FYERS_ACCESS_TOKEN", new_token)
        print("Token saved to .env")
        return True
    except Exception as e:
        print(f"Token refresh failed: {e}")
        return False


if __name__ == "__main__":
    refresh_token_if_needed()
