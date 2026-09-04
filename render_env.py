"""Helper to persist environment variables on Render via their API."""
from __future__ import annotations

import os
import requests


def update_render_env_var(key: str, value: str) -> bool:
    """Update a single environment variable on the Render service.

    Requires RENDER_API_KEY and RENDER_SERVICE_ID in the environment.
    Returns True on success, False on failure (logged but not raised).
    """
    api_key = os.environ.get("RENDER_API_KEY")
    service_id = os.environ.get("RENDER_SERVICE_ID")

    if not api_key or not service_id:
        return False

    try:
        resp = requests.put(
            f"https://api.render.com/v1/services/{service_id}/env-vars/{key}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"value": value},
            timeout=10,
        )
        if resp.status_code == 200:
            return True

        if resp.status_code == 404:
            resp2 = requests.post(
                f"https://api.render.com/v1/services/{service_id}/env-vars",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=[{"key": key, "value": value}],
                timeout=10,
            )
            return resp2.status_code in (200, 201)

        return False
    except Exception:
        return False
