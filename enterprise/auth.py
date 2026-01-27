import os
from fastapi import Header, HTTPException, Depends
from dataclasses import dataclass


@dataclass
class UserContext:
    api_key: str
    role: str


# ✅ Keys loaded securely from Render Environment Variables
ENTERPRISE_KEYS = {
    os.getenv("BLADNIR_ADMIN_KEY"): "admin",
}


def require_auth(x_api_key: str = Header(None)) -> UserContext:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    if x_api_key not in ENTERPRISE_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    role = ENTERPRISE_KEYS[x_api_key]
    return UserContext(api_key=x_api_key, role=role)
