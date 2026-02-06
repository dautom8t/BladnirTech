"""
Enterprise authentication with secure key management, audit logging, and timing-safe comparison.

In **demo mode** (``ENVIRONMENT`` unset or set to ``demo``), authentication is
bypassed and every request receives a ``UserContext(user_id="demo", role="admin")``.
In production the original API-key flow is enforced.
"""

import hashlib
import hmac
import logging
import os
from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, Depends, Request

logger = logging.getLogger(__name__)

# Import central settings
from config import settings as _settings


@dataclass
class UserContext:
    """User context with role and identifier (NOT the raw API key)."""
    user_id: str  # Identifier for the key (e.g., "admin", "service_account_1")
    role: str
    key_id: str  # Last 8 chars of key hash for logging (safe to log)


# Shared demo context returned for every request when auth is disabled.
_DEMO_USER = UserContext(user_id="demo", role="admin", key_id="demo0000")


class AuthenticationError(Exception):
    """Raised when authentication setup is invalid."""
    pass


# =====================================
# Secure Key Loading with Validation
# =====================================

def _load_api_keys() -> dict[str, tuple[str, str]]:
    """
    Load and validate API keys from environment.

    Returns:
        dict mapping hashed keys to (user_id, role) tuples

    Raises:
        AuthenticationError: If required keys are missing or invalid
    """
    keys = {}

    # Load admin key (required)
    admin_key = os.getenv("BLADNIR_ADMIN_KEY")
    if not admin_key:
        raise AuthenticationError(
            "CRITICAL: BLADNIR_ADMIN_KEY environment variable not set. "
            "Application cannot start without authentication configured."
        )

    if len(admin_key) < 32:
        raise AuthenticationError(
            "CRITICAL: BLADNIR_ADMIN_KEY must be at least 32 characters. "
            f"Current length: {len(admin_key)}"
        )

    # Hash the key for timing-safe comparison
    admin_key_hash = hashlib.sha256(admin_key.encode()).hexdigest()
    keys[admin_key_hash] = ("admin", "admin")
    logger.info(f"Loaded admin key (hash: {admin_key_hash[:8]}...)")

    # Load additional service account keys (optional)
    # Format: BLADNIR_KEY_<NAME>=<key>:<role>
    # Example: BLADNIR_KEY_MONITOR=abc123def456:read_only
    for env_var, value in os.environ.items():
        if env_var.startswith("BLADNIR_KEY_"):
            try:
                name = env_var.replace("BLADNIR_KEY_", "").lower()

                if ":" not in value:
                    logger.warning(f"Skipping {env_var}: missing role (format: key:role)")
                    continue

                key, role = value.split(":", 1)

                if len(key) < 32:
                    logger.warning(f"Skipping {env_var}: key too short (min 32 chars)")
                    continue

                key_hash = hashlib.sha256(key.encode()).hexdigest()
                keys[key_hash] = (name, role)
                logger.info(f"Loaded service account '{name}' with role '{role}' (hash: {key_hash[:8]}...)")

            except Exception as e:
                logger.error(f"Failed to load {env_var}: {e}")
                continue

    if not keys:
        raise AuthenticationError("No valid API keys loaded")

    logger.info(f"Authentication initialized with {len(keys)} valid key(s)")
    return keys


# Load keys at module import.
# In demo/development mode, missing keys are not fatal — we fall back to a
# built-in demo key so the app can start without any environment variables.
if _settings.auth_enabled:
    try:
        API_KEYS = _load_api_keys()
    except AuthenticationError as e:
        logger.critical(str(e))
        raise
else:
    # Demo / development: provide a built-in key so the app boots cleanly.
    _demo_hash = hashlib.sha256(_settings.demo_admin_key.encode()).hexdigest()
    API_KEYS = {_demo_hash: ("demo", "admin")}
    logger.info("Auth running in DEMO mode — authentication is bypassed for all requests")


# =====================================
# Timing-Safe Authentication
# =====================================

def _verify_api_key(provided_key: str) -> Optional[tuple[str, str, str]]:
    """
    Verify API key using timing-safe comparison.
    
    Args:
        provided_key: The API key to verify
        
    Returns:
        Tuple of (user_id, role, key_id) if valid, None otherwise
    """
    if not provided_key or len(provided_key) < 32:
        return None
    
    # Hash the provided key
    provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
    
    # Timing-safe comparison against all valid keys
    for stored_hash, (user_id, role) in API_KEYS.items():
        if hmac.compare_digest(provided_hash, stored_hash):
            key_id = provided_hash[:8]
            return (user_id, role, key_id)
    
    return None


def require_auth(
    request: Request,
    x_api_key: str = Header(None, description="API key for authentication")
) -> UserContext:
    """
    Authentication dependency for FastAPI endpoints.

    In demo mode (``auth_enabled=False``) every request is treated as an
    authenticated admin — no header required.

    Usage:
        @app.get("/protected")
        def protected_endpoint(user: UserContext = Depends(require_auth)):
            return {"message": f"Hello {user.user_id}"}

    Args:
        request: FastAPI request object (for IP logging)
        x_api_key: API key from X-API-Key header

    Returns:
        UserContext with user details

    Raises:
        HTTPException: 401 if authentication fails (production only)
    """
    # --- demo bypass ---
    if not _settings.auth_enabled:
        return _DEMO_USER

    client_ip = request.client.host if request.client else "unknown"

    # Check if key provided
    if not x_api_key:
        logger.warning(f"Authentication failed: Missing API key (IP: {client_ip})")
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Verify key (timing-safe)
    result = _verify_api_key(x_api_key)

    if not result:
        logger.warning(
            f"Authentication failed: Invalid API key "
            f"(IP: {client_ip}, key_prefix: {x_api_key[:8]}...)"
        )
        # Use same generic message to avoid leaking info
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    user_id, role, key_id = result

    # Log successful authentication (safe to log key_id, it's just a hash prefix)
    logger.info(
        f"Authentication successful: user={user_id}, role={role}, "
        f"key_id={key_id}, IP={client_ip}"
    )

    return UserContext(user_id=user_id, role=role, key_id=key_id)


def require_role(required_role: str):
    """
    Role-based access control dependency.
    
    Usage:
        @app.post("/admin/action")
        def admin_action(user: UserContext = Depends(require_role("admin"))):
            return {"message": "Admin action performed"}
    
    Args:
        required_role: Role required to access the endpoint
        
    Returns:
        Dependency function that checks role
    """
    def role_checker(user: UserContext = Depends(require_auth)) -> UserContext:
        if user.role != required_role:
            logger.warning(
                f"Authorization failed: user={user.user_id}, "
                f"required_role={required_role}, actual_role={user.role}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        return user
    
    return role_checker


# =====================================
# Optional: Rate Limiting
# =====================================

from collections import defaultdict
from datetime import datetime, timedelta
from threading import Lock

class RateLimiter:
    """Simple in-memory rate limiter for auth attempts."""
    
    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window = timedelta(seconds=window_seconds)
        self.attempts: dict[str, list[datetime]] = defaultdict(list)
        self.lock = Lock()
    
    def check_rate_limit(self, identifier: str) -> bool:
        """
        Check if identifier has exceeded rate limit.
        
        Args:
            identifier: IP address or other identifier
            
        Returns:
            True if under limit, False if exceeded
        """
        now = datetime.now()
        
        with self.lock:
            # Clean old attempts
            self.attempts[identifier] = [
                ts for ts in self.attempts[identifier]
                if now - ts < self.window
            ]
            
            # Check limit
            if len(self.attempts[identifier]) >= self.max_attempts:
                return False
            
            # Record attempt
            self.attempts[identifier].append(now)
            return True


# Global rate limiter instance
_rate_limiter = RateLimiter(max_attempts=10, window_seconds=300)  # 10 attempts per 5 min


def require_auth_with_rate_limit(
    request: Request,
    x_api_key: str = Header(None)
) -> UserContext:
    """Authentication with rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not _rate_limiter.check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Too many authentication attempts. Please try again later.",
            headers={"Retry-After": "300"},
        )
    
    # Proceed with normal auth
    return require_auth(request, x_api_key)
