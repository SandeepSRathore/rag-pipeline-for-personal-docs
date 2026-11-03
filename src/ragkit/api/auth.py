"""API-key auth + in-memory rate limiting.

Keys map to roles (employee / hr / admin), and the role drives ACL filtering.
Keys are compared in constant time (hmac.compare_digest) to avoid timing attacks.

In production you'd use your identity provider (OIDC/JWT) and a shared rate limiter
(Redis, or the API gateway). The in-memory limiter below is per-process only.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from ragkit.config import get_settings


@dataclass(frozen=True)
class Principal:
    key_id: str  # hash prefix of the key: safe to log
    role: str


def _lookup(api_key: str) -> Principal | None:
    match: Principal | None = None
    for key, role in get_settings().api_key_roles().items():
        # Check every key (no early exit) so timing doesn't reveal which prefix matched.
        if hmac.compare_digest(key.encode(), api_key.encode()):
            match = Principal(hashlib.sha256(key.encode()).hexdigest()[:8], role)
    return match


def require_principal(x_api_key: str | None = Header(default=None)) -> Principal:
    if not x_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing X-API-Key header")
    p = _lookup(x_api_key)
    if p is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
    limiter.check(p.key_id, get_settings().rate_limit_per_minute)
    return p


def require_admin(p: Principal = Depends(require_principal)) -> Principal:
    if p.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin role required")
    return p


class SlidingWindowLimiter:
    def __init__(self, window_s: float = 60.0):
        self.window = window_s
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str, limit: int) -> None:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= limit:
                retry = int(self.window - (now - q[0])) + 1
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    "rate limit exceeded",
                    headers={"Retry-After": str(retry)},
                )
            q.append(now)

    def reset(self) -> None:
        with self.lock:
            self.hits.clear()


limiter = SlidingWindowLimiter()
