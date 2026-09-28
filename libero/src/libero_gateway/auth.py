from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple


RUN_MANAGE_SCOPE = "runs:manage"
SESSION_OPERATE_SCOPE = "sessions:operate"
ALL_SCOPES = frozenset({RUN_MANAGE_SCOPE, SESSION_OPERATE_SCOPE})


@dataclass(frozen=True)
class AgentIdentity:
    agent_id: str
    owner_id: str
    scopes: FrozenSet[str]
    allowed_benchmarks: FrozenSet[str]
    max_sessions: int
    max_observation_level: int
    max_episode_steps: int
    allow_fixed_seed: bool
    bound_run_id: Optional[str] = None
    bound_session_id: Optional[str] = None


class TokenStore:
    def __init__(self, identities_by_hash: Dict[str, AgentIdentity]):
        self._identities_by_hash = identities_by_hash
        self._dynamic_by_hash: Dict[str, Tuple[AgentIdentity, float]] = {}
        self._dynamic_hash_by_run: Dict[str, str] = {}
        self._lock = threading.RLock()

    @classmethod
    def load(cls, path: str) -> "TokenStore":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        identities: Dict[str, AgentIdentity] = {}
        for agent_id, config in raw.get("agents", {}).items():
            digest = str(config["token_sha256"]).lower()
            if len(digest) != 64:
                raise ValueError(f"invalid token hash for {agent_id}")
            if digest in identities:
                raise ValueError("duplicate token hash in agents file")
            allowed_benchmarks = frozenset(config["allowed_benchmarks"])
            supported_benchmarks = {
                "libero_spatial",
                "libero_object",
                "libero_goal",
                "libero_10",
                "libero_90",
            }
            if not allowed_benchmarks or not allowed_benchmarks <= supported_benchmarks:
                raise ValueError(f"invalid allowed benchmark for {agent_id}")
            identities[digest] = AgentIdentity(
                agent_id=agent_id,
                owner_id=str(config.get("owner_id", agent_id)),
                scopes=frozenset(config.get("scopes", ALL_SCOPES)),
                allowed_benchmarks=allowed_benchmarks,
                max_sessions=int(config.get("max_sessions", 1)),
                max_observation_level=int(config.get("max_observation_level", 1)),
                max_episode_steps=int(config.get("max_episode_steps", 2000)),
                allow_fixed_seed=bool(config.get("allow_fixed_seed", False)),
            )
            if identities[digest].max_sessions < 1:
                raise ValueError(f"invalid session limit for {agent_id}")
            if not identities[digest].owner_id:
                raise ValueError(f"invalid owner ID for {agent_id}")
            if not identities[digest].scopes <= ALL_SCOPES:
                raise ValueError(f"invalid scopes for {agent_id}")
            if not 1 <= identities[digest].max_observation_level <= 4:
                raise ValueError(f"invalid observation level for {agent_id}")
            if identities[digest].max_episode_steps < 1:
                raise ValueError(f"invalid episode limit for {agent_id}")
        if not identities:
            raise ValueError("agents file contains no agents")
        return cls(identities)

    def authenticate(self, token: str) -> AgentIdentity:
        digest = token_sha256(token)
        # Iterate to avoid revealing token existence through dictionary timing.
        for expected, identity in self._identities_by_hash.items():
            if hmac.compare_digest(digest, expected):
                return identity
        with self._lock:
            dynamic = self._dynamic_by_hash.get(digest)
            if dynamic is not None:
                identity, expires_at = dynamic
                if time.monotonic() < expires_at:
                    return identity
                self._dynamic_by_hash.pop(digest, None)
                if identity.bound_run_id is not None:
                    self._dynamic_hash_by_run.pop(identity.bound_run_id, None)
        raise AuthenticationError("invalid bearer token")

    def issue_run_token(
        self,
        *,
        run_id: str,
        owner_id: str,
        benchmark: str,
        observation_level: int,
        episode_steps: int,
        max_sessions: int,
        ttl_seconds: int,
    ) -> str:
        """Mint a process-local credential bound to exactly one active Run.

        Issuing again for the same Run rotates the credential immediately. The
        plaintext is returned once and never persisted by the gateway.
        """
        token = "lbr_run_" + secrets.token_urlsafe(32)
        digest = token_sha256(token)
        identity = AgentIdentity(
            agent_id=f"run:{run_id}",
            owner_id=owner_id,
            scopes=frozenset({SESSION_OPERATE_SCOPE}),
            allowed_benchmarks=frozenset({benchmark}),
            max_sessions=max_sessions,
            max_observation_level=observation_level,
            max_episode_steps=episode_steps,
            allow_fixed_seed=False,
            bound_run_id=run_id,
        )
        with self._lock:
            previous = self._dynamic_hash_by_run.get(run_id)
            if previous is not None:
                self._dynamic_by_hash.pop(previous, None)
            self._dynamic_by_hash[digest] = (
                identity,
                time.monotonic() + ttl_seconds,
            )
            self._dynamic_hash_by_run[run_id] = digest
        return token

    def issue_session_token(
        self,
        *,
        run_id: str,
        session_id: str,
        owner_id: str,
        benchmark: str,
        observation_level: int,
        episode_steps: int,
        ttl_seconds: int,
    ) -> str:
        """Mint a process-local credential bound to one Run and one Session.

        The trusted allocator creates the Session before issuing this token.
        The holder can operate that Session but cannot create another one.
        """
        token = "lbr_session_" + secrets.token_urlsafe(32)
        digest = token_sha256(token)
        identity = AgentIdentity(
            agent_id=f"session:{session_id}",
            owner_id=owner_id,
            scopes=frozenset({SESSION_OPERATE_SCOPE}),
            allowed_benchmarks=frozenset({benchmark}),
            max_sessions=1,
            max_observation_level=observation_level,
            max_episode_steps=episode_steps,
            allow_fixed_seed=False,
            bound_run_id=run_id,
            bound_session_id=session_id,
        )
        with self._lock:
            previous = self._dynamic_hash_by_run.get(run_id)
            if previous is not None:
                self._dynamic_by_hash.pop(previous, None)
            self._dynamic_by_hash[digest] = (
                identity,
                time.monotonic() + ttl_seconds,
            )
            self._dynamic_hash_by_run[run_id] = digest
        return token

    def revoke_run_token(self, run_id: str) -> None:
        with self._lock:
            digest = self._dynamic_hash_by_run.pop(run_id, None)
            if digest is not None:
                self._dynamic_by_hash.pop(digest, None)


class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


def token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
