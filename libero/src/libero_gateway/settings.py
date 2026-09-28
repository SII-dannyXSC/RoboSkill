from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


def _csv_ints(value: str) -> Tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return values or (0,)


@dataclass(frozen=True)
class Settings:
    backend: str = "mock"
    host: str = "127.0.0.1"
    port: int = 8080
    max_sessions_per_gpu: int = 2
    process_start_timeout_seconds: float = 180.0
    request_timeout_seconds: float = 30.0
    session_idle_seconds: int = 21600
    # Effectively unbounded for interactive agents. This remains finite as a
    # last-resort guard against a broken client issuing actions forever.
    default_episode_steps: int = 1_000_000
    max_episode_steps: int = 1_000_000
    image_height: int = 128
    image_width: int = 128
    jpeg_quality: int = 90
    gpu_ids: Tuple[int, ...] = (0,)
    fixed_benchmark: str = ""
    fixed_task_id: int = -1
    agents_file: str = "config/agents.json"
    audit_log: str = "/var/lib/libero-gateway/audit.jsonl"
    evaluations_db: str = "/var/lib/libero-gateway/evaluations.sqlite3"
    run_token_ttl_seconds: int = 86400
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            backend=os.getenv("LIBERO_GATEWAY_BACKEND", "mock").lower(),
            host=os.getenv("LIBERO_GATEWAY_HOST", "127.0.0.1"),
            port=int(os.getenv("LIBERO_GATEWAY_PORT", "8080")),
            max_sessions_per_gpu=int(
                os.getenv("LIBERO_GATEWAY_MAX_SESSIONS_PER_GPU", "2")
            ),
            process_start_timeout_seconds=float(
                os.getenv("LIBERO_GATEWAY_PROCESS_START_TIMEOUT", "180")
            ),
            request_timeout_seconds=float(
                os.getenv("LIBERO_GATEWAY_REQUEST_TIMEOUT", "30")
            ),
            session_idle_seconds=int(
                os.getenv("LIBERO_GATEWAY_SESSION_IDLE_SECONDS", "21600")
            ),
            default_episode_steps=int(
                os.getenv("LIBERO_GATEWAY_DEFAULT_EPISODE_STEPS", "1000000")
            ),
            max_episode_steps=int(
                os.getenv("LIBERO_GATEWAY_MAX_EPISODE_STEPS", "1000000")
            ),
            image_height=int(os.getenv("LIBERO_GATEWAY_IMAGE_HEIGHT", "128")),
            image_width=int(os.getenv("LIBERO_GATEWAY_IMAGE_WIDTH", "128")),
            jpeg_quality=int(os.getenv("LIBERO_GATEWAY_JPEG_QUALITY", "90")),
            gpu_ids=_csv_ints(os.getenv("LIBERO_GATEWAY_GPU_IDS", "0")),
            fixed_benchmark=os.getenv("LIBERO_GATEWAY_FIXED_BENCHMARK", "").lower(),
            fixed_task_id=int(os.getenv("LIBERO_GATEWAY_FIXED_TASK_ID", "-1")),
            agents_file=os.getenv(
                "LIBERO_GATEWAY_AGENTS_FILE", "config/agents.json"
            ),
            audit_log=os.getenv(
                "LIBERO_GATEWAY_AUDIT_LOG",
                "/var/lib/libero-gateway/audit.jsonl",
            ),
            evaluations_db=os.getenv(
                "LIBERO_GATEWAY_EVALUATIONS_DB",
                "/var/lib/libero-gateway/evaluations.sqlite3",
            ),
            run_token_ttl_seconds=int(
                os.getenv("LIBERO_GATEWAY_RUN_TOKEN_TTL_SECONDS", "86400")
            ),
            log_level=os.getenv("LIBERO_GATEWAY_LOG_LEVEL", "info"),
        )

    def validate(self) -> None:
        if self.backend not in {"mock", "libero"}:
            raise ValueError("LIBERO_GATEWAY_BACKEND must be mock or libero")
        if self.max_sessions_per_gpu < 1 or not self.gpu_ids:
            raise ValueError("GPU capacity settings must be positive")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("LIBERO_GATEWAY_GPU_IDS must list each GPU once")
        if self.process_start_timeout_seconds <= 0 or self.request_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if self.run_token_ttl_seconds < 60:
            raise ValueError("Run token TTL must be at least 60 seconds")
        if not 1 <= self.default_episode_steps <= self.max_episode_steps:
            raise ValueError("default episode steps must be within the server limit")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")
        fixed_enabled = bool(self.fixed_benchmark)
        if fixed_enabled != (self.fixed_task_id >= 0):
            raise ValueError("fixed benchmark and fixed task ID must be configured together")
        task_counts = {
            "libero_spatial": 10,
            "libero_object": 10,
            "libero_goal": 10,
            "libero_10": 10,
            "libero_90": 90,
        }
        if fixed_enabled and self.fixed_benchmark not in task_counts:
            raise ValueError("unsupported fixed benchmark")
        if fixed_enabled and self.fixed_task_id >= task_counts[self.fixed_benchmark]:
            raise ValueError("fixed task ID is outside the benchmark")
        if not Path(self.agents_file).is_file():
            raise ValueError(f"agents file not found: {self.agents_file}")

    @property
    def capacity(self) -> int:
        """Maximum concurrent sessions; processes are still created on demand."""
        return len(self.gpu_ids) * self.max_sessions_per_gpu
