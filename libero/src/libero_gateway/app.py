from __future__ import annotations

import logging
import hashlib
import ipaddress
import json
import secrets
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .auth import (
    AgentIdentity,
    AuthenticationError,
    AuthorizationError,
    RUN_MANAGE_SCOPE,
    SESSION_OPERATE_SCOPE,
    TokenStore,
)
from .backend import BENCHMARK_TASK_COUNTS, task_catalog
from .observations import BASE_PROPRIO_KEYS, EXTENDED_PROPRIO_KEYS
from .schemas import (
    ActionSpec,
    ActionSpecV1,
    AllocateExperimentRequest,
    AllocateExperimentResponse,
    CapabilitiesResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    CreateSessionV1Response,
    CreateSessionV2Request,
    EvaluationRunResponse,
    HarnessTaskProgressResponse,
    ObservationSpec,
    ObservationSpecV1,
    ResetResponse,
    ResultResponse,
    SessionConfig,
    SessionTiming,
    StartRunRequest,
    StepRequest,
    StepResponse,
    TaskCatalogResponse,
)
from .settings import Settings
from .worker_pool import CapacityError, GatewayError, QuotaError, SimulationManager

LOGGER = logging.getLogger(__name__)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()
    token_store = TokenStore.load(settings.agents_file)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager = SimulationManager(settings)
        yield
        app.state.manager.shutdown()

    app = FastAPI(
        title="LIBERO Agent Gateway",
        version="2.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    def identity_from_header(
        authorization: Optional[str] = Header(default=None),
    ) -> AgentIdentity:
        if not authorization or not authorization.startswith("Bearer "):
            raise AuthenticationError()
        return token_store.authenticate(authorization[7:])

    @app.exception_handler(AuthenticationError)
    async def authentication_error(_request: Request, _exc: AuthenticationError):
        return _error_response("UNAUTHENTICATED", 401)

    @app.exception_handler(AuthorizationError)
    async def authorization_error(_request: Request, _exc: AuthorizationError):
        return _error_response("INSUFFICIENT_SCOPE", 403)

    def require_scope(identity: AgentIdentity, scope: str) -> None:
        if scope not in identity.scopes:
            raise AuthorizationError()

    def require_bound_session(identity: AgentIdentity, session_id: str) -> None:
        if (
            identity.bound_session_id is not None
            and identity.bound_session_id != session_id
        ):
            raise AuthorizationError()

    def require_loopback(request: Request) -> None:
        host = request.client.host if request.client is not None else ""
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = host == "localhost"
        if not is_loopback:
            raise AuthorizationError()

    @app.exception_handler(GatewayError)
    async def gateway_error(_request: Request, exc: GatewayError):
        headers = {"Retry-After": "5"} if exc.http_status == 503 else None
        return _error_response(exc.code, exc.http_status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        LOGGER.warning("request validation failed: %s", exc.errors())
        return _error_response("INVALID_REQUEST", 422)

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, exc: Exception):
        request_id = "req_" + secrets.token_urlsafe(12)
        LOGGER.exception("unexpected request failure request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL_ERROR", "request_id": request_id}},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/healthz")
    def health(request: Request):
        return request.app.state.manager.health()

    @app.get("/v2/capabilities", response_model=CapabilitiesResponse)
    def capabilities(
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        maximum = min(settings.max_episode_steps, identity.max_episode_steps)
        return {
            "worker_mode": "per_session",
            "credential_type": (
                "session"
                if identity.bound_session_id is not None
                else "run"
                if identity.bound_run_id is not None
                else "static"
            ),
            "scopes": sorted(identity.scopes),
            "benchmarks": {
                name: {"task_count": count}
                for name, count in BENCHMARK_TASK_COUNTS.items()
                if name in identity.allowed_benchmarks
            },
            "observation_levels": {
                "minimum": 1,
                "maximum": identity.max_observation_level,
                "profiles": {
                    "1": "rgb_and_basic_proprioception",
                    "2": "level_1_plus_task_object_bboxes",
                    "3": "level_2_plus_extended_proprioception",
                    "4": "level_3_plus_metric_depth_and_camera_calibration",
                },
            },
            "episode_length": {
                "default": min(settings.default_episode_steps, maximum),
                "maximum": maximum,
            },
            "max_concurrent_sessions": identity.max_sessions,
            "fixed_seed_allowed": identity.allow_fixed_seed,
            "timing_metrics": [
                "session_startup_seconds",
                "session_time_to_success_seconds",
                "run_time_to_first_success_seconds",
            ],
        }

    @app.get("/v2/tasks/{benchmark}", response_model=TaskCatalogResponse)
    def tasks(
        benchmark: str,
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        if benchmark not in BENCHMARK_TASK_COUNTS:
            return _error_response("TASK_NOT_FOUND", 404)
        if benchmark not in identity.allowed_benchmarks:
            return _error_response("TASK_FORBIDDEN", 403)
        return {
            "benchmark": benchmark,
            "tasks": list(task_catalog(settings.backend, benchmark)),
        }

    @app.post(
        "/v2/runs",
        response_model=EvaluationRunResponse,
        response_model_exclude_none=True,
        status_code=201,
    )
    def start_run_v2(
        body: StartRunRequest,
        request: Request,
        response: Response,
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=8, max_length=128
        ),
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        require_scope(identity, RUN_MANAGE_SCOPE)
        benchmark = body.task.benchmark
        if benchmark not in identity.allowed_benchmarks:
            return _error_response("TASK_FORBIDDEN", 403)
        if (
            body.task.task_id is not None
            and body.task.task_id >= BENCHMARK_TASK_COUNTS[benchmark]
        ):
            return _error_response("TASK_NOT_FOUND", 422)
        if body.observation.level > identity.max_observation_level:
            return _error_response("OBSERVATION_LEVEL_FORBIDDEN", 403)
        maximum = min(settings.max_episode_steps, identity.max_episode_steps)
        episode_length = (
            min(settings.default_episode_steps, maximum)
            if body.episode_length is None
            else body.episode_length
        )
        if not 1 <= episode_length <= maximum:
            return _error_response("MAX_STEPS_OUT_OF_RANGE", 422)

        # Fingerprint the caller's request before resolving a random task so an
        # idempotent retry returns the originally selected task.
        fingerprint = hashlib.sha256(
            json.dumps(
                body.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        task_id = (
            secrets.randbelow(BENCHMARK_TASK_COUNTS[benchmark])
            if body.task.task_id is None
            else body.task.task_id
        )
        # The model/Level pair is evaluation metadata, not an authorization
        # principal. Every new Run gets an opaque visitor identity so any two
        # trials can execute concurrently, even with identical configurations.
        agent_owner = "visitor_" + secrets.token_urlsafe(18)
        run = request.app.state.manager.begin_run(
            agent_owner,
            launcher_owner=identity.owner_id,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint,
            label=body.label,
            config={
                "benchmark": benchmark,
                "task_id": task_id,
                "episode_length": episode_length,
                "observation_level": body.observation.level,
                "bbox_scope": body.observation.bbox_scope,
            },
            max_attempts=body.max_attempts,
        )
        run["agent_token"] = token_store.issue_run_token(
            run_id=run["run_id"],
            owner_id=run["agent_identity"],
            benchmark=run["config"]["benchmark"],
            observation_level=run["config"]["observation_level"],
            episode_steps=run["config"]["episode_length"],
            max_sessions=identity.max_sessions,
            ttl_seconds=settings.run_token_ttl_seconds,
        )
        response.headers["Cache-Control"] = "no-store"
        return run

    @app.get(
        "/v2/runs/{run_id}",
        response_model=EvaluationRunResponse,
        response_model_exclude_none=True,
    )
    def run_status_v2(
        run_id: str,
        request: Request,
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        if identity.bound_session_id is not None:
            if identity.bound_run_id != run_id:
                raise AuthorizationError()
        else:
            require_scope(identity, RUN_MANAGE_SCOPE)
        return request.app.state.manager.run_status(run_id, identity.owner_id)

    @app.post(
        "/v2/runs/{run_id}/finish",
        response_model=EvaluationRunResponse,
        response_model_exclude_none=True,
    )
    def finish_run_v2(
        run_id: str,
        request: Request,
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        if identity.bound_session_id is not None:
            if identity.bound_run_id != run_id:
                raise AuthorizationError()
        else:
            require_scope(identity, RUN_MANAGE_SCOPE)
        result = request.app.state.manager.finish_run(run_id, identity.owner_id)
        token_store.revoke_run_token(run_id)
        return result

    def build_create_response_v2(
        created: dict, *, expose_seed: bool = False
    ) -> CreateSessionResponse:
        level = int(created["observation_level"])
        bbox_scope = created["bbox_scope"] if level >= 2 else None
        return CreateSessionResponse(
            session_id=created["session_id"],
            state=created["state"],
            instruction=created.get("instruction"),
            error_code=created.get("error_code"),
            action_spec=ActionSpec(),
            observation_spec=ObservationSpec(
                level=level,
                cameras={
                    "agentview_rgb": [settings.image_height, settings.image_width, 3],
                    "wrist_rgb": [settings.image_height, settings.image_width, 3],
                },
                required_proprioception_keys=list(BASE_PROPRIO_KEYS),
                optional_proprioception_keys=(
                    list(EXTENDED_PROPRIO_KEYS) if level >= 3 else []
                ),
                bbox_scope=bbox_scope,
                depth_encoding=(
                    "float32-zlib-base64" if level >= 4 else None
                ),
                camera_calibration=level >= 4,
            ),
            session_config=SessionConfig(
                run_id=created["run_id"],
                benchmark=created["benchmark"],
                task_id=created["task_id"],
                episode_length=created["episode_length"],
                observation_level=level,
                bbox_scope=bbox_scope,
                seed=created["seed"] if expose_seed else None,
            ),
            timing=SessionTiming(**created["timing"]),
        )

    @app.post(
        "/v2/experiments",
        response_model=AllocateExperimentResponse,
        response_model_exclude_none=True,
        status_code=202,
    )
    def allocate_experiment_v2(
        body: AllocateExperimentRequest,
        request: Request,
        response: Response,
        idempotency_key: str = Header(
            alias="Idempotency-Key", min_length=8, max_length=128
        ),
    ):
        """Create one Run and its only Session from a trusted local caller.

        This endpoint intentionally has no API token. It accepts requests only
        from the gateway host's loopback interface; remote clients reach it via
        the authenticated SSH tunnel. The returned credential is bound to the
        newly-created Session and cannot create another Session.
        """
        require_loopback(request)
        benchmark = body.task.benchmark
        if (
            body.task.task_id is not None
            and body.task.task_id >= BENCHMARK_TASK_COUNTS[benchmark]
        ):
            return _error_response("TASK_NOT_FOUND", 422)
        if body.observation.level > 4:
            return _error_response("OBSERVATION_LEVEL_FORBIDDEN", 403)
        episode_length = (
            settings.default_episode_steps
            if body.episode_length is None
            else body.episode_length
        )
        if not 1 <= episode_length <= settings.max_episode_steps:
            return _error_response("MAX_STEPS_OUT_OF_RANGE", 422)

        canonical_request = body.model_dump(mode="json")
        request_fingerprint = hashlib.sha256(
            json.dumps(
                canonical_request, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        # The idempotency key is opaque evaluation metadata, not a persistent
        # model/Level/seed identity. It merely makes a lost response retry-safe.
        owner_id = "direct_" + hashlib.sha256(
            idempotency_key.encode("utf-8")
        ).hexdigest()[:32]
        task_id = (
            secrets.randbelow(BENCHMARK_TASK_COUNTS[benchmark])
            if body.task.task_id is None
            else body.task.task_id
        )
        run = request.app.state.manager.begin_run(
            owner_id,
            launcher_owner=owner_id,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            label=body.label,
            config={
                "benchmark": benchmark,
                "task_id": task_id,
                "episode_length": episode_length,
                "observation_level": body.observation.level,
                "bbox_scope": body.observation.bbox_scope,
            },
            max_attempts=1,
        )
        # An idempotent retry must use the task originally selected for the Run.
        task_id = int(run["config"]["task_id"])
        session_request = {
            "run_id": run["run_id"],
            "task": {"benchmark": benchmark, "task_id": task_id},
            "episode_length": episode_length,
            "observation": body.observation.model_dump(mode="json"),
            "seed": body.seed,
        }
        session_fingerprint = hashlib.sha256(
            json.dumps(
                session_request, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        created = request.app.state.manager.begin_session(
            owner_id,
            benchmark,
            1,
            idempotency_key="session_" + idempotency_key,
            request_fingerprint=session_fingerprint,
            run_id=run["run_id"],
            task_id=task_id,
            seed=body.seed,
            episode_length=episode_length,
            observation_level=body.observation.level,
            bbox_scope=body.observation.bbox_scope,
        )
        agent_token = token_store.issue_session_token(
            run_id=run["run_id"],
            session_id=created["session_id"],
            owner_id=owner_id,
            benchmark=benchmark,
            observation_level=body.observation.level,
            episode_steps=episode_length,
            ttl_seconds=settings.run_token_ttl_seconds,
        )
        response.headers["Cache-Control"] = "no-store"
        return {
            "run": run,
            "session": build_create_response_v2(
                created, expose_seed=body.seed is not None
            ),
            "agent_token": agent_token,
        }

    @app.post(
        "/v1/sessions",
        response_model=CreateSessionV1Response,
        response_model_exclude_none=True,
        status_code=201,
    )
    def create_session_v1(
        body: CreateSessionRequest,
        request: Request,
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        require_scope(identity, SESSION_OPERATE_SCOPE)
        if identity.bound_run_id is not None:
            raise AuthorizationError()
        if body.benchmark not in identity.allowed_benchmarks:
            return _error_response("BENCHMARK_NOT_ALLOWED", 403)
        if settings.fixed_benchmark and body.benchmark != settings.fixed_benchmark:
            return _error_response("BENCHMARK_NOT_ALLOWED", 403)
        try:
            created = request.app.state.manager.create_session(
                identity.owner_id,
                body.benchmark,
                identity.max_sessions,
                task_id=(
                    settings.fixed_task_id if settings.fixed_benchmark else None
                ),
                episode_length=settings.default_episode_steps,
                observation_level=1,
            )
        except (QuotaError, CapacityError):
            # Preserve the deployed v1 error contract used by legacy clients.
            return _error_response("CAPACITY_EXHAUSTED", 503, headers={"Retry-After": "5"})
        return CreateSessionV1Response(
            session_id=created["session_id"],
            instruction=created["instruction"],
            action_spec=ActionSpecV1(),
            observation_spec=ObservationSpecV1(
                cameras={
                    "agentview_rgb": [settings.image_height, settings.image_width, 3],
                    "wrist_rgb": [settings.image_height, settings.image_width, 3],
                },
                proprioception_keys=list(BASE_PROPRIO_KEYS),
            ),
        )

    @app.post(
        "/v2/sessions",
        response_model=CreateSessionResponse,
        response_model_exclude_none=True,
        status_code=202,
    )
    def create_session_v2(
        body: CreateSessionV2Request,
        request: Request,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=8, max_length=128),
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        require_scope(identity, SESSION_OPERATE_SCOPE)
        if identity.bound_session_id is not None:
            raise AuthorizationError()
        if (
            identity.bound_run_id is not None
            and body.run_id != identity.bound_run_id
        ):
            raise AuthorizationError()
        benchmark = body.task.benchmark
        if benchmark not in identity.allowed_benchmarks:
            return _error_response("TASK_FORBIDDEN", 403)
        if (
            body.task.task_id is not None
            and body.task.task_id >= BENCHMARK_TASK_COUNTS[benchmark]
        ):
            return _error_response("TASK_NOT_FOUND", 422)
        if body.observation.level > identity.max_observation_level:
            return _error_response("OBSERVATION_LEVEL_FORBIDDEN", 403)
        if body.seed is not None and not identity.allow_fixed_seed:
            return _error_response("SEED_SELECTION_FORBIDDEN", 403)
        maximum = min(settings.max_episode_steps, identity.max_episode_steps)
        episode_length = (
            min(settings.default_episode_steps, maximum)
            if body.episode_length is None
            else body.episode_length
        )
        if not 1 <= episode_length <= maximum:
            return _error_response("MAX_STEPS_OUT_OF_RANGE", 422)
        canonical_request = {
            "run_id": body.run_id,
            "task": body.task.model_dump(mode="json"),
            "episode_length": episode_length,
            "observation": body.observation.model_dump(mode="json"),
            "seed": body.seed,
        }
        request_fingerprint = hashlib.sha256(
            json.dumps(
                canonical_request, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        created = request.app.state.manager.begin_session(
            identity.owner_id,
            benchmark,
            identity.max_sessions,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            run_id=body.run_id,
            task_id=body.task.task_id,
            seed=body.seed,
            episode_length=episode_length,
            observation_level=body.observation.level,
            bbox_scope=body.observation.bbox_scope,
        )
        return build_create_response_v2(created, expose_seed=body.seed is not None)

    @app.get(
        "/v2/sessions/{session_id}",
        response_model=CreateSessionResponse,
        response_model_exclude_none=True,
    )
    def session_status_v2(
        session_id: str,
        request: Request,
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        require_scope(identity, SESSION_OPERATE_SCOPE)
        require_bound_session(identity, session_id)
        created = request.app.state.manager.session_status(
            session_id, identity.owner_id
        )
        return build_create_response_v2(created, expose_seed=False)

    def reset_impl(
        session_id: str,
        request: Request,
        idempotency_key: Optional[str] = Header(
            default=None, alias="Idempotency-Key", min_length=8, max_length=128
        ),
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        require_scope(identity, SESSION_OPERATE_SCOPE)
        require_bound_session(identity, session_id)
        return request.app.state.manager.reset(
            session_id,
            identity.owner_id,
            idempotency_key=idempotency_key,
        )

    def step_impl(
        session_id: str,
        body: StepRequest,
        request: Request,
        idempotency_key: Optional[str] = Header(
            default=None, alias="Idempotency-Key", min_length=8, max_length=128
        ),
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        require_scope(identity, SESSION_OPERATE_SCOPE)
        require_bound_session(identity, session_id)
        return request.app.state.manager.step(
            session_id,
            identity.owner_id,
            body.action,
            idempotency_key=idempotency_key,
            expected_step_index=body.expected_step_index,
        )

    def result_impl(
        session_id: str,
        request: Request,
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        require_scope(identity, SESSION_OPERATE_SCOPE)
        require_bound_session(identity, session_id)
        return request.app.state.manager.result(session_id, identity.owner_id)

    def close_impl(
        session_id: str,
        request: Request,
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        require_scope(identity, SESSION_OPERATE_SCOPE)
        require_bound_session(identity, session_id)
        return request.app.state.manager.close_session(
            session_id, identity.owner_id
        )

    @app.get(
        "/v2/sessions/{session_id}/harness-progress",
        response_model=HarnessTaskProgressResponse,
        response_model_exclude_none=True,
    )
    def harness_progress_impl(
        session_id: str,
        request: Request,
        identity: AgentIdentity = Depends(identity_from_header),
    ):
        """Trusted finalization signal; the local Agent proxy does not expose it."""

        require_scope(identity, SESSION_OPERATE_SCOPE)
        require_bound_session(identity, session_id)
        return request.app.state.manager.task_progress(
            session_id, identity.owner_id
        )

    for version in ("v1", "v2"):
        app.add_api_route(
            f"/{version}/sessions/{{session_id}}/reset",
            reset_impl,
            methods=["POST"],
            response_model=ResetResponse,
            response_model_exclude_none=True,
        )
        app.add_api_route(
            f"/{version}/sessions/{{session_id}}/step",
            step_impl,
            methods=["POST"],
            response_model=StepResponse,
            response_model_exclude_none=True,
        )
        app.add_api_route(
            f"/{version}/sessions/{{session_id}}/result",
            result_impl,
            methods=["GET"],
            response_model=ResultResponse,
        )
        app.add_api_route(
            f"/{version}/sessions/{{session_id}}",
            close_impl,
            methods=["DELETE"],
        )

    return app


def _error_response(
    code: str, status: int, *, headers: Optional[dict] = None
) -> JSONResponse:
    request_id = "req_" + secrets.token_urlsafe(12)
    response_headers = {"Cache-Control": "no-store"}
    if headers:
        response_headers.update(headers)
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "request_id": request_id}},
        headers=response_headers,
    )


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        access_log=True,
        server_header=False,
    )


if __name__ == "__main__":
    main()
