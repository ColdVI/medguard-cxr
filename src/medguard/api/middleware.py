"""FastAPI middleware backstops for Phase 4 safety metadata."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from medguard.api.schemas import SAFETY_DISCLAIMER, ModelProvenance, ProblemDetails

PHASE = "4"


def is_available() -> bool:
    """Return whether Phase 4 API middleware is implemented."""

    return True


class ProvenanceMiddleware(BaseHTTPMiddleware):
    """Inject model provenance into JSON responses if an endpoint omitted it."""

    def __init__(self, app: Any, provenance: ModelProvenance) -> None:
        super().__init__(app)
        self.provenance = provenance

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        payload = await _json_payload(response)
        if payload is None:
            return response
        if isinstance(payload, dict) and "model_provenance" not in payload:
            payload["model_provenance"] = self.provenance.model_dump()
        return JSONResponse(payload, status_code=response.status_code)


class DisclaimerMiddleware(BaseHTTPMiddleware):
    """Reject any outgoing JSON body missing the canonical disclaimer."""

    def __init__(self, app: Any, provenance: ModelProvenance) -> None:
        super().__init__(app)
        self.provenance = provenance

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        payload = await _json_payload(response)
        if payload is None:
            problem = _problem("Missing JSON response body.", 500, self.provenance)
            return JSONResponse(problem, status_code=500)
        if not isinstance(payload, dict) or payload.get("safety_disclaimer") != SAFETY_DISCLAIMER:
            problem = _problem(
                "Outgoing response was missing the required safety disclaimer.",
                500,
                self.provenance,
            )
            return JSONResponse(problem, status_code=500)
        return JSONResponse(payload, status_code=response.status_code)


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Append endpoint decision metadata without logging image bytes."""

    def __init__(self, app: Any, log_path: str | Path = "logs/api_audit.jsonl") -> None:
        super().__init__(app)
        self.log_path = Path(log_path)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        body = await request.body()

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)
        response = await call_next(request)
        payload = await _json_payload(response)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._write_log(request, payload, body, latency_ms)
        if payload is None:
            return response
        return JSONResponse(payload, status_code=response.status_code)

    def _write_log(
        self,
        request: Request,
        payload: Any,
        body: bytes,
        latency_ms: float,
    ) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        reason = ""
        if isinstance(payload, dict):
            reason = str(payload.get("reason") or payload.get("ood", {}).get("reason") or "")
        status_value = payload.get("status", 200) if isinstance(payload, dict) else 200
        is_error = isinstance(status_value, int) and status_value >= 400
        record = {
            "endpoint": request.url.path,
            "image_hash": hashlib.sha256(body).hexdigest()[:16] if body else "",
            "decision": "error" if is_error else "ok",
            "reason": reason,
            "latency_ms": round(latency_ms, 3),
        }
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


async def _json_payload(response: Response) -> Any | None:
    body = b""
    async for chunk in response.body_iterator:
        body += chunk
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _problem(detail: str, status: int, provenance: ModelProvenance) -> dict[str, Any]:
    return ProblemDetails(
        title="Safety metadata enforcement failed",
        status=status,
        detail=detail,
        model_provenance=provenance,
    ).model_dump()
