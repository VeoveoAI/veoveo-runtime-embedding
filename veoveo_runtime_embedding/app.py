"""Private bounded HTTP surface. Errors never echo Pydantic input or upstream bodies."""
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import threading
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from .contract import *


def problem(code: str, status: int, execution: str, request_id: str | None = None) -> JSONResponse:
    return JSONResponse({"type": "https://veoveo.ai/problems/" + code.replace("_", "-"),
                         "title": code, "status": status, "execution": execution,
                         "requestId": request_id}, status_code=status,
                        media_type="application/problem+json", headers={"Cache-Control": "no-store"})


def strict_json(raw: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError()
            result[key] = value
        return result
    def invalid(_): raise ValueError()
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid)


class Admission:
    """A GPU slot belongs to the real thread, even after its caller leaves."""
    def __init__(self, backend):
        self.backend = backend
        self.slot = threading.BoundedSemaphore(1)
        self.pending = set()

    def _work(self, batch):
        try:
            return self.backend.encode(batch)
        finally:
            self.slot.release()

    def submit(self, batch):
        if not self.slot.acquire(blocking=False):
            raise Rejected("busy", 429)
        try:
            task = asyncio.create_task(asyncio.to_thread(self._work, batch))
        except BaseException:
            self.slot.release()
            raise
        self.pending.add(task)
        def finished(task):
            self.pending.discard(task)
            if not task.cancelled(): task.exception()
        task.add_done_callback(finished)
        return task


def create_app(backend_factory, credential: str, deadline: float = REQUEST_SECONDS) -> FastAPI:
    if not credential or len(credential) > 512 or any(character.isspace() for character in credential):
        raise RuntimeError("a valid private runtime bearer is required")
    state = {}
    @asynccontextmanager
    async def lifespan(_app):
        state["backend"] = await asyncio.to_thread(backend_factory)
        state["admission"] = Admission(state["backend"])
        yield
        # Native OS shutdown owns process termination; no inference replay or daemon.

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def authorized(request):
        return hmac.compare_digest(request.headers.get("authorization", "").encode(), ("Bearer " + credential).encode())

    @app.middleware("http")
    async def safe_boundary(request: Request, next_handler):
        if request.url.path != "/livez" and not authorized(request):
            return problem("unauthorized", 401, "not_started")
        try:
            response = await next_handler(request)
            response.headers["Cache-Control"] = "no-store"
            return response
        except Exception:
            return problem("runtime_failure", 500, "unknown")

    @app.get("/livez")
    async def live():
        return Response("alive")

    @app.get("/readyz")
    async def ready():
        backend = state.get("backend")
        return Response("ready" if backend and backend.ready() else "not ready", status_code=200 if backend and backend.ready() else 503)

    @app.get("/capabilities")
    async def capabilities():
        backend = state.get("backend")
        if not backend or not backend.ready():
            return problem("not_ready", 503, "not_started")
        return JSONResponse(backend.capabilities())

    @app.post("/embeddings")
    async def embeddings(request: Request):
        identifier = None
        try:
            length = request.headers.get("content-length")
            if length is not None and (not length.isdecimal() or int(length) > MAX_JSON_BYTES):
                raise Rejected("request_size", 413)
            raw = bytearray()
            async for chunk in request.stream():
                if len(raw) + len(chunk) > MAX_JSON_BYTES: raise Rejected("request_size", 413)
                raw.extend(chunk)
            try:
                batch = parse_batch(strict_json(raw))
            except (ValueError, TypeError, UnicodeError, RecursionError):
                raise Rejected("invalid_json", 400) from None
            identifier = str(batch.request_id)
            admission = state.get("admission")
            if not admission: raise Rejected("not_ready", 503)
            task = admission.submit(batch)
            try:
                result = await asyncio.wait_for(asyncio.shield(task), timeout=deadline)
            except TimeoutError:
                return problem("deadline_exceeded", 504, "unknown", identifier)
            # Bound final JSON and forbid NaN/Infinity, without logging raw output.
            raw_result = json.dumps(result, allow_nan=False, separators=(",", ":")).encode()
            if len(raw_result) > MAX_RESPONSE_BYTES:
                return problem("invalid_output", 500, "unknown", identifier)
            return Response(raw_result, media_type="application/json")
        except Rejected as error:
            return problem(error.code, error.status, "not_started", identifier)
        except Exception:
            return problem("inference_failed", 500, "unknown", identifier)
    return app
