"""Pure ASGI middleware (does not buffer request bodies, so streaming uploads stay streamed)."""

import json
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from datamedic.logging_setup import request_id_var

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = str(uuid.uuid4())
        token = request_id_var.set(rid)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-request-id", rid.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)


class OriginCheckMiddleware:
    """Reject unsafe requests whose Origin header names a foreign origin (CSRF defence)."""

    def __init__(self, app: ASGIApp, allowed_origin: str) -> None:
        self.app = app
        self.allowed = allowed_origin.rstrip("/").encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in UNSAFE_METHODS:
            origin = dict(scope["headers"]).get(b"origin")
            if origin is not None and origin.rstrip(b"/") != self.allowed:
                body = json.dumps(
                    {
                        "type": "https://docs.datamedic.app/errors/csrf-failed",
                        "title": "CSRF validation failed",
                        "status": 403,
                        "code": "CSRF_FAILED",
                        "detail": "Request origin is not allowed.",
                        "request_id": request_id_var.get(),
                    }
                ).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [
                            (b"content-type", b"application/problem+json"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


class NoStoreMiddleware:
    """API responses are user-specific: never cache them."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_no_store(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"cache-control", b"no-store"))
                headers.append((b"x-content-type-options", b"nosniff"))
                headers.append((b"referrer-policy", b"no-referrer"))
            await send(message)

        await self.app(scope, receive, send_no_store)
