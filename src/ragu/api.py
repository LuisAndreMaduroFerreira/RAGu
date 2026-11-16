import logging
from contextlib import asynccontextmanager
from typing import Annotated
from urllib.parse import urlparse
from uuid import UUID

from fastapi import Body, Depends, FastAPI, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ragu.auth import Principal
from ragu.config import Settings
from ragu.errors import AppError
from ragu.management import HttpMetricsMiddleware
from ragu.mcp_server import build_mcp
from ragu.runtime import Runtime
from ragu.schemas import (
    DEFAULT_SOURCE,
    CandidateQuery,
    ConfluenceCredential,
    CorpusCreate,
    DailySchedule,
    GrantUpdate,
    IngestionRequest,
    JobView,
    Login,
    MemberUpdate,
    MultiSourceRequest,
    PasswordChange,
    ProjectCreate,
    RerankRequest,
    SearchQuery,
    SourceKind,
    TokenView,
    UserView,
)

logger = logging.getLogger("ragu.api")


class SecurityMiddleware:
    def __init__(self, app, https_only: bool):
        self.app, self.https_only = app, https_only

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def secured(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [
                    (b"x-content-type-options", b"nosniff"),
                    (b"cache-control", b"no-store"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                ]
                if not any(key.lower() == b"content-security-policy" for key, _ in headers):
                    headers.append((b"content-security-policy", b"frame-ancestors 'none'"))
                if self.https_only:
                    headers.append(
                        (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
                    )
                message["headers"] = headers
            await send(message)

        if self.https_only and scope["scheme"] != "https":
            return await JSONResponse({"detail": "HTTPS required."}, status_code=400)(
                scope, receive, secured
            )
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > 65536:
                return await JSONResponse({"detail": "Request too large."}, status_code=413)(
                    scope, receive, secured
                )
            if not message.get("more_body"):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, secured)


def create_app(runtime: Runtime | None = None) -> FastAPI:
    runtime = runtime or Runtime(Settings())
    settings = runtime.settings
    mcp = build_mcp(runtime)

    @asynccontextmanager
    async def lifespan(app):
        try:
            async with mcp.router.lifespan_context(mcp):
                yield
        finally:
            runtime.close()

    local = urlparse(settings.public_url).hostname in {"localhost", "127.0.0.1", "::1"}
    app = FastAPI(
        title="RAGU",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if local else None,
        redoc_url=None,
        openapi_url="/openapi.json" if local else None,
    )
    app.state.runtime = runtime
    app.add_middleware(SecurityMiddleware, https_only=settings.public_url.startswith("https://"))
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[urlparse(settings.public_url).hostname]
        + (["localhost", "127.0.0.1", "testserver"] if local else []),
    )
    app.add_middleware(HttpMetricsMiddleware, management=runtime.management)
    bearer = HTTPBearer(auto_error=False)
    MonitoringCredential = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]

    def identity(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> Principal:
        if credentials is None:
            raise AppError(401, "Authentication required.")
        return runtime.auth.authenticate(credentials.credentials, settings.api_audience, "api")

    Current = Annotated[Principal, Depends(identity)]

    @app.exception_handler(AppError)
    async def application_error(request: Request, error: AppError):
        logger.warning("request_rejected status=%s path=%s", error.status, request.url.path)
        headers = {"WWW-Authenticate": "Bearer"} if error.status == 401 else {}
        if error.status == 429:
            headers["Retry-After"] = "300"
        return JSONResponse({"detail": error.message}, status_code=error.status, headers=headers)

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, error: SQLAlchemyError):
        logger.exception("database_request_failed", exc_info=error)
        return JSONResponse({"detail": "Database operation failed."}, status_code=503)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        return JSONResponse(
            {
                "detail": [
                    {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
                    for item in error.errors()
                ]
            },
            status_code=422,
        )

    @app.get("/actuator/prometheus", include_in_schema=False)
    def prometheus(credentials: MonitoringCredential):
        return runtime.management.prometheus(credentials.credentials if credentials else None)

    @app.get("/actuator/health/liveness", include_in_schema=False)
    def liveness():
        return runtime.management.liveness()

    @app.get("/actuator/health", include_in_schema=False)
    @app.get("/actuator/health/readiness", include_in_schema=False)
    def readiness(credentials: MonitoringCredential):
        return runtime.management.readiness(credentials.credentials if credentials else None)

    @app.post("/auth/login", response_model=TokenView)
    def login(data: Login, request: Request):
        return runtime.auth.login(data, request.client.host if request.client else "unknown")

    @app.get("/auth/me", response_model=UserView)
    def me(principal: Current):
        return runtime.auth.me(principal)

    @app.post("/auth/password")
    def password(data: PasswordChange, principal: Current):
        return runtime.auth.change_password(data, principal)

    @app.post("/auth/logout-all")
    def logout(principal: Current):
        return runtime.auth.logout_all(principal)

    @app.put("/auth/confluence-credential")
    def confluence_credential(data: ConfluenceCredential, principal: Current):
        return runtime.auth.set_confluence_token(data, principal)

    @app.post("/ingestions", status_code=202, response_model=JobView)
    def ingest(
        principal: Current,
        data: Annotated[IngestionRequest, Body(default_factory=IngestionRequest)],
    ):
        return runtime.ingestion.start_request(data, principal)

    @app.get("/ingestions", response_model=list[JobView])
    def jobs(principal: Current, corpus_id: UUID | None = None):
        return runtime.ingestion.list_jobs(principal, corpus_id)

    @app.post("/ingestions/multi-source", status_code=202, response_model=JobView)
    def multi_source_ingest(data: MultiSourceRequest, principal: Current):
        return runtime.ingestion.start_multi_source(principal, data)

    @app.get("/ingestions/{job_id}", response_model=JobView)
    def job(job_id: UUID, principal: Current):
        return runtime.ingestion.status(job_id, principal)

    @app.put("/schedules/daily")
    def schedule(data: DailySchedule, principal: Current):
        return runtime.ingestion.configure_schedule(data, principal)

    @app.get("/schedules/daily")
    def get_schedule(principal: Current, corpus_id: UUID | None = None):
        return runtime.ingestion.schedule(principal, corpus_id)

    @app.post("/documents/search")
    def search(data: SearchQuery, principal: Current):
        return runtime.search.search(data, principal)

    @app.post("/documents/candidates")
    def candidates(data: CandidateQuery, principal: Current):
        return runtime.search.candidates(data, principal)

    @app.post("/documents/rerank")
    def rerank(data: RerankRequest, principal: Current):
        return runtime.search.rerank(data, principal)

    @app.get("/documents/capabilities")
    def capabilities(
        principal: Current, source: SourceKind = DEFAULT_SOURCE, corpus_id: UUID | None = None
    ):
        return runtime.search.capabilities(principal, source, corpus_id)

    @app.get("/documents/{page_id}")
    def document(
        page_id: str,
        principal: Current,
        source: SourceKind = DEFAULT_SOURCE,
        corpus_id: UUID | None = None,
    ):
        return runtime.search.document(page_id, principal, source, corpus_id)

    @app.post("/projects", status_code=201)
    def create_project(data: ProjectCreate, principal: Current):
        return runtime.projects.create(data, principal)

    @app.get("/projects")
    def projects(principal: Current):
        return runtime.projects.list(principal)

    @app.put("/projects/{project_id}/members/{user_id}")
    def put_member(project_id: UUID, user_id: UUID, data: MemberUpdate, principal: Current):
        return runtime.projects.member(project_id, user_id, data, principal)

    @app.delete("/projects/{project_id}/members/{user_id}")
    def delete_member(project_id: UUID, user_id: UUID, principal: Current):
        return runtime.projects.member(project_id, user_id, None, principal)

    @app.post("/projects/{project_id}/corpora", status_code=201)
    def create_corpus(project_id: UUID, data: CorpusCreate, principal: Current):
        return runtime.projects.create_corpus(project_id, data, principal)

    @app.get("/projects/{project_id}/corpora")
    def corpora(project_id: UUID, principal: Current):
        return runtime.projects.corpora(project_id, principal)

    @app.put("/corpora/{corpus_id}/grants/{user_id}")
    def put_grant(corpus_id: UUID, user_id: UUID, data: GrantUpdate, principal: Current):
        return runtime.projects.grant(corpus_id, user_id, data, principal)

    @app.delete("/corpora/{corpus_id}/grants/{user_id}")
    def delete_grant(corpus_id: UUID, user_id: UUID, principal: Current):
        return runtime.projects.grant(corpus_id, user_id, None, principal)

    @app.get("/.well-known/oauth-authorization-server")
    def oauth_metadata():
        return runtime.oauth.metadata()

    @app.get("/oauth/authorize")
    def authorize(request: Request):
        return runtime.oauth.authorize(dict(request.query_params), request.client.host)

    @app.post("/oauth/authorize")
    def consent(
        request: Request,
        request_id: Annotated[str, Form()],
        csrf: Annotated[str, Form()],
        username: Annotated[str, Form()],
        password: Annotated[str, Form()],
    ):
        return runtime.oauth.consent(
            request_id,
            csrf,
            request.cookies.get("ragu_oauth_csrf", ""),
            username,
            password,
            request.headers.get("origin", ""),
            request.client.host,
        )

    @app.post("/oauth/token")
    def oauth_token(
        request: Request,
        grant_type: Annotated[str, Form()],
        code: Annotated[str, Form()],
        code_verifier: Annotated[str, Form()],
        client_id: Annotated[str, Form()],
        redirect_uri: Annotated[str, Form()],
        resource: Annotated[str, Form()],
    ):
        return runtime.oauth.token(
            {
                "grant_type": grant_type,
                "code": code,
                "code_verifier": code_verifier,
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "resource": resource,
            },
            request.client.host,
        )

    app.mount("/", mcp)
    return app
