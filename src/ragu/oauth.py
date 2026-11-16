import base64
import hashlib
import hmac
import re
import secrets
from datetime import timedelta
from html import escape
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID, uuid4

from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ragu.database import OAuthCode, OAuthRequest, Role, User, aware, utcnow
from ragu.errors import AppError


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class OAuthService:
    def __init__(self, db, auth, settings):
        self.db, self.auth, self.settings = db, auth, settings

    def metadata(self) -> dict:
        origin = self.settings.public_url
        return {
            "issuer": origin,
            "authorization_endpoint": f"{origin}/oauth/authorize",
            "token_endpoint": f"{origin}/oauth/token",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "token_endpoint_auth_methods_supported": ["none"],
            "code_challenge_methods_supported": ["S256"],
            "scopes_supported": ["documents:read"],
        }

    def _enabled(self):
        if not self.settings.allow_cloud_context:
            raise AppError(
                503, "MCP requires explicit approval to send retrieved context to clients."
            )
        if not self.settings.oauth_redirect_uris:
            raise AppError(503, "No OAuth callback URI has been registered.")

    def _validate(self, data: dict):
        self._enabled()
        if (
            data.get("client_id") != self.settings.oauth_client_id
            or data.get("redirect_uri") not in self.settings.oauth_redirect_uris
            or data.get("response_type") != "code"
            or data.get("resource") != self.settings.mcp_resource
            or data.get("scope") != "documents:read"
            or data.get("code_challenge_method") != "S256"
            or not re.fullmatch(r"[A-Za-z0-9_-]{43}", data.get("code_challenge", ""))
            or len(data.get("state", "")) > 1024
        ):
            raise AppError(
                400, "Invalid OAuth request, callback, scope, resource or PKCE challenge."
            )

    def _form(self, request_id, csrf: str, error: str = "") -> HTMLResponse:
        html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Authorize RAGU</title><h1>Authorize RAGU read-only access</h1>
<p>Allow {escape(self.settings.oauth_client_id)} to read your authorized local, Confluence,
multi-source and project corpora, including candidate selection and reranking.
No ingestion or administration permissions will be granted. Returned evidence may be
sent to this client's model provider; use only an approved client and destination.</p>
<p>{escape(error)}</p><form method="post" action="/oauth/authorize">
<input type="hidden" name="request_id" value="{request_id}">
<input type="hidden" name="csrf" value="{escape(csrf)}">
<label>Username <input name="username" autocomplete="username" maxlength="128" required></label>
<label>Password <input type="password" name="password" autocomplete="current-password"
maxlength="128" required></label><button type="submit">Sign in and allow read access</button>
</form></html>"""
        response = HTMLResponse(html, status_code=401 if error else 200)
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        response.set_cookie(
            "ragu_oauth_csrf",
            csrf,
            max_age=600,
            httponly=True,
            samesite="lax",
            secure=self.settings.public_url.startswith("https://"),
            path="/oauth",
        )
        return response

    def authorize(self, params: dict, client_ip: str):
        self._validate(params)
        self.auth.throttle("oauth_authorize", client_ip, 30)
        csrf = secrets.token_urlsafe(32)
        request_id = uuid4()
        with self.db.session() as session:
            session.add(
                OAuthRequest(
                    id=request_id,
                    data=params,
                    csrf_hash=digest(csrf),
                    expires_at=utcnow() + timedelta(minutes=10),
                )
            )
        return self._form(request_id, csrf)

    def consent(
        self,
        request_id: str,
        csrf: str,
        cookie: str,
        username: str,
        password: str,
        origin: str,
        client_ip: str,
    ):
        self._enabled()
        if (
            origin != self.settings.public_url
            or not csrf
            or not cookie
            or not hmac.compare_digest(digest(csrf), digest(cookie))
        ):
            raise AppError(403, "Invalid authorization form origin or CSRF token.")
        try:
            identifier = UUID(request_id)
        except ValueError as exc:
            raise AppError(400, "Invalid authorization request.") from exc
        with self.db.session() as session:
            pending = session.get(OAuthRequest, identifier)
            if (
                pending is None
                or aware(pending.expires_at) <= utcnow()
                or not hmac.compare_digest(pending.csrf_hash, digest(csrf))
            ):
                raise AppError(400, "Authorization request expired or already used.")
            data = dict(pending.data)
            self._validate(data)
        if len(username) > 128 or len(password) > 128:
            raise AppError(400, "Invalid credentials.")
        try:
            principal = self.auth.credentials(username, password, client_ip)
        except AppError as exc:
            if exc.status != 401:
                raise
            return self._form(identifier, csrf, "Invalid credentials.")
        code = secrets.token_urlsafe(32)
        with self.db.session() as session:
            pending = session.get(OAuthRequest, identifier, with_for_update=True)
            if pending is None or aware(pending.expires_at) <= utcnow():
                raise AppError(400, "Authorization request expired or already used.")
            self.auth.current(session, principal)
            session.add(
                OAuthCode(
                    code_hash=digest(code),
                    user_id=principal.id,
                    auth_version=principal.auth_version,
                    data=data,
                    expires_at=utcnow() + timedelta(seconds=120),
                )
            )
            session.delete(pending)
        redirect = urlsplit(data["redirect_uri"])
        query = parse_qsl(redirect.query, keep_blank_values=True)
        query += [("code", code), ("iss", self.settings.public_url)]
        if data.get("state"):
            query.append(("state", data["state"]))
        response = RedirectResponse(
            urlunsplit(redirect._replace(query=urlencode(query))), status_code=303
        )
        response.delete_cookie("ragu_oauth_csrf", path="/oauth")
        return response

    def token(self, data: dict, client_ip: str):
        self._enabled()
        self.auth.throttle("oauth_token", client_ip, 30)
        if (
            data.get("grant_type") != "authorization_code"
            or not re.fullmatch(r"[A-Za-z0-9._~-]{43,128}", data.get("code_verifier", ""))
            or len(data.get("code", "")) > 256
        ):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        principal = None
        with self.db.session() as session:
            entry = session.get(OAuthCode, digest(data.get("code", "")), with_for_update=True)
            if entry is not None:
                challenge = (
                    base64.urlsafe_b64encode(
                        hashlib.sha256(data["code_verifier"].encode()).digest()
                    )
                    .rstrip(b"=")
                    .decode()
                )
                valid = (
                    aware(entry.expires_at) > utcnow()
                    and entry.data["client_id"] == data.get("client_id")
                    and entry.data["redirect_uri"] == data.get("redirect_uri")
                    and entry.data["resource"] == data.get("resource") == self.settings.mcp_resource
                    and hmac.compare_digest(entry.data["code_challenge"], challenge)
                )
                user = session.get(User, entry.user_id)
                role = session.get(Role, user.role_id) if user else None
                if (
                    valid
                    and user
                    and role
                    and user.is_active
                    and user.auth_version == entry.auth_version
                ):
                    from ragu.auth import Principal

                    principal = Principal(
                        user.id,
                        user.username,
                        role.code,
                        user.auth_version,
                        frozenset({"documents:read"}),
                    )
                session.delete(entry)
        if principal is None:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        token = self.auth.issue(
            principal,
            self.settings.mcp_resource,
            ["documents:read"],
            client_id=self.settings.oauth_client_id,
        )
        return JSONResponse({**token.model_dump(), "scope": "documents:read"})
