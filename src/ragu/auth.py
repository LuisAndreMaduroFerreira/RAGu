import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

import jwt
from cryptography.fernet import Fernet, InvalidToken
from pwdlib import PasswordHash
from sqlalchemy import func, select

from ragu.database import Database, Role, Throttle, User, aware, utcnow
from ragu.errors import AppError
from ragu.schemas import ConfluenceCredential, Login, PasswordChange, TokenView, UserView

logger = logging.getLogger("ragu.audit")


@dataclass(frozen=True)
class Principal:
    id: UUID
    username: str
    role: str
    auth_version: int
    scopes: frozenset[str]


class AuthService:
    def __init__(self, db: Database, settings):
        self.db = db
        self.settings = settings
        self.hasher = PasswordHash.recommended()
        self.dummy_hash = self.hasher.hash(secrets.token_urlsafe(32))
        self.cipher = Fernet(settings.credential_key.get_secret_value().encode())

    def throttle(self, category: str, identifier: str, limit: int, seconds: int = 300) -> None:
        key = hmac.new(
            self.settings.jwt_secret.get_secret_value().encode(),
            f"{category}:{identifier}".encode(),
            hashlib.sha256,
        ).hexdigest()
        exceeded = False
        with self.db.session() as session:
            Database.lock(session, int(key[:15], 16))
            entry = session.get(Throttle, key)
            now = utcnow()
            if entry is None:
                session.add(Throttle(key=key, attempts=1, window_start=now))
            elif (now - aware(entry.window_start)).total_seconds() >= seconds:
                entry.attempts = 1
                entry.window_start = now
            else:
                entry.attempts += 1
                exceeded = entry.attempts > limit
        if exceeded:
            logger.warning("rate_limit category=%s", category)
            raise AppError(429, "Too many requests; try again later.")

    def credentials(self, username: str, password: str, client_ip: str) -> Principal:
        self.throttle("login_ip", client_ip, 30)
        self.throttle("login_user", username.strip().lower(), 10)
        with self.db.session() as session:
            row = session.execute(
                select(User, Role.code)
                .join(Role)
                .where(func.lower(User.username) == username.strip().lower())
                .with_for_update(of=User)
            ).first()
            stored_hash = row[0].password_hash if row else self.dummy_hash
            valid, replacement = self.hasher.verify_and_update(password, stored_hash)
            if not row or not valid or not row[0].is_active:
                logger.warning("login_failed")
                raise AppError(401, "Invalid credentials.")
            user, role = row
            if replacement:
                user.password_hash = replacement
            user.last_login_at = utcnow()
            logger.info("login_succeeded user_id=%s", user.id)
            return Principal(user.id, user.username, role, user.auth_version, frozenset({"api"}))

    def issue(
        self,
        principal: Principal,
        audience: str,
        scopes: list[str],
        *,
        client_id: str | None = None,
    ) -> TokenView:
        now = utcnow()
        expiry = now + timedelta(minutes=self.settings.access_token_minutes)
        payload = {
            "sub": str(principal.id),
            "ver": principal.auth_version,
            "scope": " ".join(scopes),
            "iss": self.settings.public_url,
            "aud": audience,
            "iat": now,
            "exp": expiry,
            "jti": secrets.token_urlsafe(24),
        }
        if client_id:
            payload["client_id"] = client_id
        token = jwt.encode(payload, self.settings.jwt_secret.get_secret_value(), algorithm="HS256")
        return TokenView(access_token=token, expires_in=int((expiry - now).total_seconds()))

    def login(self, data: Login, client_ip: str) -> TokenView:
        return self.issue(
            self.credentials(data.username, data.password.get_secret_value(), client_ip),
            self.settings.api_audience,
            ["api"],
        )

    def create_reader(self, username: str, password: str, principal: Principal) -> dict:
        username = username.strip()
        if not username or len(username) > 128 or not 15 <= len(password) <= 128:
            raise AppError(422, "Use a 1-128 character username and a 15-128 character password.")
        with self.db.session() as session:
            Database.lock(session, 1001)
            self.current(session, principal, admin=True)
            if session.scalar(select(User.id).where(func.lower(User.username) == username.lower())):
                raise AppError(409, "Username already exists.")
            role = session.scalar(select(Role).where(Role.code == "reader"))
            if role is None:
                raise AppError(409, "Run migrations before creating a reader.")
            user = User(
                username=username,
                password_hash=self.hasher.hash(password),
                role_id=role.id,
            )
            session.add(user)
            session.flush()
            logger.info("reader_created user_id=%s actor_id=%s", user.id, principal.id)
            return {"id": str(user.id), "username": user.username, "role": role.code}

    def authenticate(self, token: str, audience: str, scope: str) -> Principal:
        try:
            payload = jwt.decode(
                token,
                self.settings.jwt_secret.get_secret_value(),
                algorithms=["HS256"],
                issuer=self.settings.public_url,
                audience=audience,
                options={"require": ["sub", "ver", "scope", "iss", "aud", "iat", "exp", "jti"]},
            )
            user_id = UUID(payload["sub"])
            version = payload["ver"]
            if type(version) is not int or not isinstance(payload["scope"], str):
                raise ValueError("Invalid token claim types")
            scopes = frozenset(payload["scope"].split())
        except (jwt.InvalidTokenError, ValueError, TypeError, KeyError) as exc:
            raise AppError(401, "Invalid or expired access token.") from exc
        with self.db.session() as session:
            row = session.execute(
                select(User, Role.code).join(Role).where(User.id == user_id)
            ).first()
            if not row or not row[0].is_active or row[0].auth_version != version:
                raise AppError(401, "Invalid or expired access token.")
            if scope not in scopes:
                raise AppError(403, "Insufficient permission.")
            return Principal(user_id, row[0].username, row[1], version, scopes)

    def current(self, session, principal: Principal, *, admin: bool = False) -> User:
        # Non-key locks serialize account changes without blocking corpus-grant foreign keys.
        user = session.get(User, principal.id, with_for_update={"key_share": True})
        if user is None or not user.is_active or user.auth_version != principal.auth_version:
            raise AppError(401, "Account credentials are no longer valid.")
        role = session.get(Role, user.role_id)
        if admin and (role is None or role.code != "admin"):
            logger.warning("authorization_denied user_id=%s", principal.id)
            raise AppError(403, "Admin role required.")
        return user

    def me(self, principal: Principal) -> UserView:
        return UserView(id=principal.id, username=principal.username, role=principal.role)

    def change_password(self, data: PasswordChange, principal: Principal) -> dict:
        self.throttle("password", str(principal.id), 5)
        with self.db.session() as session:
            user = self.current(session, principal)
            if not self.hasher.verify(data.current_password.get_secret_value(), user.password_hash):
                raise AppError(401, "Invalid credentials.")
            if self.hasher.verify(data.new_password.get_secret_value(), user.password_hash):
                raise AppError(400, "Choose a different password.")
            user.password_hash = self.hasher.hash(data.new_password.get_secret_value())
            user.password_changed_at = utcnow()
            user.auth_version += 1
            logger.info("password_changed user_id=%s", user.id)
        return {"message": "Password changed. Sign in again."}

    def logout_all(self, principal: Principal) -> dict:
        with self.db.session() as session:
            user = self.current(session, principal)
            user.auth_version += 1
            logger.info("logout_all user_id=%s", user.id)
        return {"message": "All sessions revoked."}

    def set_confluence_token(self, data: ConfluenceCredential, principal: Principal) -> dict:
        with self.db.session() as session:
            user = self.current(session, principal)
            user.confluence_token = self.cipher.encrypt(
                data.token.get_secret_value().encode()
            ).decode()
            logger.info("confluence_credential_changed user_id=%s", user.id)
        return {"message": "Personal Confluence credential saved."}

    def confluence_token(self, principal: Principal) -> str:
        with self.db.session() as session:
            user = self.current(session, principal)
            if not user.confluence_token:
                raise AppError(409, "Configure your personal Confluence token before searching.")
            try:
                return self.cipher.decrypt(user.confluence_token.encode()).decode()
            except InvalidToken as exc:
                logger.error("credential_decryption_failed user_id=%s", user.id)
                raise AppError(503, "Confluence credential cannot be decrypted.") from exc
