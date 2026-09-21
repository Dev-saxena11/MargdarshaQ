"""
test_auth_jwt.py
-----------------
Token verification for the dashboard's login.

Supabase signs access tokens either with the legacy shared secret (HS256) or
with an asymmetric signing key published at the project's JWKS endpoint
(ES256). The deployed project uses ES256, and verifying those against a shared
secret refused every login with `401 {"detail":"Invalid token"}`. These tests
pin both paths so that cannot regress into a message nobody can act on.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api import routes


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


@pytest.fixture(autouse=True)
def _clear_jwk_cache():
    """The signing-key client is cached for the process, so tests must not share one."""
    routes._jwk_client.cache_clear()
    yield
    routes._jwk_client.cache_clear()


# ---------------------------------------------------------------------------
# ES256 — asymmetric signing keys, fetched from JWKS
# ---------------------------------------------------------------------------

@pytest.fixture
def jwks_project(monkeypatch):
    """A stand-in Supabase project that signs ES256 and serves its public key."""
    key = ec.generate_private_key(ec.SECP256R1())
    kid = "test-signing-key"
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(key.public_key()))
    jwk.update({"kid": kid, "use": "sig", "alg": "ES256"})
    body = json.dumps({"keys": [jwk]}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/auth/v1/.well-known/jwks.json", self.path
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(routes, "SUPABASE_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        yield lambda claims: jwt.encode(claims, key, algorithm="ES256", headers={"kid": kid})
    finally:
        server.shutdown()


def test_es256_token_is_accepted(jwks_project):
    assert routes.get_current_user(_creds(jwks_project({"sub": "user-1"}))) == "user-1"


def test_es256_token_signed_by_another_key_is_refused(jwks_project):
    other = ec.generate_private_key(ec.SECP256R1())
    forged = jwt.encode({"sub": "user-1"}, other, algorithm="ES256",
                        headers={"kid": "test-signing-key"})
    with pytest.raises(HTTPException) as exc:
        routes.get_current_user(_creds(forged))
    assert exc.value.status_code == 401


def test_token_without_a_subject_is_refused(jwks_project):
    """Every such caller would otherwise land in one shared workspace."""
    with pytest.raises(HTTPException) as exc:
        routes.get_current_user(_creds(jwks_project({"aud": "authenticated"})))
    assert exc.value.status_code == 401


def test_unreachable_jwks_is_not_reported_as_a_bad_token(monkeypatch, jwks_project):
    """
    The visitor cannot fix an auth service that is down, so sending them to the
    login form would be a dead end. 503 says retry instead.
    """
    token = jwks_project({"sub": "user-1"})
    routes._jwk_client.cache_clear()
    monkeypatch.setattr(routes, "SUPABASE_URL", "http://127.0.0.1:1")  # nothing listening
    with pytest.raises(HTTPException) as exc:
        routes.get_current_user(_creds(token))
    assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# HS256 — the legacy shared secret, still in use on older projects
# ---------------------------------------------------------------------------

def test_hs256_token_is_still_accepted(monkeypatch):
    monkeypatch.setattr(routes, "SUPABASE_JWT_SECRET", "legacy-secret")
    token = jwt.encode({"sub": "user-2"}, "legacy-secret", algorithm="HS256")
    assert routes.get_current_user(_creds(token)) == "user-2"


def test_hs256_token_with_the_wrong_secret_is_refused(monkeypatch):
    monkeypatch.setattr(routes, "SUPABASE_JWT_SECRET", "legacy-secret")
    token = jwt.encode({"sub": "user-2"}, "not-the-secret", algorithm="HS256")
    with pytest.raises(HTTPException) as exc:
        routes.get_current_user(_creds(token))
    assert exc.value.status_code == 401


def test_expired_token_names_the_session_rather_than_the_token(monkeypatch):
    monkeypatch.setattr(routes, "SUPABASE_JWT_SECRET", "legacy-secret")
    token = jwt.encode({"sub": "user-2", "exp": 1}, "legacy-secret", algorithm="HS256")
    with pytest.raises(HTTPException) as exc:
        routes.get_current_user(_creds(token))
    assert exc.value.status_code == 401
    assert "expired" in exc.value.detail.lower()


def test_garbage_is_refused_rather_than_crashing():
    with pytest.raises(HTTPException) as exc:
        routes.get_current_user(_creds("not-a-jwt"))
    assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# The "none" algorithm, which carries no signature at all
# ---------------------------------------------------------------------------

def test_unsigned_token_is_refused(monkeypatch):
    """
    An alg of "none" takes neither branch of the key lookup by accident: it does
    not start with HS, and it is not a registered algorithm, so it must be
    refused rather than waved through as "no signature to check".
    """
    monkeypatch.setattr(routes, "SUPABASE_JWT_SECRET", "legacy-secret")
    forged = jwt.encode({"sub": "admin"}, key=None, algorithm="none")
    with pytest.raises(HTTPException) as exc:
        routes.get_current_user(_creds(forged))
    assert exc.value.status_code == 401
