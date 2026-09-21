"""
test_auth_tokens.py
-------------------
Tests for the Supabase bearer-token check that guards every network and VRP
endpoint (app/api/routes.py:get_current_user).

Why these matter: Supabase signs access tokens one of two ways, and which one
is not the backend's choice. Projects on the newer `sb_publishable_…` API keys
sign with an asymmetric key (ES256) and publish only the public half as a JWKS;
older projects use the shared HS256 "JWT secret". A backend that verifies HS256
alone rejects every token such a project issues, and the signature failure
surfaces as a flat 401 "Invalid token" — which reads to everyone looking at it
like a forged or stale token rather than a server that cannot check the real
one. The whole dashboard is unusable in that state.

Runs fully offline; no deployed backend and no Supabase project needed.

Run with:  python test_auth_tokens.py
"""

import importlib
import os
import sys
import time

sys.path.insert(0, ".")

import jwt
from cryptography.hazmat.primitives.asymmetric import ec

HS_SECRET = "test-jwt-secret-not-a-real-one"
SUPABASE_URL = "https://example-project.supabase.co"

failures = []


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


# --- A project signing with an asymmetric key, as Supabase now does ---------
_ec_key = ec.generate_private_key(ec.SECP256R1())
_public_jwk = jwt.algorithms.ECAlgorithm.to_jwk(_ec_key.public_key(), as_dict=True)
_public_jwk.update({"kid": "test-key", "use": "sig", "alg": "ES256"})


class _FakeJWKSClient:
    """Stands in for PyJWKClient so no network call is made."""

    def __init__(self, *_args, **_kwargs):
        pass

    def get_signing_key_from_jwt(self, _token):
        return jwt.PyJWK(_public_jwk, algorithm="ES256")


def load_routes():
    """Reimport the API under a known Supabase configuration."""
    os.environ["SUPABASE_JWT_SECRET"] = HS_SECRET
    os.environ["SUPABASE_URL"] = SUPABASE_URL
    for mod in [m for m in list(sys.modules) if m.startswith("app.")]:
        del sys.modules[mod]
    routes = importlib.import_module("app.api.routes")
    routes.PyJWKClient = _FakeJWKSClient
    routes._jwks_client = None
    return routes


def claims(sub="user-1234", **extra):
    payload = {"sub": sub, "aud": "authenticated", "exp": int(time.time()) + 3600}
    payload.update(extra)
    if sub is None:
        payload.pop("sub")
    return payload


def es256(**kw):
    return jwt.encode(claims(**kw), _ec_key, algorithm="ES256",
                      headers={"kid": "test-key"})


def hs256(secret=HS_SECRET, **kw):
    return jwt.encode(claims(**kw), secret, algorithm="HS256")


def user_for(routes, token):
    """Run the dependency directly and return the user id or the HTTP status."""
    from fastapi import HTTPException
    from fastapi.security import HTTPAuthorizationCredentials

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    try:
        return routes.get_current_user(creds)
    except HTTPException as e:
        return e.status_code


routes = load_routes()

# The regression this file exists for.
check("an ES256 token from a modern Supabase project is accepted",
      user_for(routes, es256()) == "user-1234",
      f"got {user_for(routes, es256())!r}")

check("an HS256 token from a legacy project is still accepted",
      user_for(routes, hs256()) == "user-1234",
      f"got {user_for(routes, hs256())!r}")

check("the subject is returned as the user id, not the whole token",
      user_for(routes, es256(sub="abc-def")) == "abc-def")

# Forgery and staleness must still be refused.
check("an ES256 token signed with someone else's key is refused",
      user_for(routes, jwt.encode(claims(), ec.generate_private_key(ec.SECP256R1()),
                                  algorithm="ES256", headers={"kid": "test-key"})) == 401)
check("an HS256 token signed with the wrong secret is refused",
      user_for(routes, hs256(secret="wrong-secret")) == 401)
check("an expired token is refused",
      user_for(routes, es256(exp=int(time.time()) - 60)) == 401)
check("a token that is not a token at all is refused",
      user_for(routes, "not-a-jwt") == 401)
check("an unsigned token is refused",
      user_for(routes, jwt.encode(claims(), key="", algorithm="none")) == 401)

# A token with no subject would pool every such caller's data under one empty
# key, which is the opposite of the isolation this check exists to provide.
check("a token carrying no subject is refused",
      user_for(routes, es256(sub=None)) == 401)


# --- The endpoints actually carry the dependency ---------------------------
from fastapi.testclient import TestClient

app = importlib.import_module("app.main").app
client = TestClient(app)

check("an unauthenticated call to a guarded endpoint is refused",
      client.post("/api/network/generate", json={"n_nodes": 10}).status_code in (401, 403))

check("/api/health needs no token",
      client.get("/api/health").status_code == 200)


# --- Every store call site matches the user-keyed signature ----------------
# #93 changed store.put_network/get_network/put_vrp to take user_id first; a
# call site left on the old signature raises TypeError at request time, which
# the caller sees as a bare 500 with no hint of what went wrong.
import inspect
from app.core import store

for fn, first in [(store.put_network, "user_id"), (store.get_network, "user_id"),
                  (store.put_vrp, "user_id"), (store.get_vrp, "user_id"),
                  (store.is_geo_network, "user_id"), (store.get_vrp_network_id, "user_id")]:
    params = list(inspect.signature(fn).parameters)
    check(f"store.{fn.__name__} takes the user id first", params[0] == first,
          f"got {params}")

import re
src = open("app/core/assistant.py", encoding="utf-8").read()
for call in re.findall(r"store\.(put_network|get_network|put_vrp|get_vrp)\((.*?)\)", src):
    name, args = call
    check(f"assistant's store.{name}(…) passes a user id",
          args.split(",")[0].strip() in ("user_id", "_ASSISTANT_STORE_USER"),
          f"first argument is {args.split(',')[0].strip()!r}")


# --- The deployed environment can actually do ES256 -------------------------
# PyJWT registers ES256 only when `cryptography` is installed, which plain
# `pyjwt` does not pull in — requirements.txt asks for `pyjwt[crypto]`. Without
# it every real token is refused again, on a machine where nothing else looks
# wrong.
check("ES256 is registered in this environment (pyjwt[crypto] installed)",
      "ES256" in jwt.algorithms.get_default_algorithms())

check("requirements.txt asks for the crypto extra",
      "pyjwt[crypto]" in open("requirements.txt", encoding="utf-8").read())


print()
if failures:
    print(f"{len(failures)} check(s) failed: " + ", ".join(failures))
    sys.exit(1)
print("All auth token checks passed.")
