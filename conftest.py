"""
conftest.py
------------
Session-wide setup for the pytest suite.

The environment is pinned here rather than left to whatever the machine happens
to have, because two different tests had already been quietly decided by it.
See `_pin_environment` below.
"""

import os

# ---------------------------------------------------------------------------
# Environment, pinned before anything under app/ is imported
# ---------------------------------------------------------------------------
# This runs at module scope on purpose. A fixture would be too late: several
# modules read these at import time, and collection imports the test modules
# (and through them app.api.routes) before any fixture has run.
#
#   JWT_SECRET    routes.py refuses to import without it, which is deliberate --
#                 a deployment that forgets it would otherwise sign tokens with
#                 a string committed to this repo. Tests still need *a* value,
#                 and it must not be a real one.
#
#   provider keys the assistant and RAG tests assert the offline path: that with
#                 no LLM configured the answer still comes back, grounded and
#                 local. A developer with OPENROUTER_API_KEY exported for
#                 ordinary use sends those tests down the live-provider path
#                 instead, and two of them fail on a machine where nothing is
#                 wrong. Cleared so the suite tests the same thing everywhere.
#                 A test that wants the configured path should set the key
#                 itself, via monkeypatch, and say so.
#
#   DATABASE_URL  store.py falls back to an in-memory store when this is unset,
#                 which is what the suite should run against. Left alone, a
#                 developer's .env points it at the live database: the tests
#                 then read production rows, and the assistant checks -- which
#                 assert that with nothing solved yet it says so -- fail,
#                 because something *had* been solved, by someone else, days
#                 ago. Cleared so the suite neither depends on nor writes to a
#                 real database.
#
# Each is set to an empty string rather than deleted. store.py calls
# load_dotenv(), which does not overwrite a variable that is already present but
# does happily set one that is absent -- so deleting these would hand .env the
# last word and undo the whole point of pinning them.
def _pin_environment():
    os.environ.setdefault("JWT_SECRET", "test-only-secret-not-used-anywhere-real")
    os.environ["DATABASE_URL"] = ""
    for provider_key in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        os.environ[provider_key] = ""


_pin_environment()


# ---------------------------------------------------------------------------
# Standalone scripts that must not be collected
# ---------------------------------------------------------------------------
# These are scripts, not pytest modules (see each one's docstring: "Run with:
# python <name>.py"). They call sys.exit() at module scope, which does not
# merely fail collection -- it aborts the whole pytest run with an
# INTERNALERROR, so every other test in the repo silently stops running too,
# and the output says only "no tests ran", naming neither the file nor the
# reason.
#
# test_overpass_compose_persistence.py is worth singling out: it aborts the run
# on SUCCESS as well, because it ends in sys.exit(0), and a SystemExit raised
# while pytest is importing a module stops collection whatever its code. So
# "all its checks passed" and "no tests collected in the whole repo" were the
# same run.
#
# If you add another, add it here as well.
collect_ignore = [
    "test_overpass_endpoint_config.py",
    "test_auth_tokens.py",
    "test_assistant_providers.py",
    "test_overpass_compose_persistence.py",
]
