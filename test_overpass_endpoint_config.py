"""
test_overpass_endpoint_config.py
--------------------------------
Tests for where an Overpass query actually goes: the local instance configured
by OVERPASS_URL, or the public mirrors.

Why these matter: the fallback from a dead local instance to the public mirrors
is deliberately silent, so that forgetting to start Docker costs latency rather
than taking the draw-a-boundary feature down. That silence is also how a demo
ends up back on the 44s/169s/502 public API without anyone noticing. These tests
pin the routing rules so the fallback stays a safety net rather than becoming
the default path by accident.

Runs fully offline — every endpoint is stubbed, nothing touches the network.

Run with:  python test_overpass_endpoint_config.py
"""

import importlib
import os
import sys

sys.path.insert(0, ".")

from app.core import overpass_network as ON

failures = []

PUBLIC_HOST = "overpass-api.de"
LOCAL_URL = "http://localhost:12345/api/interpreter"

# A minimal but real-shaped Overpass reply: two nodes joined by one residential
# way. Enough for build_graph to produce an actual edge.
GOOD = {
    "elements": [
        {"type": "node", "id": 1, "lat": 28.63, "lon": 77.21},
        {"type": "node", "id": 2, "lat": 28.64, "lon": 77.22},
        {"type": "way", "id": 10, "nodes": [1, 2], "tags": {"highway": "residential"}},
    ]
}
EMPTY = {"elements": []}


def check(label, got, expected):
    ok = got == expected
    print(f"{'PASS' if ok else 'FAIL'}  {label:52s} expected={expected!r:24s} got={got!r}")
    if not ok:
        failures.append(label)


def set_env(**env):
    """Reset every Overpass variable, then apply the given ones."""
    for key in ("OVERPASS_URL", "OVERPASS_PUBLIC_FALLBACK", "OVERPASS_LOCAL_TIMEOUT"):
        os.environ.pop(key, None)
    os.environ.update({k: v for k, v in env.items() if v is not None})


class Recorder:
    """Stands in for ask_endpoint, recording who was asked and replying to script."""

    def __init__(self, replies):
        # replies: {host_substring: payload | Exception}
        self.replies = replies
        self.asked = []

    def __call__(self, url, body, socket_timeout):
        self.asked.append(url)
        for needle, reply in self.replies.items():
            if needle in url:
                if isinstance(reply, Exception):
                    raise reply
                return url, reply, 0.01
        raise ConnectionRefusedError(f"no stub for {url}")

    @property
    def hit_public(self):
        return any("overpass-api.de" in u or "kumi" in u or "osm.ch" in u
                   for u in self.asked)

    @property
    def hit_local(self):
        return any("localhost" in u for u in self.asked)


class _FakeResp:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def run(recorder, **kwargs):
    """fetch_overpass against a stubbed network. attempts=1 to avoid retry sleeps."""
    original = ON.ask_endpoint
    ON.ask_endpoint = recorder
    try:
        return ON.fetch_overpass(28.615, 77.195, 28.655, 77.245, attempts=1, **kwargs)
    finally:
        ON.ask_endpoint = original


def check_readiness(label, fake_text=None, fake_exc=None, expected_ready=True):
    original = ON.urllib.request.urlopen

    def _fake(req, timeout=0):
        if fake_exc is not None:
            raise fake_exc
        return _FakeResp(fake_text or "")

    ON.urllib.request.urlopen = _fake
    try:
        ready, detail = ON.local_overpass_ready()
    finally:
        ON.urllib.request.urlopen = original

    check(label + " ready flag", ready, expected_ready)
    check(label + " detail", detail, None if expected_ready else ON.OVERPASS_LOADING_MESSAGE)


def health_payload(ready, detail):
    os.environ["JWT_SECRET"] = "test-secret"
    if "app.main" in sys.modules:
        del sys.modules["app.main"]
    main = importlib.import_module("app.main")
    from fastapi.testclient import TestClient

    original = main.local_overpass_ready
    main.local_overpass_ready = lambda: (ready, detail)
    try:
        return TestClient(main.app).get("/api/health").json()
    finally:
        main.local_overpass_ready = original


# ---------------------------------------------------------------------------
# Config readers
# ---------------------------------------------------------------------------
print("\nConfig readers")

set_env()
check("OVERPASS_URL unset -> None", ON.local_overpass_url(), None)
check("public fallback defaults on", ON.public_fallback_enabled(), True)
check("local timeout defaults to 20", ON.local_timeout(), 20)

set_env(OVERPASS_URL="  " + LOCAL_URL + "  ")
check("OVERPASS_URL is stripped", ON.local_overpass_url(), LOCAL_URL)

set_env(OVERPASS_URL="   ")
check("whitespace-only OVERPASS_URL -> None", ON.local_overpass_url(), None)

for value in ("off", "0", "false", "no", "OFF", "False"):
    set_env(OVERPASS_PUBLIC_FALLBACK=value)
    check(f"fallback disabled by {value!r}", ON.public_fallback_enabled(), False)

set_env(OVERPASS_PUBLIC_FALLBACK="on")
check("fallback enabled by 'on'", ON.public_fallback_enabled(), True)

set_env(OVERPASS_LOCAL_TIMEOUT="5")
check("local timeout parses int", ON.local_timeout(), 5)
set_env(OVERPASS_LOCAL_TIMEOUT="garbage")
check("garbage timeout -> default", ON.local_timeout(), 20)
set_env(OVERPASS_LOCAL_TIMEOUT="0")
check("zero timeout -> default", ON.local_timeout(), 20)
set_env(OVERPASS_LOCAL_TIMEOUT="-3")
check("negative timeout -> default", ON.local_timeout(), 20)


# ---------------------------------------------------------------------------
# Local readiness
# ---------------------------------------------------------------------------
print("\nLocal readiness")

set_env()
check("no local configured -> ready", ON.local_overpass_ready(), (True, None))

set_env(OVERPASS_URL=LOCAL_URL)
check_readiness(
    "status with available slots is ready",
    fake_text="Rate limit: 2\n2 slots available now.\nCurrently running queries:\n",
    expected_ready=True,
)
check_readiness(
    "status without ready markers is loading",
    fake_text="Database not available yet\n",
    expected_ready=False,
)
check_readiness(
    "status endpoint connection error is loading",
    fake_exc=ConnectionRefusedError("refused"),
    expected_ready=False,
)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
print("\nRouting")

# The happy path: local answers, and the public API is never touched. This is
# the whole point of the change — if this regresses, the demo is quietly back on
# a rate-limited shared endpoint.
set_env(OVERPASS_URL=LOCAL_URL)
rec = Recorder({"localhost": GOOD, PUBLIC_HOST: GOOD})
payload = run(rec)
check("local answers -> local used", payload is GOOD, True)
check("local answers -> public untouched", rec.hit_public, False)

# Local configured but refusing connections (Docker not started).
set_env(OVERPASS_URL=LOCAL_URL)
rec = Recorder({"localhost": ConnectionRefusedError("refused"), PUBLIC_HOST: GOOD})
payload = run(rec)
check("local down -> falls back to public", payload is GOOD, True)
check("local down -> local was tried first", rec.asked[0], LOCAL_URL)

# An empty local reply means "box outside the imported extract", not "no roads
# here" — a regional .osm.pbf makes those two indistinguishable, so it must fall
# through rather than be reported as a real answer about the box.
set_env(OVERPASS_URL=LOCAL_URL)
rec = Recorder({"localhost": EMPTY, PUBLIC_HOST: GOOD})
payload = run(rec)
check("empty local reply -> falls back to public", payload is GOOD, True)

# Strict mode: a failing local instance is an error, not a silent downgrade.
set_env(OVERPASS_URL=LOCAL_URL, OVERPASS_PUBLIC_FALLBACK="off")
rec = Recorder({"localhost": ConnectionRefusedError("refused"), PUBLIC_HOST: GOOD})
try:
    run(rec)
    check("fallback off -> raises", False, True)
except RuntimeError:
    check("fallback off -> raises", True, True)
except Exception as exc:
    check("fallback off -> raises RuntimeError", type(exc).__name__, "RuntimeError")
check("fallback off -> public untouched", rec.hit_public, False)

# No local configured: unchanged behaviour, straight to the mirrors.
set_env()
rec = Recorder({PUBLIC_HOST: GOOD, "kumi": GOOD, "osm.ch": GOOD})
payload = run(rec)
check("no OVERPASS_URL -> public mirrors", payload is GOOD, True)
check("no OVERPASS_URL -> local untouched", rec.hit_local, False)

# An explicit mirrors= argument is a deliberate override (build_osm_cache.py's
# --overpass-url), so it must win over the environment rather than be ignored.
set_env(OVERPASS_URL=LOCAL_URL)
rec = Recorder({"example.org": GOOD, "localhost": GOOD})
payload = run(rec, mirrors=["https://example.org/api/interpreter"])
check("explicit mirrors= bypasses env local", rec.hit_local, False)
check("explicit mirrors= is used", rec.asked, ["https://example.org/api/interpreter"])

# Every mirror agreeing the box is empty is a real answer about the box.
set_env()
rec = Recorder({PUBLIC_HOST: EMPTY, "kumi": EMPTY, "osm.ch": EMPTY})
payload = run(rec)
check("all mirrors empty -> empty returned", payload.get("elements"), [])


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------
print("\nQuery")

q = ON.build_query(28.615, 77.195, 28.655, 77.245, timeout=120)
check("query carries the bbox", "(28.615,77.195,28.655,77.245)" in q, True)
check("query filters to drivable ways", 'way["highway"' in q, True)
check("query asks for json", "[out:json]" in q, True)
check("query includes residential", "residential" in q, True)

# The local instance's OVERPASS_MAX_TIMEOUT must cover this or big boxes are
# refused outright rather than answered slowly.
check("query timeout is 120", "[timeout:120]" in q, True)


# ---------------------------------------------------------------------------
# /api/health payload
# ---------------------------------------------------------------------------
print("\nHealth endpoint")

healthy = health_payload(True, None)
check("health reports ok when ready", healthy.get("status"), "ok")
check("health includes osm ready=true", healthy.get("osm", {}).get("ready"), True)
check("health includes osm status=ready", healthy.get("osm", {}).get("status"), "ready")

loading = health_payload(False, ON.OVERPASS_LOADING_MESSAGE)
check("health reports degraded while loading", loading.get("status"), "degraded")
check("health includes osm ready=false", loading.get("osm", {}).get("ready"), False)
check("health includes osm status=loading", loading.get("osm", {}).get("status"), "loading")


print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("All Overpass endpoint-routing tests passed.")
sys.exit(0)
