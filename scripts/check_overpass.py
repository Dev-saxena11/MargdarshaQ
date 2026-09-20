"""
check_overpass.py
-----------------
Preflight for the Overpass dependency. Answers one question: if a judge draws a
box on the map right now, where does that request go and how long does it take?

Why this exists
===============
The failure this guards against is silent. `OVERPASS_PUBLIC_FALLBACK` defaults
to on, so an app pointed at a local instance that never started still works — it
just quietly goes back to the public mirrors and the 25s waits they come with.
Everything looks fine in testing and the demo is slow for reasons nobody can
explain on the spot. This script makes that fallback visible while there is
still time to fix it.

Run it the night before the demo, and again on the morning of it.

Usage
=====
    python scripts/check_overpass.py
    python scripts/check_overpass.py --compare        # also time the public mirrors
    python scripts/check_overpass.py --south 19.06 --north 19.10 \
        --west 72.83 --east 72.88                     # check a specific box

Exit codes
==========
    0  the configured endpoint answered
    1  it did not, or nothing local is configured
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:                                    # the app reads .env the same way
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:                     # python-dotenv is optional here
    pass

from app.core.overpass_network import (          # noqa: E402
    OVERPASS_MIRRORS, ask_endpoint, build_graph, build_query,
    local_overpass_url, local_timeout, public_fallback_enabled,
)


def probe(url: str, body: bytes, timeout: int) -> tuple:
    """(ok, seconds, detail) for one endpoint."""
    started = time.time()
    try:
        _, payload, took = ask_endpoint(url, body, timeout)
    except Exception as exc:
        return False, time.time() - started, f"{type(exc).__name__}: {exc}"

    elements = payload.get("elements") or []
    if not elements:
        return False, took, "answered but returned no elements"

    # Parsing here is not ceremony: an endpoint can return well-formed JSON that
    # still yields no routable graph, and finding that out now is the entire
    # point of a preflight.
    coords, adj = build_graph(payload)
    edges = sum(len(v) for v in adj.values())
    return True, took, f"{len(elements)} elements -> {len(coords)} nodes, {edges} directed edges"


def main() -> int:
    ap = argparse.ArgumentParser(description="Check which Overpass endpoint the app will use.")
    # Connaught Place, matching build_osm_cache.py's default box — small enough
    # to be quick, dense enough that an empty answer is definitely a fault.
    ap.add_argument("--south", type=float, default=28.615)
    ap.add_argument("--north", type=float, default=28.655)
    ap.add_argument("--west", type=float, default=77.195)
    ap.add_argument("--east", type=float, default=77.245)
    ap.add_argument("--compare", action="store_true",
                    help="also time the public mirrors, for a before/after number")
    args = ap.parse_args()

    body = build_query(args.south, args.west, args.north, args.east).encode("utf-8")
    local = local_overpass_url()

    print(f"Box: ({args.south},{args.west}) -> ({args.north},{args.east})\n")

    if not local:
        print("OVERPASS_URL is not set.")
        print("  The app will use the public mirrors — shared, rate-limited, and")
        print("  measured at 44s / 169s / 502 for the same box inside one hour.")
        print("  For a demo, start the local instance and set OVERPASS_URL:")
        print("    docker compose -f docker-compose.overpass.yml up -d")
        print("    OVERPASS_URL=http://localhost:12345/api/interpreter")
        if args.compare:
            print()
            check_public(body)
        return 1

    print(f"OVERPASS_URL = {local}")
    print(f"  public fallback: {'on' if public_fallback_enabled() else 'off'}")
    print(f"  local timeout  : {local_timeout()}s\n")

    ok, took, detail = probe(local, body, local_timeout())
    if ok:
        print(f"  OK  local instance answered in {took:.2f}s — {detail}")
        if took > 5:
            print("      Slower than a healthy local instance should be (expect <1s).")
            print("      Check the container has finished importing and has RAM to spare.")
    else:
        print(f"  FAIL  local instance did not answer after {took:.2f}s — {detail}")
        print()
        print("  Things to check, in order:")
        print("    docker compose -f docker-compose.overpass.yml ps")
        print("    docker compose -f docker-compose.overpass.yml logs --tail 50")
        print("  An import still in progress looks exactly like this; so does a box")
        print("  outside the imported extract, which returns zero elements.")
        if public_fallback_enabled():
            print()
            print("  The app will still work — it falls back to the public mirrors —")
            print("  but at public-API speed, which is the thing this was meant to fix.")

    if args.compare:
        print()
        check_public(body)

    return 0 if ok else 1


def check_public(body: bytes) -> None:
    """Time each public mirror in turn, for a number to put next to the local one."""
    print("Public mirrors (for comparison):")
    for url in OVERPASS_MIRRORS:
        ok, took, detail = probe(url, body, 30)
        status = "OK  " if ok else "FAIL"
        print(f"  {status} {url.split('/')[2]:<28} {took:6.2f}s  {detail}")


if __name__ == "__main__":
    sys.exit(main())
