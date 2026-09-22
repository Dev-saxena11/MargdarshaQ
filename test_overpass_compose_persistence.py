"""
test_overpass_compose_persistence.py
------------------------------------
Guardrails for demo reliability: the Overpass container must restart after host
reboots and keep using a persistent named volume rather than an ephemeral one.

Run with: python test_overpass_compose_persistence.py
"""

from pathlib import Path
import re
import sys

text = Path("docker-compose.overpass.yml").read_text(encoding="utf-8")
failures = []


def check(label, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        failures.append(label)


check(
    "restart policy is unless-stopped",
    bool(re.search(r"^\s*restart:\s*unless-stopped\s*$", text, re.MULTILINE)),
)
check(
    "service mounts named overpass_db volume to /db",
    bool(re.search(r"^\s*-\s*overpass_db:/db\s*$", text, re.MULTILINE)),
)
check(
    "overpass_db uses stable explicit volume name",
    bool(re.search(r"^\s*name:\s*sih26137-overpass-db\s*$", text, re.MULTILINE)),
)
check(
    "overpass_db volume is external (not lifecycle-coupled to compose down -v)",
    bool(re.search(r"^\s*external:\s*true\s*$", text, re.MULTILINE)),
)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)

print("\nOverpass compose persistence checks passed.")
sys.exit(0)
