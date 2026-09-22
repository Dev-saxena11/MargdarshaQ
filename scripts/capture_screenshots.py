"""
capture_screenshots.py
-----------------------
Regenerate the dashboard screenshots in the README.

Drives the real dashboard in a headless browser against a local backend, so the
images are of the running app rather than a mock-up, and go stale visibly when
the UI changes instead of quietly misrepresenting it.

Prerequisites:

    pip install playwright
    python -m playwright install chromium

Then, in two other terminals:

    uvicorn app.main:app --port 8100          # needs JWT_SECRET set
    python -m http.server 8101 -d frontend

Then:

    python scripts/capture_screenshots.py

Note on the login wall: the dashboard reads its bearer token out of
localStorage, so this seeds one signed with the same JWT_SECRET the local API
was started with rather than registering an account. Point --secret at whatever
that was.
"""

from __future__ import annotations

import argparse
import os
import sys

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "images"
)


def capture(web: str, api: str, secret: str, out_dir: str) -> int:
    try:
        import jwt
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        print(f"missing dependency: {exc}. See this file's docstring.", file=sys.stderr)
        return 2

    os.makedirs(out_dir, exist_ok=True)
    token = jwt.encode({"sub": "screenshot-user"}, secret, algorithm="HS256")
    errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1600, "height": 1050}, device_scale_factor=1
        )
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(web, wait_until="domcontentloaded")
        page.evaluate("t => localStorage.setItem('margdarshaq_token', t)", token)
        page.goto(web, wait_until="networkidle")
        page.wait_for_timeout(1500)

        # The opening calls fire before this runs, so the panels that depend on
        # the API are re-run once it points somewhere real. Without this the
        # header reads OFFLINE and the comparison table reports that it could
        # not reach the backend -- in a picture meant to show it working.
        page.evaluate("u => { document.getElementById('apiBase').value = u; }", api)
        page.evaluate("() => checkHealth()")
        page.evaluate("() => loadBenchmarkMatrix()")
        for _ in range(40):
            online = page.evaluate(
                "() => (document.getElementById('connLabel')||{}).textContent || ''"
            )
            failing = page.evaluate(
                "() => document.body.innerText.includes('Benchmark not available')"
            )
            if "ONLINE" in online and not failing:
                break
            page.wait_for_timeout(1000)
        page.wait_for_timeout(1500)
        page.screenshot(path=os.path.join(out_dir, "01-executive-overview.png"))
        print("captured 01-executive-overview.png")

        page.evaluate("() => switchView('control')")
        page.wait_for_timeout(3000)
        page.screenshot(path=os.path.join(out_dir, "02-control-room.png"))
        print("captured 02-control-room.png")

        # The one-click scenario, run for real: load a district, place stops,
        # build the instance, race all five algorithms.
        page.evaluate("() => runDemoScenario()")
        for _ in range(120):
            # `state` is a script-scope binding, not a property of window.
            if page.evaluate("() => !!(state && state.benchmarkResults)"):
                break
            page.wait_for_timeout(1000)
        page.wait_for_timeout(2500)
        page.screenshot(path=os.path.join(out_dir, "03-solved-routes.png"))
        print("captured 03-solved-routes.png")

        status = page.evaluate(
            "() => (document.getElementById('demoScenarioStatus')||{}).textContent"
        )
        print("scenario:", (status or "").strip()[:160])
        browser.close()

    if errors:
        print("page errors:", errors[:5], file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web", default="http://127.0.0.1:8101/dashboard.html")
    parser.add_argument("--api", default="http://127.0.0.1:8100")
    parser.add_argument(
        "--secret", default=os.getenv("JWT_SECRET", ""),
        help="the JWT_SECRET the local API was started with",
    )
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    if not args.secret:
        print(
            "No signing secret. Pass --secret with the same value the local API "
            "was started with, or set JWT_SECRET in this shell.",
            file=sys.stderr,
        )
        return 2
    return capture(args.web, args.api, args.secret, args.out)


if __name__ == "__main__":
    sys.exit(main())
