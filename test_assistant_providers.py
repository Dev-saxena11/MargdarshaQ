"""
test_assistant_providers.py
---------------------------
Tests for the AI Assistant's LLM provider layer.

Everything here runs OFFLINE — no API key required and no network calls. Live
provider behaviour is exercised separately (see DEPLOYMENT.md); these checks
cover the logic that must hold regardless of which provider is configured.

Run with:  python test_assistant_providers.py
"""

import os
import sys

# These checks assert the offline path: with no LLM configured and nothing
# solved yet, the assistant still answers, locally, and says there is nothing to
# report. "No API key required" was true; what the file actually needed was no
# API key and no database *present*, which is a different thing. Run on a
# machine whose .env supplies either, three checks failed -- the assistant took
# the live-provider path, and read rows somebody else had stored days ago.
#
# Set empty rather than deleted: the app calls load_dotenv(), which leaves an
# existing variable alone but will happily set an absent one from .env.
os.environ["DATABASE_URL"] = ""
for _provider_key in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
    os.environ[_provider_key] = ""
os.environ.setdefault("JWT_SECRET", "test-only-secret-not-used-anywhere-real")

from app.core.llm_providers import (
    OpenRouterProvider, GeminiProvider, OpenAIProvider,
    available_providers, get_provider,
)
from app.core.assistant import AIAssistantExplainer
from app.models.schemas import AssistantChatRequest, AssistantContext

failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        failures.append(label)


print("=" * 70)
print("1. Free-model policy is enforced")
print("=" * 70)

check("'openrouter/free' accepted", OpenRouterProvider.is_free_model("openrouter/free"))
check("':free' suffix accepted", OpenRouterProvider.is_free_model("google/gemma-4-31b-it:free"))
check("paid model rejected", not OpenRouterProvider.is_free_model("openai/gpt-4o"))
check("empty model rejected", not OpenRouterProvider.is_free_model(""))

try:
    OpenRouterProvider(api_key="test", model="openai/gpt-4o")
    check("constructing with a paid model raises", False, "no exception raised")
except ValueError:
    check("constructing with a paid model raises", True)

try:
    p = OpenRouterProvider(api_key="test", model="openrouter/free")
    check("constructing with a free model works", p.model == "openrouter/free")
except ValueError as e:
    check("constructing with a free model works", False, str(e))

print()
print("=" * 70)
print("2. Provider selection degrades safely")
print("=" * 70)

check("no credentials -> no provider",
      get_provider() is None or get_provider().is_configured(),
      f"got {get_provider()}")
check("unconfigured providers report is_configured() False",
      not GeminiProvider(api_key=None).is_configured()
      and not OpenAIProvider(api_key=None).is_configured()
      and not OpenRouterProvider(api_key=None).is_configured())
check("a configured provider is discoverable",
      OpenRouterProvider(api_key="test").is_configured())

print()
print("=" * 70)
print("3. Assistant still works with no provider configured")
print("=" * 70)

engine = AIAssistantExplainer()

resp = engine.chat(AssistantChatRequest(message="explain the route savings",
                                        context=AssistantContext()))
check("no-run question answered locally", "No optimization run yet" in resp.reply)
check("no fabricated percentage appears",
      not any(tok in resp.reply for tok in ("30.0%", "77.1%", "254.6")))

ctx = AssistantContext(scenario_name="Rush Hour", time_saved_pct=22.4,
                       baseline_time=228.4, optimized_time=177.2,
                       optimized_feasible=True)
resp2 = engine.chat(AssistantChatRequest(message="explain the route savings", context=ctx))
check("measured figures are reported verbatim",
      "22.4" in resp2.reply and "228.4" in resp2.reply)

resp3 = engine.chat(AssistantChatRequest(message="anything", chip="why_qpso",
                                         context=AssistantContext()))
check("preset chips answered without an LLM call", "QPSO" in resp3.reply)

print()
print("=" * 70)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("ALL CHECKS PASSED")
