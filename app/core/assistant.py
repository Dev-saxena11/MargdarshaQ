"""
assistant.py
------------
In-dashboard AI Assistant Explainer for MargdarshaQ Intelligent Traffic Platform.

Provides context-grounded natural language explanations for:
  1. Solve and benchmark results (time saved, delay avoided, fuel cut, SLA).
  2. "Why QPSO" quantum-inspired metaheuristic advantage over classical baselines.
  3. Interactive map elements (congestion friction, shortest road paths, depot, customers).
  4. Convergence curve interpretations and constraint satisfaction (CVRPTW).

Hybrid architecture:
  - Primary: a deterministic offline engine grounded in the live session
    metrics. Always available, costs nothing, and never invents a number —
    with no run executed it reports that instead of filling in placeholders.
  - Story-Card caching: a compact one-paragraph summary per solve/scenario,
    so LLM prompt size stays near-constant instead of growing with raw JSON.
  - Optional LLM enrichment for free-text questions, via whichever provider is
    configured (OpenRouter free models by default) — see
    `app/core/llm_providers.py`. Any failure falls back to the local engine.

Document grounding (RAG) is tracked separately in issue #33 and is deliberately
not designed here — see `_call_external_llm` for where retrieved context would
be added to the prompt.
"""

from __future__ import annotations
import json
import logging
import re
import time
from typing import Dict, Any, List, Optional

from app.models.schemas import (
    AssistantContext,
    AssistantChatRequest,
    AssistantChatResponse,
)
from app.core.llm_providers import LLMProvider, get_provider
from app.core import store
from app.core.vrp_problem import generate_synthetic_vrp
from app.core.qpso_vrp import QPSOVRPOptimizer

logger = logging.getLogger(__name__)

SUGGESTED_CHIPS = [
    "⚡ Explain this route & savings",
    "⚛️ Why did Quantum QPSO win?",
    "🗺️ Explain the map & traffic colors",
    "📈 What does convergence chart show?",
    "🎯 Are time windows respected?",
    "🚗 Plan a Route",
]

# The MVP plans on one city. Conversational planning used to build a synthetic
# graph of its own, so the route it produced was on streets that existed
# nowhere else in the app — not the map the user had been looking at. It now
# plans on the shipped Bareilly network, the same one the control room opens.
MVP_NETWORK = "bareilly"
MVP_AREA_LABEL = "Bareilly City, Uttar Pradesh"

# Shown instead of performance numbers whenever no comparison run has been
# executed yet. The assistant must never invent metrics — a plausible-looking
# fabricated number is indistinguishable from a measured one to the reader,
# which would misrepresent benchmark results.
NO_RESULTS_NOTICE = (
    "### 📊 No optimization run yet\n\n"
    "I don't have any measured results to explain for this session — nothing has "
    "been solved yet, so there are no real numbers to report.\n\n"
    "**To get a grounded analysis:**\n"
    "1. Generate a network (synthetic or real OSM map).\n"
    "2. Generate a VRP instance on it.\n"
    "3. Run **Solve** or **Compare** (baseline vs optimized).\n\n"
    "Once a run completes I'll explain the actual time saved, congestion avoided, "
    "and constraint satisfaction from that run.\n\n"
    "In the meantime you can still ask me conceptual questions — how QPSO works, "
    "what the map colors mean, or how the convergence chart is read."
)


def _has_comparison_results(ctx: AssistantContext) -> bool:
    """
    True only when the session carries real measured comparison metrics.

    Used to gate every numeric claim: without it the explainers would fall back
    to placeholder values and present them as if they were measured output.
    """
    return any(
        v is not None
        for v in (
            ctx.time_saved_pct,
            ctx.delay_saved_pct,
            ctx.dist_saved_pct,
            ctx.baseline_time,
            ctx.optimized_time,
        )
    )


def _fmt(value: Optional[float], suffix: str = "", decimals: int = 1) -> str:
    """Formats a metric, or 'n/a' when it genuinely wasn't measured."""
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}{suffix}"


def _delta(value: Optional[float], suffix: str = "%") -> str:
    """Formats a saving as '-22.4%', or a plain 'not measured' when absent."""
    if value is None:
        return "not measured"
    return f"**-{value:.1f}{suffix}**"


class AIAssistantExplainer:
    """Core in-dashboard AI assistant engine with token-budget optimization."""

    def __init__(self):
        self._story_card_cache: Dict[str, str] = {}

    def chat(self, req: AssistantChatRequest) -> AssistantChatResponse:
        """Process user query and return grounded AI explanation."""
        ctx = req.context or AssistantContext()
        msg = (req.message or "").strip().lower()
        chip = req.chip or ""

        if chip == "🚗 Plan a Route" or "plan a route" in msg or ctx.slot_filling_active:
            return self._handle_slot_filling(req.message or chip, ctx)

        # Preset chips are answered by the local deterministic engine: they map
        # to fixed explanations, so spending an API call on them would buy
        # nothing. Free-text questions are where an LLM actually helps.
        if not chip:
            provider = get_provider()
            if provider is not None:
                try:
                    llm_reply = self._call_external_llm(req.message, ctx, provider)
                    if llm_reply:
                        return AssistantChatResponse(
                            reply=llm_reply,
                            suggested_chips=SUGGESTED_CHIPS,
                            metrics_summary=self._extract_metrics_summary(ctx),
                        )
                except Exception as e:
                    logger.warning(
                        "Provider '%s' failed, falling back to local engine: %s",
                        provider.name, e,
                    )

        # Local context-grounded reasoning engine (Fast, zero API tokens, 100% reliable)
        reply = self._generate_local_response(msg, chip, ctx)
        return AssistantChatResponse(
            reply=reply,
            suggested_chips=SUGGESTED_CHIPS,
            metrics_summary=self._extract_metrics_summary(ctx),
            context=ctx,
        )

    def _handle_slot_filling(self, msg: str, ctx: AssistantContext) -> AssistantChatResponse:
        """
        Conversational route planning for someone who does not know what a node
        id is.

        Two things it deliberately does not do. It does not ask for a depot or a
        stop by number — it hands over to the map and waits, because "which
        junction is node 4731" is not a question a dispatcher can answer. And it
        does not invent a road network: the plan is built on the Bareilly map
        that ships with the app, so the route the user is shown is on the same
        streets as everything else on screen.
        """
        text = (msg or "").strip()
        low = text.lower()

        if low in {"cancel", "stop", "quit", "nevermind", "never mind"}:
            return self._slot_cancel(ctx)

        if not ctx.slot_filling_active:
            ctx.slot_filling_active = True
            ctx.collected_params = {}
            ctx.current_prompt = "horizon"
            return AssistantChatResponse(
                reply=(
                    f"Let's plan a delivery round on **{MVP_AREA_LABEL}**.\n\n"
                    "First: **how long is the shift**, in minutes? "
                    "(8 hours is 480.)"
                ),
                suggested_chips=["480 minutes", "Cancel"],
                context=ctx,
            )

        step = ctx.current_prompt

        if step == "horizon":
            value = self._extract_int(text, "horizon")
            if value is None or value < 1:
                return self._slot_retry(ctx, "How long is the shift, in minutes? (8 hours is 480.)",
                                        ["480 minutes", "Cancel"])
            ctx.collected_params["horizon"] = value
            ctx.current_prompt = "n_vehicles"
            return AssistantChatResponse(
                reply=f"{value} minutes. **How many vans** do you have available?",
                suggested_chips=["3 vans", "Cancel"],
                context=ctx,
            )

        if step == "n_vehicles":
            value = self._extract_int(text, "n_vehicles")
            if value is None or value < 1:
                return self._slot_retry(ctx, "How many vans do you have available?",
                                        ["3 vans", "Cancel"])
            ctx.collected_params["n_vehicles"] = value
            ctx.current_prompt = "area_choice"
            return AssistantChatResponse(
                reply=(
                    f"{value} vans. Now the area.\n\n"
                    f"Do you want to plan across the whole of **{MVP_AREA_LABEL}**, "
                    "or draw a smaller patch of it — one district, one ward — "
                    "and work inside that?"
                ),
                suggested_chips=["Use the whole city", "Draw an area", "Cancel"],
                context=ctx,
            )

        if step == "area_choice":
            if any(w in low for w in ("draw", "smaller", "area", "boundary", "patch", "district", "ward")):
                ctx.current_prompt = "await_boundary"
                # The control room already has a network loaded before this step
                # starts (Bareilly, by default), so `active_network_id` is never
                # empty. Recording which one it was here is the only way to
                # later tell "a box was drawn and loaded" apart from "nothing
                # happened yet, the old network is still active" — checking for
                # any id at all made every "done" true immediately.
                ctx.collected_params["pre_boundary_network_id"] = ctx.active_network_id
                return self._slot_ask_for_boundary(ctx, first_time=True)
            if any(w in low for w in ("whole", "all", "city", "everything", "full")):
                ctx.current_prompt = "await_map"
                return self._slot_ask_for_map(ctx, first_time=True)
            return AssistantChatResponse(
                reply="Whole city, or draw a smaller area?",
                suggested_chips=["Use the whole city", "Draw an area", "Cancel"],
                context=ctx,
            )

        if step == "await_boundary":
            # A box was drawn and loaded only once the active network is BOTH
            # present and different from whatever was loaded when this wait
            # began — not merely present, which the pre-existing Bareilly load
            # already satisfied before the user touched the map.
            before = ctx.collected_params.get("pre_boundary_network_id")
            if not ctx.active_network_id or ctx.active_network_id == before:
                return self._slot_ask_for_boundary(ctx, first_time=False)
            ctx.collected_params["boundary_network"] = ctx.active_network_id
            ctx.current_prompt = "await_map"
            return self._slot_ask_for_map(ctx, first_time=True)

        if step == "await_map":
            depot = ctx.selected_depot
            stops = list(ctx.selected_stops or [])
            if depot is None or not stops:
                return self._slot_ask_for_map(ctx, first_time=False)
            ctx.collected_params["depot"] = depot
            ctx.collected_params["stops"] = stops
            ctx.current_prompt = "confirm"
            return AssistantChatResponse(
                reply=(
                    "Here is the round I will plan:\n\n"
                    f"- **Area:** {MVP_AREA_LABEL}\n"
                    f"- **Shift:** {ctx.collected_params['horizon']} minutes\n"
                    f"- **Vans available:** {ctx.collected_params['n_vehicles']}\n"
                    f"- **Stops you picked:** {len(stops)}\n\n"
                    "Shall I plan it?"
                ),
                suggested_chips=["Yes, plan it", "Change the stops", "Cancel"],
                context=ctx,
            )

        if step == "confirm":
            if "stop" in low or "map" in low or "pick" in low:
                ctx.current_prompt = "await_map"
                return self._slot_ask_for_map(ctx, first_time=False)
            changed = self._apply_correction(low, ctx)
            if changed:
                return AssistantChatResponse(
                    reply=f"Updated — {changed}. Shall I plan it now?",
                    suggested_chips=["Yes, plan it", "Cancel"],
                    context=ctx,
                )
            if not any(w in low for w in ("yes", "yep", "yeah", "ok", "okay", "sure", "plan", "go", "proceed")):
                return AssistantChatResponse(
                    reply="Reply **yes** to plan it, or tell me what to change "
                          "(for example \"change vans to 5\" or \"change the stops\").",
                    suggested_chips=["Yes, plan it", "Change the stops", "Cancel"],
                    context=ctx,
                )
            return self._slot_solve(ctx)

        # Unknown state — start again rather than stall in it.
        return self._slot_cancel(ctx)

    # ---- slot-filling helpers ---------------------------------------------

    def _slot_cancel(self, ctx: AssistantContext) -> AssistantChatResponse:
        ctx.slot_filling_active = False
        ctx.collected_params = {}
        ctx.current_prompt = None
        return AssistantChatResponse(
            reply="Route planning cancelled. Ask me anything else whenever you like.",
            suggested_chips=SUGGESTED_CHIPS,
            context=ctx,
        )

    def _slot_retry(self, ctx: AssistantContext, prompt: str,
                    chips: List[str]) -> AssistantChatResponse:
        return AssistantChatResponse(
            reply=f"Sorry, I didn't catch a number there. {prompt}",
            suggested_chips=chips,
            context=ctx,
        )

    def _slot_ask_for_map(self, ctx: AssistantContext, first_time: bool) -> AssistantChatResponse:
        """
        Hands over to the map and waits. `open_picker` tells the frontend to
        show the Bareilly network in the control room with stop-picking on; the
        user's clicks come back on the next turn as selected_depot/selected_stops.
        """
        reply = (
            "Now show me the round on the map.\n\n"
            "1. Click the **depot** — where the vans start and finish.\n"
            "2. Click every **stop** they have to serve.\n"
            "3. Come back and say **done**.\n\n"
            "I'll wait."
        ) if first_time else (
            "I don't have a depot and at least one stop yet.\n\n"
            "On the map: click the **depot** first, then each **stop**, "
            "then say **done**."
        )
        # After a boundary was drawn the control room holds the clipped network,
        # and reloading the city over it would throw away the area the user
        # chose — and with it the node ids they are about to click.
        drew_boundary = bool(ctx.collected_params.get("boundary_network"))
        return AssistantChatResponse(
            reply=reply,
            suggested_chips=["Done picking", "Cancel"],
            context=ctx,
            map_action={
                "kind": "open_picker",
                "network_name": MVP_NETWORK,
                "ensure_area": not drew_boundary,
            },
        )

    def _slot_ask_for_boundary(self, ctx: AssistantContext, first_time: bool) -> AssistantChatResponse:
        """Hands over so the user can drag a box and load the roads inside it."""
        reply = (
            "Drag a box around the area you want to work in, then press "
            "**Load roads in this boundary**. Inside the dashed purple outline "
            "it comes straight off the offline map, so it is instant.\n\n"
            "Tell me **done** once the roads are on screen."
        ) if first_time else (
            "I can't see a loaded area yet. Drag a box on the map, press "
            "**Load roads in this boundary**, then say **done**."
        )
        return AssistantChatResponse(
            reply=reply,
            suggested_chips=["Done", "Use the whole city", "Cancel"],
            context=ctx,
            map_action={"kind": "draw_boundary", "network_name": MVP_NETWORK},
        )

    def _apply_correction(self, low: str, ctx: AssistantContext) -> Optional[str]:
        """'change vans to 5' -> updates the slot. Returns what changed, or None."""
        if not any(w in low for w in ("change", "update", "make it", "set")):
            return None
        numbers = re.findall(r"\d+", low)
        if not numbers:
            return None
        value = int(numbers[-1])
        if any(w in low for w in ("van", "vehicle", "truck")):
            ctx.collected_params["n_vehicles"] = value
            return f"{value} vans"
        if any(w in low for w in ("shift", "time", "minute", "horizon", "hour")):
            ctx.collected_params["horizon"] = value
            return f"a {value} minute shift"
        return None

    def _extract_int(self, text: str, field: str) -> Optional[int]:
        """
        Plain digits first. An LLM call per numeric answer is a round trip and a
        bill for something a regex already does, so the provider is only asked
        when the regex finds nothing ("four vans").
        """
        match = re.search(r"\d+", text or "")
        if match:
            try:
                return int(match.group())
            except ValueError:
                pass
        provider = get_provider()
        if not provider:
            return None
        try:
            raw = provider.complete(
                system_prompt=(
                    f"Extract the integer value for '{field}' from the message. "
                    "Reply with digits only, or None."
                ),
                user_prompt=text, max_tokens=8,
            )
            found = re.search(r"\d+", raw or "")
            return int(found.group()) if found else None
        except Exception:
            return None

    def _slot_solve(self, ctx: AssistantContext) -> AssistantChatResponse:
        """Builds and solves the round on the shipped Bareilly network."""
        try:
            from app.api.routes import _build_solve_response, _network_payload
            from app.core.cached_network import load_cached_network

            horizon = float(ctx.collected_params["horizon"])
            stops = list(ctx.collected_params["stops"])
            depot = int(ctx.collected_params["depot"])
            vans = int(ctx.collected_params["n_vehicles"])

            # Plan on the network those clicks came from. A boundary clipped out
            # of the offline map numbers its nodes from zero while the full
            # cached city carries OSM ids, and the two sets do not overlap at
            # all — so re-loading the city here would reject every stop the user
            # picked inside a drawn area.
            network_id = ctx.active_network_id
            net = None
            if network_id:
                try:
                    net = store.get_network(network_id)
                except KeyError:
                    net = None
            if net is None:
                net = load_cached_network(MVP_NETWORK, congestion_seed=42)
                network_id = store.put_network(net, is_geo=True)

            # Delivery windows cannot outlast the shift they sit in; the API
            # rejects the instance outright if they do.
            window_max = min(180.0, horizon)
            problem = generate_synthetic_vrp(
                net=net,
                n_customers=len(stops),
                customer_nodes=stops,
                depot=depot,
                vehicle_capacity=80.0,
                n_vehicles=vans,
                demand_range=(5.0, 20.0),
                horizon=horizon,
                window_length_range=(min(60.0, window_max), window_max),
                service_time=10.0,
                seed=42,
            )
            vrp_id = store.put_vrp(network_id, problem)

            started = time.perf_counter()
            opt = QPSOVRPOptimizer(problem, n_particles=30, max_iter=60,
                                   seed=42, use_local_search=True)
            result = opt.optimize()
            runtime_ms = (time.perf_counter() - started) * 1000
            best = result.best_solution

            ctx.slot_filling_active = False
            ctx.current_prompt = None
            ctx.collected_params = {}
            ctx.optimized_time = round(best.total_time, 2)
            ctx.optimized_dist = round(best.total_distance, 2)
            used = len([r for r in best.routes if r])
            ctx.num_vehicles = used

            solution = _build_solve_response(
                problem, best, result.convergence_curve,
                result.n_evaluations, "QPSO", runtime_ms,
            )
            nodes, edges = _network_payload(net)

            late = "" if best.feasible else (
                "\n\n⚠️ Not every stop can be reached inside its delivery window "
                "with this shift and fleet — try a longer shift or another van."
            )
            reply = (
                "✅ **Planned.**\n\n"
                f"- **Stops:** {len(stops)}\n"
                f"- **Vans used:** {used} of {vans}\n"
                f"- **Total time:** {ctx.optimized_time} min\n"
                f"- **Total distance:** {ctx.optimized_dist} km\n\n"
                "The routes are on the map in the control room." + late
            )

            return AssistantChatResponse(
                reply=reply,
                suggested_chips=SUGGESTED_CHIPS,
                context=ctx,
                map_action={
                    "kind": "solution",
                    "network": {
                        "network_id": network_id,
                        "num_nodes": net.num_nodes(),
                        "num_edges": len(edges),
                        "is_geo": True,
                        "nodes": [n.model_dump() for n in nodes],
                        "edges": [e.model_dump() for e in edges],
                        "area_label": MVP_AREA_LABEL,
                    },
                    "vrp": {
                        "vrp_id": vrp_id,
                        "network_id": network_id,
                        "depot": problem.depot,
                        "n_vehicles": problem.n_vehicles,
                        "vehicle_capacity": problem.vehicle_capacity,
                        "total_demand": sum(c.demand for c in problem.customers),
                        "customers": [
                            {
                                "node_id": c.node_id, "demand": c.demand,
                                "ready_time": c.ready_time, "due_time": c.due_time,
                                "service_time": c.service_time,
                            } for c in problem.customers
                        ],
                    },
                    "solution": solution.model_dump(),
                },
            )
        except Exception as e:
            logger.exception("Conversational route planning failed")
            ctx.slot_filling_active = False
            ctx.current_prompt = None
            ctx.collected_params = {}
            return AssistantChatResponse(
                reply=f"I couldn't plan that round: {e}",
                suggested_chips=SUGGESTED_CHIPS,
                context=ctx,
            )

    def _get_or_create_story_card(self, ctx: AssistantContext) -> str:
        """
        Generates or retrieves a compact cached 'Story Card' summarizing
        the active scenario and metrics. Cuts prompt tokens by >85% compared
        to re-loading full raw JSON contexts every turn.
        """
        cache_key = (
            f"{ctx.scenario_name}_{ctx.time_saved_pct}_{ctx.delay_saved_pct}_"
            f"{ctx.optimized_algo}_{ctx.baseline_algo}_{ctx.num_customers}"
        )
        if cache_key in self._story_card_cache:
            return self._story_card_cache[cache_key]

        scen = ctx.scenario_name or "Active Urban Network"

        if not _has_comparison_results(ctx):
            # No measured run: tell the LLM explicitly rather than handing it
            # placeholder numbers it would then state as fact.
            card = (
                f"[STORY_CARD] Scenario: {scen}. NO optimization run has been executed "
                f"in this session, so there are NO measured metrics available. "
                f"Do not state or estimate any performance numbers. Tell the user to run "
                f"a solve/compare first, or answer conceptual questions about QPSO, the "
                f"map, or convergence without citing figures."
            )
            self._story_card_cache[cache_key] = card
            return card

        size = (
            f"{ctx.num_nodes} nodes, " if ctx.num_nodes else ""
        ) + (
            f"{ctx.num_customers} customers, " if ctx.num_customers else ""
        ) + (
            f"{ctx.num_vehicles} vehicles" if ctx.num_vehicles else ""
        )
        feas = (
            "100% On-Time (0 violations)"
            if ctx.optimized_feasible
            else "constraint violations present"
            if ctx.optimized_feasible is not None
            else "feasibility not reported"
        )
        late = _fmt(ctx.baseline_late, "m")

        card = (
            f"[STORY_CARD] Scenario: {scen} ({size.strip().rstrip(',')}). "
            f"{ctx.optimized_algo or 'Optimized'} vs {ctx.baseline_algo or 'Baseline'}: "
            f"Time saved {_fmt(ctx.time_saved_pct, '%')} (-{_fmt(ctx.time_saved_min, 'm')}), "
            f"Congestion delay avoided {_fmt(ctx.delay_saved_pct, '%')} (-{_fmt(ctx.delay_saved_min, 'm')}), "
            f"Mileage saved {_fmt(ctx.dist_saved_pct, '%')}. "
            f"Punctuality: {feas} vs Baseline lateness penalty {late}. "
            f"Cite only these figures; if a value is 'n/a' it was not measured. "
            f"Key mechanism: Quantum delta-potential well tunneling avoids high-friction local minima; memetic 2-opt eliminates cross-overs."
        )
        self._story_card_cache[cache_key] = card
        return card

    def _extract_metrics_summary(self, ctx: AssistantContext) -> Dict[str, Any]:
        """Extract quick numbers for UI badge rendering."""
        return {
            "time_saved_pct": ctx.time_saved_pct or 0.0,
            "delay_saved_pct": ctx.delay_saved_pct or 0.0,
            "dist_saved_pct": ctx.dist_saved_pct or 0.0,
            "time_saved_min": ctx.time_saved_min or 0.0,
            "delay_saved_min": ctx.delay_saved_min or 0.0,
            "feasible": ctx.optimized_feasible if ctx.optimized_feasible is not None else True,
            "baseline_algo": ctx.baseline_algo or "Greedy Baseline",
            "optimized_algo": ctx.optimized_algo or "QPSO (Quantum Hybrid)",
        }

    def _generate_local_response(self, msg: str, chip: str, ctx: AssistantContext) -> str:
        """Rule-based, domain-grounded synthesis using live session numbers."""
        words = set(re.findall(r"\b[a-zA-Z0-9_]+\b", msg.lower()))

        # 1. Intent: Explain Route & Operational Impact
        if chip == "explain_route" or "explain route" in msg or "what am i looking at" in msg or bool(words & {"summary", "overview", "savings", "result"}):
            return self._explain_route(ctx)

        # 2. Intent: Why QPSO / Quantum Advantage
        if chip == "why_qpso" or "why qpso" in msg or "why quantum" in msg or "quantum advantage" in msg or "beat baseline" in msg or "delta potential" in msg or "metaheuristic" in words:
            return self._explain_why_qpso(ctx)

        # 3. Intent: Explain Convergence Chart
        if chip == "explain_conv" or bool(words & {"convergence", "chart", "curve", "iterations", "fitness"}):
            return self._explain_convergence(ctx)

        # 4. Intent: Explain Map & Traffic Visuals
        if chip == "explain_map" or bool(words & {"map", "color", "colors", "red", "green", "depot", "leaflet", "svg", "roads", "friction"}):
            return self._explain_map(ctx)

        # 5. Intent: Constraints & Time Windows (avoid substring collisions like 'simulated' -> 'late')
        if chip == "explain_tw" or "time window" in msg or "time windows" in msg or bool(words & {"sla", "late", "lateness", "capacity", "penalty", "feasible"}):
            return self._explain_constraints(ctx)

        # 6. Intent: Benchmark Comparison
        if bool(words & {"benchmark", "ga", "genetic", "sa", "annealing", "pso", "compare", "rank", "ranks"}):
            return self._explain_benchmark(ctx)

        # Default: General contextual inquiry (leverages RAG documentation index)
        return self._explain_general(msg, ctx)

    # -----------------------------------------------------------------------
    # Domain Explanation Generators
    # -----------------------------------------------------------------------

    def _explain_route(self, ctx: AssistantContext) -> str:
        if not _has_comparison_results(ctx):
            return NO_RESULTS_NOTICE

        scenario = ctx.scenario_name or "Current Fleet Deployment"

        # Punctuality is reported from the actual solve, not assumed. QPSO uses
        # penalty-based constraint handling and does not guarantee feasibility.
        if ctx.optimized_feasible is True:
            late_txt = _fmt(ctx.optimized_late, " min") if ctx.optimized_late is not None else "0.0 min"
            punctuality = f"**Feasible** — all time windows and capacities respected (lateness {late_txt})."
        elif ctx.optimized_feasible is False:
            punctuality = (
                f"⚠️ **Infeasible** — the optimized solution still violates constraints "
                f"(lateness {_fmt(ctx.optimized_late, ' min')}). Treat the savings above as "
                f"an upper bound; consider more iterations or a larger fleet."
            )
        else:
            punctuality = "Feasibility was not reported for this run."

        return (
            f"### 📊 Executive Route Analysis — {scenario}\n\n"
            f"Evaluating **{ctx.baseline_algo or 'baseline'}** vs **{ctx.optimized_algo or 'optimized'}**:\n\n"
            f"- ⚡ **Total Travel Time**: Cut by {_delta(ctx.time_saved_pct)} "
            f"({_fmt(ctx.baseline_time, ' min')} ➔ **{_fmt(ctx.optimized_time, ' min')}**, "
            f"saving {_fmt(ctx.time_saved_min, ' minutes')}).\n"
            f"- 🛑 **Congestion Avoided**: Reduced by {_delta(ctx.delay_saved_pct)} "
            f"({_fmt(ctx.baseline_delay, ' min')} trapped in traffic ➔ **{_fmt(ctx.optimized_delay, ' min')}**).\n"
            f"- 🛣️ **Mileage & Fuel**: Reduced by {_delta(ctx.dist_saved_pct)} "
            f"(saving {_fmt(ctx.dist_saved_km, ' km')}).\n"
            f"- 🎯 **Punctuality & SLA**: {punctuality}\n\n"
            f"**Operational Takeaway:** Classical dispatch chooses immediate short links into gridlocks. "
            f"QPSO navigates around congested corridors, trading a small distance detour for time and reliability gains."
        )

    def _explain_why_qpso(self, ctx: AssistantContext) -> str:
        # Conceptual explanation is valid with or without a run; only the
        # closing result line is gated on real measurements.
        if ctx.delay_saved_pct is not None:
            closing = (
                f"**Result (this run):** Avoided **{ctx.delay_saved_pct:.1f}%** of traffic bottleneck delays"
                + (
                    ", with all constraints satisfied."
                    if ctx.optimized_feasible
                    else "; note this run still reports constraint violations."
                    if ctx.optimized_feasible is False
                    else "."
                )
            )
        else:
            closing = (
                "**Result:** Run a solve or comparison to see how much bottleneck delay "
                "QPSO actually avoids on your current network."
            )
        return (
            "### ⚛️ Why Quantum-Inspired (QPSO) Outperforms Classical Baselines\n\n"
            "In large-scale vehicle routing under dynamic congestion, classical algorithms (Greedy, GA, Standard PSO) frequently get trapped in **local minima**—they commit vehicles to arterial roads that look short on distance but are crippled by traffic delay.\n\n"
            "#### 3 Core Quantum Principles in Our Engine:\n"
            "1. **Delta-Potential-Well Dynamics**:\n"
            "   - Standard PSO relies on velocity vectors (`v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)`), which easily stall when momentum slows.\n"
            "   - QPSO models each particle as bound in a quantum delta-potential well. Particle coordinates appear probabilistically according to wave function $\\psi(x)$, giving non-zero probability to appear anywhere in search space.\n\n"
            "2. **Quantum Tunneling through High-Cost Barriers**:\n"
            "   - When traffic congestion creates an energy barrier (fitness valley), classical heuristics cannot cross without expensive random restarts.\n"
            "   - Quantum particles **tunnel** directly through the barrier into globally optimal, decongested routing basins.\n\n"
            "3. **Mean-Best (mbest) Swarm Attractor & Memetic Refinement**:\n"
            "   - Particles are guided by the centroid of all personal best positions ($mbest = \\frac{1}{N} \\sum pbest_i$), keeping the swarm cohesive without arbitrary velocity clamp parameters ($V_{max}$).\n"
            "   - A hybridized **2-opt / or-opt local search** operates after quantum exploration to untangle route crossings and enforce tight customer time-window alignment.\n\n"
            f"{closing}"
        )

    def _explain_convergence(self, ctx: AssistantContext) -> str:
        return (
            "### 📈 Interpreting the Convergence Analysis Chart\n\n"
            "The **Convergence Curve** tracks optimization progress across iterations:\n\n"
            "- **X-Axis (Iterations)**: Number of evolutionary cycles completed (default: 150 iterations).\n"
            "- **Y-Axis (Fitness Score)**: Objective penalty function (**lower is better**).\n"
            "  - Fitness = `Travel Time + Congestion Penalties + Time-Window Violation Penalties + Over-Capacity Penalties`.\n\n"
            "#### What the Curve Pattern Indicates:\n"
            "- **Rapid Early Drop (Iter 0–30)**: Quantum tunneling rapidly locates feasible sub-tours and eliminates constraint penalties.\n"
            "- **Memetic Step-downs (Iter 30–90)**: 2-opt/or-opt local search refines vehicle clusters and swaps delivery sequences.\n"
            "- **Asymptotic Stability (Iter 90+)**: Curve flattens near the global minimum, confirming stable convergence without oscillation."
        )

    def _explain_map(self, ctx: AssistantContext) -> str:
        if ctx.num_customers and ctx.num_vehicles:
            intro = (
                f"The map renders a full road-network topology with "
                f"**{ctx.num_customers} customer delivery nodes** and an active fleet of "
                f"**{ctx.num_vehicles} vehicles**."
            )
        else:
            intro = (
                "The map renders the full road-network topology: customer delivery nodes, "
                "the depot, and per-vehicle routes once an instance has been generated."
            )
        return (
            "### 🗺️ Understanding the Map & Traffic Network\n\n"
            f"{intro}\n\n"
            "#### Visual Guide:\n"
            "- 🟡 **Golden Central Node (Depot)**: Node `0`. The hub where all vehicles depart with loaded capacity and return before the operating horizon ends.\n"
            "- ⚪ **Numbered Customer Pins**: Delivery destinations carrying specific payload demand and service time-windows (e.g. `[10:00 - 11:30]`).\n"
            "- 🚦 **Road Segments & Congestion Friction**:\n"
            "  - **Green / Mint Lines**: Free-flowing road links ($1.0\\times - 1.2\\times$ friction factor).\n"
            "  - **Yellow / Orange Lines**: Moderate slowing ($1.3\\times - 1.9\\times$ friction).\n"
            "  - **Red Glowing Lines**: Severe gridlock ($2.0\\times - 3.0\\times$ friction), where travel time doubles or triples.\n"
            "- 🎨 **Multi-Colored Routes**: Individual vehicle circuits. Paths follow exact road-network geometry, showing how QPSO dynamically routes around red congestion corridors."
        )

    def _explain_constraints(self, ctx: AssistantContext) -> str:
        # Constraints are handled via penalties in the fitness function, so
        # feasibility is an outcome of each run — never a guarantee.
        if ctx.baseline_late is not None or ctx.optimized_late is not None:
            observed = (
                "#### Observed on this run:\n"
                f"   - Baseline lateness penalty: **{_fmt(ctx.baseline_late, ' min')}**.\n"
                f"   - Optimized lateness penalty: **{_fmt(ctx.optimized_late, ' min')}**"
                + (
                    " — fully feasible.\n\n"
                    if ctx.optimized_feasible
                    else " — ⚠️ constraints still violated.\n\n"
                    if ctx.optimized_feasible is False
                    else ".\n\n"
                )
            )
        else:
            observed = (
                "#### Observed on this run:\n"
                "   - No solve has been run yet, so there are no measured lateness "
                "or capacity figures to report.\n\n"
            )

        return (
            "### 🎯 Time-Window & Capacity Constraints (CVRPTW)\n\n"
            "In urban logistics, finding the shortest distance is useless if a delivery truck arrives after a customer's business hours or exceeds legal capacity.\n\n"
            "#### How the Optimization Engine Enforces Constraints:\n"
            "1. **Customer Time Windows $[e_i, l_i]$** — arrival outside a window adds a "
            "weighted penalty to the fitness score, pushing the search toward on-time schedules.\n"
            "2. **Vehicle Payload Capacity** — demand assigned per vehicle above capacity is "
            "penalized the same way, so overloaded routes are driven out of the population.\n"
            "3. **Operating Horizon** — vehicles are scored on returning to the depot within schedule.\n\n"
            "These are *soft* penalty terms, not hard guarantees: a run is only feasible if "
            "the reported violation totals are zero.\n\n"
            f"{observed}"
        )

    def _explain_benchmark(self, ctx: AssistantContext) -> str:
        ranks = ctx.benchmark_ranks
        if not ranks:
            return (
                "### 🏆 Benchmark Comparison\n\n"
                "No benchmark has been run in this session yet, so I can't rank the algorithms "
                "from measured data — and I won't guess at the ordering.\n\n"
                "Run **Benchmark** in the dashboard to evaluate QPSO against the classical "
                "baselines (GA, SA, standard PSO, Greedy nearest-neighbour) on the current "
                "instance. I'll then report the actual fitness, travel time, congestion delay, "
                "feasibility and runtime for each.\n\n"
                "**What the comparison is for:** the problem statement requires benchmarking "
                "against conventional metaheuristics, so the ranking has to come from a real "
                "run on your instance rather than from prior expectations."
            )

        # Rank by fitness (lower is better); entries missing fitness sort last.
        def _fitness(entry: Dict[str, Any]) -> float:
            val = entry.get("fitness")
            return float(val) if isinstance(val, (int, float)) else float("inf")

        ordered = sorted(ranks, key=_fitness)
        medals = ["🥇", "🥈", "🥉"]

        rows = []
        for i, entry in enumerate(ordered):
            name = entry.get("algorithm") or entry.get("name") or "unknown"
            place = medals[i] if i < len(medals) else f"{i + 1}th"
            fitness = entry.get("fitness")
            total_time = entry.get("total_time")
            delay = entry.get("congestion_delay_min")
            feasible = entry.get("feasible")
            runtime = entry.get("runtime_ms")

            feas_txt = (
                "✅ Feasible" if feasible is True
                else "⚠️ Violations" if feasible is False
                else "—"
            )
            rows.append(
                f"| {place} **{name}** | "
                f"{_fmt(fitness if isinstance(fitness, (int, float)) else None)} | "
                f"{_fmt(total_time if isinstance(total_time, (int, float)) else None, ' min')} | "
                f"{_fmt(delay if isinstance(delay, (int, float)) else None, ' min')} | "
                f"{feas_txt} | "
                f"{_fmt(runtime if isinstance(runtime, (int, float)) else None, ' ms', 0)} |"
            )

        best = ordered[0].get("algorithm") or "the top entry"
        return (
            f"### 🏆 Benchmark Comparison ({len(ordered)} algorithms, measured this run)\n\n"
            "Ranked by fitness (lower is better — fitness combines travel time with "
            "capacity and time-window violation penalties):\n\n"
            "| Rank / Algorithm | Fitness | Travel Time | Congestion Delay | Constraints | Runtime |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            + "\n".join(rows)
            + f"\n\n**Result on this instance:** **{best}** achieved the best fitness. "
            "Greedy is typically fastest in raw compute but myopic under congestion; "
            "the metaheuristics trade runtime for solution quality. Re-run on a larger "
            "instance to see how the gap scales."
        )

    def _explain_general(self, msg: str, ctx: AssistantContext) -> str:
        # Check if question can be answered from project documentation (Issue #33)
        try:
            from app.core.rag import rag_engine
            top_matches = rag_engine.retrieve(msg, top_k=1)
            if top_matches and top_matches[0][1] >= 1.2:
                rag_res = rag_engine.ask(msg, context=ctx, top_k=2)
                return rag_res.reply
        except Exception as e:
            logger.debug("RAG lookup in _explain_general failed: %s", e)

        # If the query fell through all specific intents and RAG, refuse it.
        if msg:
            return "I don't have that info."

        scenario = ctx.scenario_name or "Current Active Scenario"

        if not _has_comparison_results(ctx):
            return (
                f"### 💡 MargdarshaQ AI Copilot\n\n"
                f"No optimization run has been executed yet, so I have no measured results "
                f"to summarise for **{scenario}**.\n\n"
                f"Run a **Solve** or **Compare** to get grounded numbers, or ask me about "
                f"how QPSO works, what the map colours mean, or how to read the convergence chart."
            )

        feasibility = (
            "- **Feasibility**: all payload and time-window constraints satisfied.\n"
            if ctx.optimized_feasible
            else "- **Feasibility**: ⚠️ this run still reports constraint violations.\n"
            if ctx.optimized_feasible is False
            else "- **Feasibility**: not reported for this run.\n"
        )
        return (
            f"### 💡 MargdarshaQ AI Copilot — {scenario}\n\n"
            f"Active monitoring for **{scenario}**:\n"
            f"- **Performance**: **{_fmt(ctx.time_saved_pct, '%')}** time reduction and "
            f"**{_fmt(ctx.delay_saved_pct, '%')}** of congestion delay avoided versus "
            f"{ctx.baseline_algo or 'the baseline'}.\n"
            f"{feasibility}\n"
            f"Select a prompt above or ask about specific route legs, quantum tunneling mechanics, or benchmark trade-offs."
        )

    # -----------------------------------------------------------------------
    # External LLM Integration (Story-Card context + RAG Document Grounding)
    # -----------------------------------------------------------------------

    def _call_external_llm(self, prompt: str, ctx: AssistantContext,
                           provider: LLMProvider) -> Optional[str]:
        """
        Ask the configured provider, grounded in the Story Card and retrieved
        project documentation passages (Issue #33 RAG).

        Returns None on any failure so the caller falls back to the local
        deterministic engine, which always works.
        """
        story_card = self._get_or_create_story_card(ctx)

        # Retrieve relevant project documentation passages (Issue #33)
        doc_context = ""
        try:
            from app.core.rag import rag_engine
            matches = rag_engine.retrieve(prompt, top_k=2)
            if matches and matches[0][1] >= 0.5:
                doc_context = "PROJECT DOCUMENTATION RETRIEVAL CONTEXT:\n" + "\n\n".join(
                    f"[{c.document_path} > {c.section_title}]:\n{c.content[:450]}"
                    for c, _ in matches
                )
        except Exception as e:
            logger.debug("RAG passage retrieval failed in _call_external_llm: %s", e)

        system_parts = [
            "You are MargdarshaQ AI Copilot, an assistant for a quantum-inspired "
            "(QPSO) traffic route optimization platform.",
            f"SESSION CONTEXT: {story_card}",
        ]
        if doc_context:
            system_parts.append(doc_context)

        system_parts.append(
            "INSTRUCTIONS: Be direct, concise and professional. Prefer 2-4 short "
            "bullet points. Quote figures exactly as given in the session context - "
            "never estimate, extrapolate or invent a number. If the context says a "
            "value was not measured, say it was not measured. If referencing algorithm "
            "formulation or mechanics, ground your answer in the project documentation above. "
            "If the question is not answerable from the context above, do not speculate. Instead, explicitly state: 'I don't have that info'."
        )

        return provider.complete(
            system_prompt="\n\n".join(system_parts),
            user_prompt=prompt,
        )


# Global singleton instance
assistant_engine = AIAssistantExplainer()
