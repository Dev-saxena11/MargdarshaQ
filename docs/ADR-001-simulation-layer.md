# ADR-001: Keep the analytic cost model; use SUMO as a validator, not a replacement

**Status:** Accepted
**Date:** 2026-09-23

## Context

[Eclipse SUMO](https://eclipse.dev/sumo/) is the established open-source
microscopic traffic simulator: calibrated car-following and lane-change models,
traffic lights, junction capacity, queue spillback, and a live control
interface (TraCI). It is more detailed than anything in this repository, and
the obvious question is whether this project should use it instead of its own
traffic model.

It is worth being precise about what the two things are, because "simulator"
covers both and they are not the same job:

* **SUMO** simulates how traffic *emerges* from individual vehicles obeying
  local rules. It answers "what happens if these vehicles drive these routes".
* **This project** decides *which van serves which stops in what order*, under
  capacity and time-window constraints. It answers "which routes should they
  drive".

SUMO does not solve CVRPTW. It has a taxi/DRT device with dispatch heuristics,
but dispatch heuristics are not a metaheuristic optimality study, and nothing in
SUMO produces the gap-to-optimal analysis this project is built around.

Inside this repository "our simulator" is also two different things with
opposite requirements:

1. **The cost model inside the optimiser** — `evaluate_solution` plus the
   time-dependent travel-time matrices. The metaheuristic calls it to score
   every candidate solution.
2. **The presentation layer** — the "Run the shift" playback and
   `dynamic_vrp.py`'s incident re-planning. Called once, for a human.

## Decision

**Keep the analytic cost model for optimisation. Do not put SUMO in that loop.**

**Use SUMO, if and when it is integrated, as an external validator of a finished
plan and optionally as a higher-fidelity playback** — both of which sit outside
the search.

## Rationale

The deciding number is call volume. Measured on a 20-customer instance:

```
one cost-model evaluation :     52.1 us
evaluations in one solve  :    7,500
one full QPSO solve       :     1.12 s
```

Substituting a SUMO run for each evaluation:

| Cost of one SUMO run | Time for one solve |
| ---: | ---: |
| 1 s | 2.1 hours |
| 5 s | 10.4 hours |
| 30 s | 62.5 hours |

A 1.12-second solve becomes hours, and one second is optimistic for a
493-junction network over a full day's clock. The five-algorithm, five-district
benchmark would take months.

This is not an engineering problem to be optimised around. A metaheuristic *is*
"evaluate an enormous number of candidate solutions"; the cost model has to be
microseconds by construction. Every serious VRP solver — OR-Tools, LKH, VROOM —
uses an analytic cost model for the same reason.

A second consequence is specific to this repository.
[`app/core/exact_vrp.py`](../app/core/exact_vrp.py) can prove optimality only
because the objective is a closed-form, separable function: it decomposes the
cost into per-route pieces and assembles the optimum with a DP over subsets.
Against a black-box stochastic simulator, "proven optimum" is not computable at
all. Adopting SUMO as the objective would delete the strongest credibility
claim in the project.

## Trade-offs, stated plainly

### The analytic cost model

**For**

* Fast enough to optimise against — non-negotiable, per the numbers above.
* Decomposable, so fitness splits into distance, time, capacity and
  time-window penalty. That is what makes the gap columns meaningful.
* Exactly solvable, which is what `exact_vrp.py` depends on.
* No dependency: runs on a free-tier host, in CI, in the test suite, offline.

**Against**

* Fidelity is assumed, not validated. The congestion curve is two Gaussians
  with estimated amplitudes (see `app/core/traffic_profile.py`, which says so).
* No queueing, no intersection delay, no signals, no vehicle interaction.
* Congestion is exogenous — our vans experience traffic but do not create it.
  Fine for three vans, wrong for a large fleet.
* Nobody outside this project has validated it.

### SUMO

**For**

* Calibrated microscopic fidelity, including effects the analytic model cannot
  express at all (self-congestion, signal coordination, junction capacity).
* External credibility: "validated against SUMO" is a claim a reviewer accepts
  without argument.

**Against**

* 10^5–10^6 times too slow to sit inside the search. Decisive.
* Removes the exact baseline, since optimality proofs need an analytic
  objective.
* Heavy dependency — a C++ toolchain, `netconvert`, demand files. Will not run
  on the free-tier backend or in the test suite.
* Stochastic, so the seeded, reproducible benchmark tables stop being
  reproducible.
* **Has its own calibration problem.** SUMO with default parameters on an
  uncalibrated Indian road network is not automatically more accurate than the
  model here. It is more *detailed*, which is not the same thing. This is the
  point most comparisons miss.

## Consequences

The architecture is layered, which is standard practice in the field:

```
Optimise  ->  analytic cost model     (52 us x 7,500 per solve, exact-solvable)
Validate  ->  SUMO, once per plan     (predicted vs simulated travel time)
Present   ->  SUMO-backed playback, or the existing animation
```

Positive: the strongest tool in the space becomes evidence for our numbers
rather than a competitor to them. It gives a defensible answer to the obvious
challenge to the impact figures — *the predicted 5.8 driver-hours held to within
N% inside an independently-calibrated microscopic simulator*.

Negative: a second model to keep in step with the first, and a heavyweight
dependency in the validation path (not in the deployed service).

**The caveat that must travel with any such claim:** a SUMO comparison only
means something if SUMO is calibrated for the network in question. Otherwise it
compares our assumptions against SUMO's defaults, which is not validation.

## Status of the validation path

Not built. SUMO is not installed in the development environment, so an
integration written here could not be run, and shipping an unexercised export
would be worse than shipping none. The export side is tractable when SUMO is
available: the road network and a solved plan already carry everything
`netconvert` and a `.rou.xml` need.

## Related

* [`app/core/exact_vrp.py`](../app/core/exact_vrp.py) — the optimality proof
  that depends on an analytic objective
* [`app/core/solomon.py`](../app/core/solomon.py) — the other half of the
  credibility answer, at 100 customers
* [`app/core/traffic_profile.py`](../app/core/traffic_profile.py) — the
  congestion model whose amplitudes are the thing SUMO would validate
