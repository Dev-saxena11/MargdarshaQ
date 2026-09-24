# Mathematical Formulation — Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)

This document provides the formal mathematical model, constraint equations, decision variable definitions, and quantum-inspired metaheuristic formulation for the SIH26137 Quantum-Inspired Intelligent Traffic Route Optimization platform.

---

## 1. Problem Description & Graph Network Model

Let the urban road network be represented as a weighted directed graph:
$$G = (V, E)$$
where:
- $V = \{0\} \cup C$ is the set of all vertices, with vertex $0$ denoting the central depot and $C = \{1, 2, \dots, N\}$ representing the set of customer delivery locations.
- $E = \{(u, v) \mid u, v \in V, u \neq v\}$ is the set of traversable road segments.

### Network Edge Weights & Traffic Congestion
Each road edge $(u, v) \in E$ has:
- Physical road distance: $D(u, v) \ge 0$ (kilometers)
- Base free-flow travel time: $T_0(u, v) \ge 0$ (minutes)
- Congestion friction factor: $C(u, v) \ge 1.0$, updated dynamically based on real-time traffic incidents or time-of-day rush hour profiles.

The effective travel time along edge $(u, v)$ is:
$$T(u, v) = T_0(u, v) \cdot C(u, v)$$

For time-dependent routing with time buckets $b \in \{0, 1, \dots, B-1\}$ (e.g. 30-minute intervals over operating horizon $H$):
$$T(u, v, t) = T_0(u, v) \cdot C(u, v, \text{bucket}(t))$$

The all-pairs shortest path distance matrix $d_{ij}$ and travel time matrix $t_{ij}$ between any two problem locations $i, j \in V$ are computed via Dijkstra's algorithm over graph $G$.

---

## 2. Customer & Fleet Parameters

- **Customer demands**: Each customer $i \in C$ has a positive demand $q_i > 0$, with depot demand $q_0 = 0$.
- **Time windows**: Each customer $i \in C$ specifies a service time window $[e_i, l_i]$, where:
  - $e_i \ge 0$ is the earliest arrival time (ready time). If a vehicle arrives at $t < e_i$, it waits until $e_i$.
  - $l_i \ge e_i$ is the latest acceptable service start time (due time / SLA deadline).
- **Service duration**: Customer $i$ requires service duration $s_i \ge 0$ minutes. For depot, $s_0 = 0$.
- **Vehicle fleet**: $K$ vehicles. Capacity is $Q_k$ per vehicle, defaulting to the uniform $Q$ when no per-vehicle list is supplied (per-vehicle capacities are used for mid-route re-planning, where a van has already consumed part of its load).
- **Mixed fleet**: two further per-vehicle factors, both defaulting to $1.0$ so a uniform fleet is arithmetically unchanged:
  - $\sigma_k > 0$ — **speed factor**, multiplying this vehicle's travel time on every leg. Below $1.0$ is faster than the network's base speed. It multiplies rather than replaces, so a two-wheeler is quicker *through the same congestion*, not quicker in free flow.
  - $\gamma_k > 0$ — **cost per kilometre**, weighting distance driven by this vehicle. Relative, not currency; only the ratio matters to the objective.
- **Fleet utilisation mode**: a flag $\rho \in \{0, 1\}$ (`require_all_vehicles`, default $0$). When $\rho = 1$, $K$ is a fleet that must all be dispatched rather than a ceiling. See the idle-vehicle penalty in §4.
- **Depot operating window**: The depot operates within $[e_0, l_0]$, where $l_0$ represents the planning horizon $H$.

---

## 3. Decision Variables & Chromosome Encoding

### Classical Binary Decision Variables
For mathematical completeness, the discrete multi-commodity flow formulation defines:
$$x_{ijk} = \begin{cases} 1 & \text{if vehicle } k \text{ traverses arc } (i, j) \\ 0 & \text{otherwise} \end{cases} \quad \forall i, j \in V, k \in \{1, \dots, K\}$$
$$y_{ik} = \begin{cases} 1 & \text{if customer } i \text{ is served by vehicle } k \\ 0 & \text{otherwise} \end{cases} \quad \forall i \in C, k \in \{1, \dots, K\}$$
$$t_{ik} \ge 0 \quad \text{arrival time of vehicle } k \text{ at node } i$$

### Continuous Random-Key Encoding (for Metaheuristic Swarm)
Combinatorial permutations are mapped to continuous particle positions $X \in \mathbb{R}^N$ using a random-key representation:
$$X = [x_1, x_2, \dots, x_N], \quad x_i \in [0, K)$$
For customer $i \in \{1, \dots, N\}$:
- **Vehicle Assignment**: $k_i = \lfloor x_i \rfloor \in \{0, 1, \dots, K-1\}$
- **In-Route Visiting Priority**: $p_i = x_i - \lfloor x_i \rfloor \in [0, 1)$

Decoding sorts the customers assigned to vehicle $k$ in ascending order of their priorities $p_i$, constructing the tour:
$$R_k = (0, c_{k,1}, c_{k,2}, \dots, c_{k, m_k}, 0)$$

---

## 4. Objective Function & Penalty Formulation

This section states the function that is actually minimised. Every solver in
this project — QPSO, GA, SA, PSO and the greedy baselines — reaches the
objective through the single implementation in
[`app/core/vrp_problem.py`](../app/core/vrp_problem.py) (`evaluate_solution`),
so the model below is the one they all agree on.

### 4.1 Route clock

A route for vehicle $k$ is the ordered customer sequence
$R_k = (c_{k,1}, \dots, c_{k,m_k})$, driven as $0 \to c_{k,1} \to \dots \to c_{k,m_k} \to 0$.
The vehicle leaves the depot at $\tau_{k,0} = 0$. Writing $c_{k,0} = 0$ for the
depot, for each $j \in \{1, \dots, m_k\}$:

$$\theta_{k,j} = \sigma_k \cdot t\big(c_{k,j-1},\, c_{k,j},\, \tau_{k,j-1}\big) \qquad \text{(travel time, priced at the moment of departure)}$$

$$a_{k,j} = \tau_{k,j-1} + \theta_{k,j} \qquad \text{(arrival)}$$

$$w_{k,j} = \max\big(0,\; e_{c_{k,j}} - a_{k,j}\big) \qquad \text{(wait, if early)}$$

$$\beta_{k,j} = \max\big(a_{k,j},\; e_{c_{k,j}}\big) = a_{k,j} + w_{k,j} \qquad \text{(service start)}$$

$$\tau_{k,j} = \beta_{k,j} + s_{c_{k,j}} \qquad \text{(departure, after service)}$$

The return leg is $\theta_{k,m_k+1} = \sigma_k \cdot t(c_{k,m_k}, 0, \tau_{k,m_k})$, with no wait
and no service at the depot.

Note that $t(\cdot,\cdot,\tau)$ is evaluated **at the departure time**: with
time-dependence enabled the lookup selects the matrix for bucket
$b(\tau) = \min\big(\lfloor \tau / \Delta \rfloor,\, B-1\big)$, so a leg driven through rush
hour costs more than the same leg at midday. With time-dependence disabled
(the default) this degrades to the single static matrix $t_{ij}$.

### 4.2 Accumulated quantities

$$T_{\text{total}} = \sum_{k=1}^{K} \left[ \sum_{j=1}^{m_k} \big(\theta_{k,j} + w_{k,j}\big) \;+\; \theta_{k,m_k+1} \right]$$

$$D_{\text{total}} = \sum_{k=1}^{K} \gamma_k \left[ \sum_{j=1}^{m_k} d\big(c_{k,j-1}, c_{k,j}\big) \;+\; d\big(c_{k,m_k}, 0\big) \right]$$

Two things worth stating plainly, because both are easy to misread from the
symbols alone:

- $T_{\text{total}}$ **includes waiting time**, not just driving. A vehicle that
  arrives early and idles until the window opens is charged for the idling.
- $D_{\text{total}}$ is a **weighted** distance. For a uniform fleet every
  $\gamma_k = 1$ and it is distance in kilometres, but for a mixed fleet the
  figure reported as `total_distance` is a cost, not a raw odometer reading.

### 4.3 Penalties

Constraints are handled as soft penalties so the swarm can cross an infeasible
region rather than being walled out of it.

**Capacity.** Per vehicle, against that vehicle's own capacity:

$$\mathcal{P}_{\text{cap}} = \sum_{k=1}^{K} \max\left(0,\; \sum_{j=1}^{m_k} q_{c_{k,j}} - Q_k \right)$$

**Time window.** Lateness is measured from the **service start**
$\beta_{k,j}$, not from arrival — a vehicle that arrives before $e_i$ and waits
is late only if the wait itself pushes it past the deadline:

$$\mathcal{P}_{\text{time}} = \sum_{k=1}^{K} \sum_{j=1}^{m_k} \max\big(0,\; \beta_{k,j} - l_{c_{k,j}}\big) \;+\; 1000 \cdot \big|U\big|$$

where $U$ is the set of legs whose travel time is not finite — an unreachable
node, which a disconnected sub-graph or a closed road can produce. Such a leg
contributes a flat $1000$ and accrues no time or distance; the vehicle's clock
does not advance across it.

**Idle vehicle.** Only when the fleet must all be dispatched ($\rho = 1$):

$$\mathcal{P}_{\text{idle}} = \rho \cdot \big|\{\, k : R_k = \varnothing \,\}\big|$$

Left alone the optimiser parks any van it does not need, so asking for five and
being shown three is the correct answer to "how many do I need". This penalty
exists for the opposite question — a depot with five drivers rostered and paid
either way — and makes leaving one parked the expensive option instead.

### 4.4 Fitness

$$\boxed{\;\min \; \mathcal{F} \;=\; w_T \cdot T_{\text{total}} \;+\; w_D \cdot D_{\text{total}} \;+\; \lambda_{\text{cap}} \cdot \mathcal{P}_{\text{cap}} \;+\; \lambda_{\text{time}} \cdot \mathcal{P}_{\text{time}} \;+\; \lambda_{\text{idle}} \cdot \mathcal{P}_{\text{idle}}\;}$$

| symbol | meaning | default |
|---|---|---|
| $w_T$ | weight on fleet time | $0.6$ |
| $w_D$ | weight on distance driven | $0.4$ |
| $\lambda_{\text{cap}}$ | capacity violation multiplier | $50.0$ |
| $\lambda_{\text{time}}$ | lateness multiplier | $10.0$ |
| $\lambda_{\text{idle}}$ | idle-vehicle multiplier | $200.0$ |

The objective is therefore **bi-objective**, not pure travel time: $w_T$ and
$w_D$ trade fleet hours against kilometres driven. They live on the problem
instance rather than on a solver, because they are a property of the question
being asked rather than of the method used to answer it. The defaults above are
this project's own — a municipal fleet mostly cares about finishing the round on
time. A published benchmark may score something else entirely; Solomon's CVRPTW
set is judged on total distance alone, and an instance loaded from it sets
$w_T = 0$, $w_D = 1$ so the optimiser answers the question that benchmark
actually asks.

A solution is reported **feasible** when both hard-constraint penalties vanish
to numerical tolerance:

$$\mathcal{P}_{\text{cap}} < 10^{-6} \quad \text{and} \quad \mathcal{P}_{\text{time}} < 10^{-6}$$

Note that $\mathcal{P}_{\text{idle}}$ does not enter this test: an idle van is a
preference, not an infeasibility.

---

## 5. Exact Constraints (Reference MIP Formulation)

1. **Routing and Single Visit**:
   $$\sum_{k=1}^K \sum_{j \in V, j \neq i} x_{ijk} = 1 \quad \forall i \in C$$
2. **Flow Conservation**:
   $$\sum_{j \in V, j \neq p} x_{jpk} - \sum_{j \in V, j \neq p} x_{pjk} = 0 \quad \forall p \in C, \forall k \in \{1, \dots, K\}$$
3. **Depot Departure and Return**:
   $$\sum_{j \in C} x_{0jk} \le 1, \quad \sum_{i \in C} x_{i0k} \le 1 \quad \forall k \in \{1, \dots, K\}$$
4. **Capacity Limits**:
   $$\sum_{i \in C} q_i \sum_{j \in V, j \neq i} x_{ijk} \le Q_k \quad \forall k \in \{1, \dots, K\}$$
5. **Time Window Precedence**:
   $$t_{ik} + s_i + t_{ij} - M(1 - x_{ijk}) \le t_{jk} \quad \forall i \in V, j \in C, i \neq j, \forall k$$
   $$e_i \le t_{ik} \le l_i \quad \forall i \in C, \forall k$$

---

## 6. Quantum-Behaved Particle Swarm Optimization (QPSO)

### Quantum Delta-Potential-Well Mechanics
In classical PSO, a particle moves along Newtonian trajectories with position $x$ and velocity $v$. In QPSO, particles exhibit quantum behavior bound by a delta potential well at the local attractor point $P$.

According to the Schrödinger equation for a particle in a 1D delta potential well:
$$\psi(y) = \frac{1}{\sqrt{L}} e^{-|y|/L}$$
where $L$ is the characteristic length of the potential well. The probability density function of position is:
$$Q(y) = |\psi(y)|^2 = \frac{1}{L} e^{-2|y|/L}$$

Using the inverse transform sampling method with uniform random variable $u \sim U(0, 1)$:
$$y = \pm \frac{L}{2} \ln\left(\frac{1}{u}\right)$$

### Algorithmic Update Rules
For particle $m \in \{1, \dots, M\}$ on dimension $j \in \{1, \dots, N\}$ at iteration $t$:

1. **Mean Best Position ($mbest$)**:
   The center of gravity of all individual personal best positions:
   $$mbest_j(t) = \frac{1}{M} \sum_{m=1}^M pbest_{m,j}(t)$$

2. **Local Attractor ($P$)**:
   A stochastic combination of personal best $pbest_m$ and global swarm best $gbest$:
   $$P_{m,j}(t) = \phi_j(t) \cdot pbest_{m,j}(t) + (1 - \phi_j(t)) \cdot gbest_j(t), \quad \phi_j \sim U(0, 1)$$

3. **Position Update**:
   $$X_{m,j}(t+1) = P_{m,j}(t) \pm \alpha(t) \cdot |mbest_j(t) - X_{m,j}(t)| \cdot \ln\left(\frac{1}{u}\right), \quad u \sim U(0, 1)$$
   The sign $\pm$ is selected with equal probability ($p = 0.5$).

4. **Contraction-Expansion Coefficient ($\alpha$)**:
   Linearly annealed across iterations to balance early global exploration with late local exploitation:
   $$\alpha(t) = \alpha_{\max} - \frac{t}{T_{\max}} (\alpha_{\max} - \alpha_{\min})$$
   Nominal values: $\alpha_{\max} = 1.0, \alpha_{\min} = 0.5$.

### High-Dimensional Instability Fix (Jump-Cap)
As the number of dimensions (customers $N$) increases, the stochastic jump factor $\ln(1/u)$ becomes unbounded as $u \to 0$. In high dimensions, an extreme jump on any single coordinate shatters an otherwise high-quality route structure.

To restore convergence stability at scale, the jump step is bounded:
$$\text{jump} = \text{clip}\left(\ln\left(\frac{1}{u}\right), \; 0, \; \text{jump\_cap}\right)$$
$$\text{jump\_cap} = \frac{\gamma \cdot \text{scale}}{\sqrt{N}}$$
where $\gamma$ is a scaling factor and $\text{scale} = K$. This ensures stable asymptotic convergence for large urban logistics networks ($N \ge 50$).

---

## 7. Memetic Hybridization (Lamarckian Local Search)

While QPSO excels at global exploration through quantum tunneling, fine-grained routing permutations benefit from dedicated local neighborhood operators.

Periodically (every $L_{\text{interval}}$ iterations), the global best chromosome is decoded and refined via:
1. **2-opt Intra-Route Operator**: Reverses path sub-segments $(i, \dots, j)$ within a single vehicle route to untangle crossing paths.
2. **Or-opt Inter-Route Relocation Operator**: Evaluates moving blocks of 1, 2, or 3 consecutive customers from one vehicle route to another whenever travel time or constraint penalties decrease.

The improved discrete solution is encoded back into continuous chromosome space and reinjected into $gbest$ (Lamarckian learning), closing the performance gap against classical heuristics across all fleet scales.

---

## 8. Quantum-Inspired Execution vs. Physical Quantum Hardware Roadmap

A critical design consideration for judges evaluating AICTE's Quantum Technology Vertical:

### Classical Simulation on Standard Hardware
QuantaRoute implements a **quantum-inspired** metaheuristic (QPSO) running on classical CPUs, rather than execution on physical quantum processors (QPUs).
- **Delta-Potential-Well Wave Function**: Particles simulate quantum tunneling through energy barriers using classical random variables sampled from the Schrödinger probability distribution $Q(y) = |\psi(y)|^2$.
- **Immediate Production Viability**: Runs instantly on standard cloud infrastructure (Render, standard VMs, edge nodes) with zero cryogenic hardware requirements or NISQ (Noisy Intermediate-Scale Quantum) decoherence issues.

### Extensibility to Physical Quantum Hardware (Future Roadmap)
While QPSO provides near-optimal routing on classical hardware today, the mathematical formulation is intentionally structured for transition to quantum hardware:
1. **Gate-Based Quantum Processors (QAOA)**: The CVRPTW decision variables ($x_{ijk}$) and penalty formulations can be cast as an Ising Hamiltonian or QUBO (Quadratic Unconstrained Binary Optimization) problem solved via the Quantum Approximate Optimization Algorithm (QAOA) on IBM Quantum / Google Sycamore hardware.
2. **Quantum Annealing**: The customer partitioning and vehicle assignment sub-problem maps directly onto quantum annealers (e.g. D-Wave Advantage) with minor graph embedding.
3. **Hybrid Classical-Quantum Deployment**: Classical QPSO handles real-time dynamic rerouting under traffic incidents, while quantum hardware handles macro-level multi-depot fleet partitioning.
