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
- **Vehicle fleet**: A homogeneous fleet of $K$ vehicles, each having maximum capacity $Q$ (or heterogeneous per-vehicle capacities $Q_k$ for mid-route re-planning).
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

The primary objective is to minimize total fleet travel time and congestion delays, while satisfying vehicle capacity and time window constraints.

To enable the metaheuristic swarm to traverse constraint boundaries smoothly and converge reliably toward the feasible global optimum, constraints are handled via soft penalties:

$$\min \mathcal{F}(X) = T_{\text{total}}(X) + \lambda_{\text{cap}} \cdot \mathcal{P}_{\text{cap}}(X) + \lambda_{\text{time}} \cdot \mathcal{P}_{\text{time}}(X)$$

where:
1. **Total Travel Time**:
   $$T_{\text{total}}(X) = \sum_{k=1}^K \left[ t_{0, c_{k,1}} + \sum_{j=1}^{m_k - 1} t_{c_{k,j}, c_{k,j+1}} + t_{c_{k,m_k}, 0} \right]$$

2. **Capacity Violation Penalty**:
   $$\mathcal{P}_{\text{cap}}(X) = \sum_{k=1}^K \max\left(0, \; \sum_{j=1}^{m_k} q_{c_{k,j}} - Q_k\right)$$
   Weighted by penalty multiplier $\lambda_{\text{cap}} = 50.0$.

3. **Time-Window Lateness Penalty**:
   For each vehicle $k$, departure from depot starts at $t_0 = 0$. For customer $j$ on route $k$:
   $$a_j = t_{\text{prev}} + t_{\text{prev}, j} \quad \text{(arrival time)}$$
   $$\text{start}_j = \max(a_j, e_j) \quad \text{(service start after potential wait)}$$
   $$t_j = \text{start}_j + s_j \quad \text{(departure time after service)}$$
   $$\text{Lateness}_j = \max(0, \; a_j - l_j)$$

   $$\mathcal{P}_{\text{time}}(X) = \sum_{j \in C} \text{Lateness}_j$$
   Weighted by penalty multiplier $\lambda_{\text{time}} = 10.0$.

A solution is strictly **feasible** if and only if $\mathcal{P}_{\text{cap}}(X) = 0$ and $\mathcal{P}_{\text{time}}(X) = 0$.

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
