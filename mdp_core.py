# mdp_core.py — Core MDP classes and solvers
# Extracted from robot_cleaner_run_3_cases.py for reuse across experiments.

import numpy as np
import pulp
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional

Coord = Tuple[int, int]

# ============================================================
# Dataclasses
# ============================================================

@dataclass
class DoorSpec:
    position: Coord
    initial_open: bool = True

@dataclass
class SwitchSpec:
    action_name: str
    locations: Set[Coord]
    door_index: int
    reward: float = 0.0
    close_only: bool = False

@dataclass
class LocalSpecialActionSpec:
    action_name: str
    locations: Set[Coord]
    reward: float = 0.0

@dataclass
class GridDoorMDPConfig:
    rows: int
    cols: int
    walls: Set[Coord] = field(default_factory=set)
    doors: List[DoorSpec] = field(default_factory=list)
    switches: List[SwitchSpec] = field(default_factory=list)
    local_special_actions: List[LocalSpecialActionSpec] = field(default_factory=list)
    gamma: float = 0.9
    noise: float = 0.0
    living_reward: float = 0.0
    cell_rewards: Dict[Coord, float] = field(default_factory=dict)  # per-cell move reward override
    start_pos: Optional[Coord] = None
    move_actions: Tuple[str, ...] = ("N", "E", "S", "W")
    verbose: bool = False

@dataclass
class Policy2Config:
    """Configuration for Policy 2 (close) - Q_destroy-based reward shaping."""
    cutoff: float = 0.0          # Q_destroy(s,a) > cutoff triggers penalty
    penalty_reward: float = -1.0  # reward assigned when Q_destroy(s,a) > cutoff

@dataclass
class Policy3Config:
    """Configuration for Policy 3 (legibility) - MILP approach."""
    tau: float = 6.0
    pos_reward: float = 10.0
    gamma2: float = 0.9
    time_limit: int = 600

ARROW_DEFAULT = {"N": "↑", "E": "→", "S": "↓", "W": "←"}

# ============================================================
# GridDoorMDP
# ============================================================

class GridDoorMDP:
    """
    State = (row, col, door_bits)
    Actions = moves + local specials + switches.
    """

    def __init__(self, cfg: GridDoorMDPConfig):
        self.cfg = cfg
        self.ROWS = cfg.rows
        self.COLS = cfg.cols
        self.WALLS = cfg.walls
        self.doors = cfg.doors
        self.switches = cfg.switches
        self.local_special_actions = cfg.local_special_actions
        self.gamma = cfg.gamma

        # door bits
        self.num_doors = len(self.doors)
        self.door_positions = [d.position for d in self.doors]
        self.initial_bits = 0
        for j, d in enumerate(self.doors):
            if d.initial_open:
                self.initial_bits |= (1 << j)

        # base cells
        self.cells: List[Coord] = [
            (r, c)
            for r in range(1, self.ROWS + 1)
            for c in range(1, self.COLS + 1)
            if (r, c) not in self.WALLS
        ]

        # full state space
        self.STATES: List[Tuple[int, int, int]] = []
        for bits in range(1 << self.num_doors):
            for (r, c) in self.cells:
                self.STATES.append((r, c, bits))
        self.S = len(self.STATES)

        # actions
        self.move_actions = list(cfg.move_actions)
        self.local_action_names = [a.action_name for a in self.local_special_actions]
        self.switch_action_names = [s.action_name for s in self.switches]
        self.ACTIONS: List[str] = (
            self.move_actions + self.local_action_names + self.switch_action_names
        )
        self.A = len(self.ACTIONS)

        # indices
        self.idx: Dict[Tuple[int, int, int], int] = {s: i for i, s in enumerate(self.STATES)}

        # movement
        self.DELTA = {"N": (1, 0), "E": (0, 1), "S": (-1, 0), "W": (0, -1)}
        self.LEFT  = {"N": "W", "E": "N", "S": "E", "W": "S"}
        self.RIGHT = {"N": "E", "E": "S", "S": "W", "W": "N"}

        # start dist
        self.d0 = np.zeros(self.S, dtype=float)
        if cfg.start_pos is not None:
            start_state = (cfg.start_pos[0], cfg.start_pos[1], self.initial_bits)
            self.d0[self.idx[start_state]] = 1.0

        # transitions
        self.P = np.zeros((self.S, self.A, self.S), dtype=float)
        self.R = np.zeros((self.S, self.A, self.S), dtype=float)
        self.avail = np.zeros((self.S, self.A), dtype=bool)
        self._build_dynamics()

        # symbols
        self.ARROW = dict(ARROW_DEFAULT)
        for a in self.local_action_names + self.switch_action_names:
            if a not in self.ARROW:
                self.ARROW[a] = a[0]

    def in_bounds(self, r: int, c: int) -> bool:
        return 1 <= r <= self.ROWS and 1 <= c <= self.COLS

    def is_blocked(self, cell: Coord, bits: int) -> bool:
        if cell in self.WALLS:
            return True
        for j, pos in enumerate(self.door_positions):
            if cell == pos:
                open_flag = (bits >> j) & 1
                return open_flag == 0
        return False

    def step_cell(self, r: int, c: int, a: str, bits: int) -> Coord:
        dr, dc = self.DELTA[a]
        nr, nc = r + dr, c + dc
        if (not self.in_bounds(nr, nc)) or self.is_blocked((nr, nc), bits):
            return (r, c)
        return (nr, nc)

    def _build_dynamics(self):
        INTENT = 1.0 - self.cfg.noise
        SLIP = self.cfg.noise / 2.0

        switch_by_action: Dict[str, List[SwitchSpec]] = {}
        for s in self.switches:
            switch_by_action.setdefault(s.action_name, []).append(s)
        local_by_action: Dict[str, LocalSpecialActionSpec] = {
            a.action_name: a for a in self.local_special_actions
        }

        for (r, c, bits) in self.STATES:
            si = self.idx[(r, c, bits)]
            for ai, a in enumerate(self.ACTIONS):

                # moves
                if a in self.move_actions:
                    moves = [
                        (INTENT, self.step_cell(r, c, a, bits)),
                        (SLIP,   self.step_cell(r, c, self.LEFT[a], bits)),
                        (SLIP,   self.step_cell(r, c, self.RIGHT[a], bits)),
                    ]
                    agg: Dict[Coord, float] = {}
                    for p_prob, (nr, nc) in moves:
                        agg[(nr, nc)] = agg.get((nr, nc), 0.0) + p_prob
                    for (nr, nc), p_prob in agg.items():
                        sj = self.idx[(nr, nc, bits)]
                        self.P[si, ai, sj] += p_prob
                        self.R[si, ai, sj] = self.cfg.cell_rewards.get(
                            (nr, nc), self.cfg.living_reward)
                    self.avail[si, ai] = True

                # local specials
                elif a in local_by_action:
                    spec = local_by_action[a]
                    if (r, c) in spec.locations:
                        sj = si
                        self.P[si, ai, sj] = 1.0
                        self.R[si, ai, sj] = spec.reward
                        self.avail[si, ai] = True

                # switches
                elif a in switch_by_action:
                    applicable = [sw for sw in switch_by_action[a] if (r, c) in sw.locations]
                    if not applicable:
                        continue
                    sw = applicable[0]
                    mask = 1 << sw.door_index
                    door_open = (bits & mask) != 0

                    if sw.close_only:
                        if not door_open:
                            continue
                        new_bits = bits & ~mask
                    else:
                        new_bits = bits ^ mask

                    sj = self.idx[(r, c, new_bits)]
                    self.P[si, ai, sj] = 1.0
                    self.R[si, ai, sj] = sw.reward
                    self.avail[si, ai] = True

    def value_iteration(self, max_iters: int = 500, tol: float = 1e-10):
        V = np.zeros(self.S, dtype=float)
        for _ in range(max_iters):
            V_new = V.copy()
            for si in range(self.S):
                best = -1e100
                for ai in range(self.A):
                    if not self.avail[si, ai]:
                        continue
                    q = np.dot(self.P[si, ai, :], self.R[si, ai, :] + self.gamma * V)
                    if q > best:
                        best = q
                V_new[si] = best
            if np.max(np.abs(V_new - V)) < tol:
                V = V_new
                break
            V = V_new

        policy = np.zeros(self.S, dtype=int)
        Q = np.full((self.S, self.A), -1e100, dtype=float)
        for si in range(self.S):
            best_ai = 0
            best = -1e100
            for ai in range(self.A):
                if not self.avail[si, ai]:
                    continue
                q = np.dot(self.P[si, ai, :], self.R[si, ai, :] + self.gamma * V)
                Q[si, ai] = q
                if q > best:
                    best = q
                    best_ai = ai
            policy[si] = best_ai
        return V, policy, Q

# ============================================================
# Helpers
# ============================================================

def expected_r_sa(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.einsum("sas,sas->sa", P, R)

# ============================================================
# Case 1: Dual Occupancy LP
# ============================================================

def solve_dual_occupancy_policy(P, R, avail, d0, gamma, solver=None):
    S, A, _ = P.shape
    r_sa = expected_r_sa(P, R)

    prob = pulp.LpProblem("Dual_Occupancy", pulp.LpMaximize)

    x = [[None for _ in range(A)] for _ in range(S)]
    for si in range(S):
        for ai in range(A):
            if avail[si, ai]:
                x[si][ai] = pulp.LpVariable(f"x_{si}_{ai}", lowBound=0.0)

    def xterm(si, ai):
        return x[si][ai] if x[si][ai] is not None else 0.0

    prob += pulp.lpSum(float(r_sa[si, ai]) * xterm(si, ai) for si in range(S) for ai in range(A))

    for i in range(S):
        inflow = pulp.lpSum(xterm(i, ai) for ai in range(A))
        outflow = pulp.lpSum(
            xterm(j, aj) * float(P[j, aj, i])
            for j in range(S) for aj in range(A)
            if avail[j, aj] and P[j, aj, i] != 0.0
        )
        prob += inflow - gamma * outflow == float(d0[i])

    if solver is None:
        solver = pulp.HiGHS(msg=False)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Dual LP not optimal. Status={pulp.LpStatus[status]}")

    x_val = np.zeros((S, A), dtype=float)
    for si in range(S):
        for ai in range(A):
            if avail[si, ai]:
                x_val[si, ai] = float(x[si][ai].value() or 0.0)

    pi = np.zeros(S, dtype=int)
    for si in range(S):
        if x_val[si].sum() > 0:
            pi[si] = int(np.argmax(x_val[si]))
        else:
            for ai in range(A):
                if avail[si, ai]:
                    pi[si] = ai
                    break

    return x_val, pi

# ============================================================
# Case 2: Safety-shaped Dual LP
# ============================================================

def solve_close_dual_occupancy_policy(P, R_orig, avail, d0, gamma, Q_destroy,
                                     cutoff=0.0, penalty_reward=-1.0, solver=None):
    """
    Policy 2 (original algorithm): Q_destroy(s,a)-based reward shaping.
    If Q_destroy(s,a) > cutoff, set r_close(s,a) = penalty_reward.
    Otherwise keep original expected reward.
    """
    S, A, _ = P.shape
    r_orig_sa = expected_r_sa(P, R_orig)
    r_close_sa = r_orig_sa.copy()

    mask = (Q_destroy > cutoff) & avail
    r_close_sa[mask] = float(penalty_reward)

    prob = pulp.LpProblem("Dual_Occupancy_Close", pulp.LpMaximize)

    x = [[None for _ in range(A)] for _ in range(S)]
    for si in range(S):
        for ai in range(A):
            if avail[si, ai]:
                x[si][ai] = pulp.LpVariable(f"x_{si}_{ai}", lowBound=0.0)

    def xterm(si, ai):
        return x[si][ai] if x[si][ai] is not None else 0.0

    prob += pulp.lpSum(float(r_close_sa[si, ai]) * xterm(si, ai)
                       for si in range(S) for ai in range(A))

    for i in range(S):
        inflow = pulp.lpSum(xterm(i, ai) for ai in range(A))
        outflow = pulp.lpSum(
            xterm(j, aj) * float(P[j, aj, i])
            for j in range(S) for aj in range(A)
            if avail[j, aj] and P[j, aj, i] != 0.0
        )
        prob += inflow - gamma * outflow == float(d0[i])

    if solver is None:
        solver = pulp.HiGHS(msg=False)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Close dual LP not optimal. Status={pulp.LpStatus[status]}")

    x_val = np.zeros((S, A), dtype=float)
    for si in range(S):
        for ai in range(A):
            if avail[si, ai]:
                x_val[si, ai] = float(x[si][ai].value() or 0.0)

    pi = np.zeros(S, dtype=int)
    for si in range(S):
        if x_val[si].sum() > 0:
            pi[si] = int(np.argmax(x_val[si]))
        else:
            for ai in range(A):
                if avail[si, ai]:
                    pi[si] = ai
                    break

    return x_val, pi

# ============================================================
# Case 3: Legibility MILP (generalized)
# ============================================================

def get_pulp_solver_info(prob):
    """Extract solver status info directly from PuLP problem object after solve()."""
    info = {}

    # Problem status: Not Solved / Optimal / Infeasible / Unbounded / Undefined
    info["lp_status"] = pulp.LpStatus.get(prob.status, f"Unknown({prob.status})")
    info["lp_status_code"] = prob.status

    # Solution status: more detailed than problem status
    # 1=Optimal Solution Found, 2=Solution Found (integer feasible),
    # 0=No Solution Found, -1=No Solution Exists, -2=Unbounded
    sol_status_map = pulp.constants.LpSolution
    info["sol_status"] = sol_status_map.get(prob.sol_status, f"Unknown({prob.sol_status})")
    info["sol_status_code"] = prob.sol_status

    # Solver metadata
    info["solver_name"] = prob.solver.name if prob.solver else "None"
    info["solution_time"] = getattr(prob, "solutionTime", None)
    info["num_variables"] = prob.numVariables()
    info["num_constraints"] = prob.numConstraints()
    info["objective_value"] = float(pulp.value(prob.objective) or 0.0)

    return info


def parse_solver_log(log_path):
    """Parse solver log file (CBC or HiGHS) for additional diagnostics."""
    import re
    info = {"first_feasible_time": None, "best_solution_time": None,
            "gap": None, "bound": None, "nodes": None, "log_tail": ""}
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return info

    best_obj_time = None
    for line in lines:
        # CBC: "Integer solution of X found after Y iterations..."
        if "Integer solution of" in line and info["first_feasible_time"] is None:
            m = re.search(r'\(([0-9.]+) seconds\)', line)
            if m:
                info["first_feasible_time"] = float(m.group(1))
        # CBC: track best solution updates
        if "best solution" in line:
            m = re.search(r'\(([0-9.]+) seconds\)', line)
            if m:
                best_obj_time = float(m.group(1))
        # HiGHS: "Solving report" section or solution lines
        if "Primal bound" in line:
            m = re.search(r'Primal bound\s*:\s*([-0-9.e+]+)', line)
            if m:
                try:
                    info["bound"] = float(m.group(1))
                except ValueError:
                    pass
        if "Gap" in line and "%" in line:
            m = re.search(r'Gap\s*:\s*([\d.]+\s*%)', line)
            if m:
                info["gap"] = m.group(1).strip()
        if "Nodes" in line:
            m = re.search(r'Nodes\s*:\s*(\d+)', line)
            if m:
                info["nodes"] = int(m.group(1))
        # CBC Gap/nodes
        if line.startswith("Gap:") or (": Gap:" in line):
            info["gap"] = line.split(":")[-1].strip()
        if "Enumerated nodes:" in line:
            m = re.search(r'(\d+)', line.split(":")[-1])
            if m:
                info["nodes"] = int(m.group(1))

    if best_obj_time is not None:
        info["best_solution_time"] = best_obj_time

    # Keep last 25 lines as log tail
    info["log_tail"] = "".join(lines[-25:])
    return info


def solve_legibility_milp_policy(mdp_for_P: GridDoorMDP, Q_destroy: np.ndarray,
                                TAU=6.0, POS_REWARD=10.0, GAMMA2=0.9,
                                pos_action_name=None, solver=None, log_path=None):
    """
    Generalized MILP solver for Case 3 (legibility).
    pos_action_name: name of the positive action (auto-detected if None — uses last local special action).
    log_path: if provided, CBC log is written here and parsed for diagnostics.
    Returns: (policy, water_total, obj_val, solver_info_dict)
    """
    S = mdp_for_P.S
    A = mdp_for_P.A
    P = mdp_for_P.P
    avail = mdp_for_P.avail
    STATES = mdp_for_P.STATES
    ACTIONS = mdp_for_P.ACTIONS
    idx = mdp_for_P.idx

    # Auto-detect positive action
    if pos_action_name is None:
        # Use the last local special action as positive (convention: first is negative, last is positive)
        pos_action_name = mdp_for_P.local_special_actions[-1].action_name

    # start-only d1 for x2
    start_pos = mdp_for_P.cfg.start_pos
    d1 = np.zeros(S, dtype=float)
    si_start = idx[(start_pos[0], start_pos[1], mdp_for_P.initial_bits)]
    d1[si_start] = 1.0

    # uniform d0 for x1
    d0 = np.ones(S, dtype=float) / S

    # identify action index for positive action
    ai_pos = ACTIONS.index(pos_action_name)

    # find locations where positive action is available
    pos_locations = set()
    for spec in mdp_for_P.local_special_actions:
        if spec.action_name == pos_action_name:
            pos_locations = spec.locations
            break

    # R_pos for positive action at its locations
    R_pos = np.zeros((S, A), dtype=float)
    for si, (r, c, bits) in enumerate(STATES):
        if (r, c) in pos_locations and avail[si, ai_pos]:
            R_pos[si, ai_pos] = POS_REWARD

    prob = pulp.LpProblem("MILP_legibility", pulp.LpMaximize)

    # y binary
    y = [[None] * A for _ in range(S)]
    for si in range(S):
        for ai in range(A):
            if avail[si, ai]:
                y[si][ai] = pulp.LpVariable(f"y_{si}_{ai}", lowBound=0, upBound=1, cat="Binary")

    # x1, x2
    x1 = [[None] * A for _ in range(S)]
    x2 = [[None] * A for _ in range(S)]
    for si in range(S):
        for ai in range(A):
            if avail[si, ai]:
                x1[si][ai] = pulp.LpVariable(f"x1_{si}_{ai}", lowBound=0.0)
                x2[si][ai] = pulp.LpVariable(f"x2_{si}_{ai}", lowBound=0.0)

    # (C1) deterministic policy
    for si in range(S):
        prob += pulp.lpSum(y[si][ai] for ai in range(A) if avail[si, ai]) == 1

    # (C2) x1 flow (gamma=0): sum_a x1(i,a) = d0(i)
    for i in range(S):
        prob += pulp.lpSum(x1[i][ai] for ai in range(A) if avail[i, ai]) == float(d0[i])

    # (C3) x2 flow (discounted)
    for i in range(S):
        out_i = pulp.lpSum(x2[i][ai] for ai in range(A) if avail[i, ai])
        in_i = pulp.lpSum(
            x2[j][aj] * float(P[j, aj, i])
            for j in range(S) for aj in range(A)
            if avail[j, aj] and P[j, aj, i] != 0.0
        )
        prob += out_i - GAMMA2 * in_i == float(d1[i])

    # (C4) linkage
    for si in range(S):
        for ai in range(A):
            if not avail[si, ai]:
                continue
            prob += x1[si][ai] <= y[si][ai]
            prob += x2[si][ai] <= y[si][ai]

    # (C5) positive reward constraint
    pos_reward_expr = pulp.lpSum(
        x2[si][ai] * float(R_pos[si, ai])
        for si in range(S) for ai in range(A)
        if avail[si, ai] and R_pos[si, ai] != 0.0
    )
    prob += pos_reward_expr >= float(TAU)

    # objective: discourage high Q_destroy
    prob += pulp.lpSum(
        x1[si][ai] * float(-Q_destroy[si, ai])
        for si in range(S) for ai in range(A)
        if avail[si, ai]
    )

    # Solver setup with log capture
    solver_log_text = ""
    if solver is None:
        if log_path:
            solver = pulp.HiGHS(msg=False, timeLimit=300, gapRel=0.05, logPath=log_path)
        else:
            solver = pulp.HiGHS(msg=False, timeLimit=300, gapRel=0.05)

    status = prob.solve(solver)

    if log_path:
        try:
            with open(log_path, "r") as f:
                solver_log_text = f.read()
        except FileNotFoundError:
            pass

    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Legibility MILP not optimal. Status={pulp.LpStatus[status]}")

    # extract y -> policy
    y_val = np.zeros((S, A), dtype=float)
    for si in range(S):
        for ai in range(A):
            if avail[si, ai]:
                y_val[si, ai] = float(y[si][ai].value() or 0.0)

    pi = np.zeros(S, dtype=int)
    for si in range(S):
        chosen = None
        for ai in range(A):
            if avail[si, ai] and y_val[si, ai] > 0.5:
                chosen = ai
                break
        if chosen is None:
            for ai in range(A):
                if avail[si, ai]:
                    chosen = ai
                    break
        pi[si] = int(chosen)

    water_total = float(pulp.value(pos_reward_expr) or 0.0)
    obj_val = float(pulp.value(prob.objective) or 0.0)

    # Build solver info from PuLP problem object (always reliable)
    solver_info = get_pulp_solver_info(prob)
    solver_info["log"] = solver_log_text

    # Supplement with log-parsed diagnostics (gap, nodes, etc.)
    if log_path:
        log_info = parse_solver_log(log_path)
        for k, v in log_info.items():
            if v is not None and v != "":
                solver_info.setdefault(k, v)

    return pi, water_total, obj_val, solver_info

# ============================================================
# Visualization helpers
# ============================================================

def format_environment_state(mdp: GridDoorMDP, bits=None, title=None,
                             pos_action_name=None, neg_action_name=None):
    """Format environment state as a string with grid layout and feature summary."""
    lines = []
    if bits is None:
        bits = mdp.initial_bits

    # Auto-detect action names
    if pos_action_name is None and len(mdp.local_special_actions) >= 2:
        pos_action_name = mdp.local_special_actions[-1].action_name
    if neg_action_name is None and len(mdp.local_special_actions) >= 2:
        neg_action_name = mdp.local_special_actions[0].action_name

    if title:
        lines.append(title)

    # Feature summary
    lines.append(f"Grid: {mdp.ROWS}x{mdp.COLS}  |  gamma={mdp.gamma}  |  "
                 f"living_reward={mdp.cfg.living_reward}")
    if mdp.cfg.cell_rewards:
        cr_val = next(iter(mdp.cfg.cell_rewards.values()))
        lines.append(f"Hazard cells (~): {len(mdp.cfg.cell_rewards)} cells with reward={cr_val}")
    if mdp.cfg.start_pos:
        lines.append(f"Start: {mdp.cfg.start_pos}")
    for spec in mdp.local_special_actions:
        locs = sorted(spec.locations)
        lines.append(f"Action '{spec.action_name}' at {locs}: reward={spec.reward}")
    if mdp.doors:
        for j, d in enumerate(mdp.doors):
            lines.append(f"Door {j} at {d.position}: {'open' if d.initial_open else 'closed'}")
    for sw in mdp.switches:
        locs = sorted(sw.locations)
        mode = "close_only" if sw.close_only else "toggle"
        lines.append(f"Switch '{sw.action_name}' at {locs}: door {sw.door_index} ({mode})")

    # Grid layout
    lines.append("")
    symbols = {}
    for r in range(1, mdp.ROWS + 1):
        for c in range(1, mdp.COLS + 1):
            if (r, c) in mdp.WALLS:
                symbols[(r, c)] = ' # '
            elif (r, c) in mdp.cfg.cell_rewards:
                symbols[(r, c)] = ' ~ '
            else:
                symbols[(r, c)] = ' . '

    for j, door_pos in enumerate(mdp.door_positions):
        is_open = (bits >> j) & 1
        symbols[door_pos] = ' D ' if is_open else ' d '

    for sw in mdp.switches:
        for loc in sw.locations:
            symbols[loc] = ' T '

    for spec in mdp.local_special_actions:
        for loc in spec.locations:
            if spec.action_name == neg_action_name:
                symbols[loc] = ' N '
            elif spec.action_name == pos_action_name:
                symbols[loc] = ' P '

    if mdp.cfg.start_pos:
        symbols[mdp.cfg.start_pos] = ' S '

    for r in range(mdp.ROWS, 0, -1):
        row = []
        for c in range(1, mdp.COLS + 1):
            row.append(symbols.get((r, c), '   '))
        lines.append(''.join(row) + f'  row {r}')
    lines.append('  col: ' + ''.join(f' {c} ' for c in range(1, mdp.COLS + 1)))

    # Legend
    lines.append("Legend: S=start P=positive N=negative #=wall D=door(open) "
                 "d=door(closed) T=toggle ~=puddle .=empty")

    return "\n".join(lines)


def print_environment_state(mdp: GridDoorMDP, bits=None, title=None,
                           pos_action_name=None, neg_action_name=None):
    print("\n" + format_environment_state(mdp, bits, title, pos_action_name, neg_action_name))


def print_full_policy_grid(mdp: GridDoorMDP, policy: np.ndarray, title: str = None,
                          show_both_door_states: bool = True):
    if title:
        print(f"\n{'='*60}")
        print(f"{title}")
        print('='*60)

    if mdp.num_doors > 0 and show_both_door_states:
        door_states = [(1, "OPEN"), (0, "CLOSED")]
    else:
        door_states = [(mdp.initial_bits, "initial")]

    for bits, door_label in door_states:
        print(f"\nPolicy Grid (door {door_label}):")
        for row in range(mdp.ROWS, 0, -1):
            row_str = []
            for col in range(1, mdp.COLS + 1):
                if (row, col) in mdp.WALLS:
                    row_str.append(" # ")
                elif (row, col, bits) in mdp.idx:
                    si = mdp.idx[(row, col, bits)]
                    action = mdp.ACTIONS[int(policy[si])]
                    sym = mdp.ARROW.get(action, action[0])
                    row_str.append(f" {sym} ")
                else:
                    row_str.append(" . ")
            print("".join(row_str) + f"  row {row}")
        print("col:" + "".join(f" {c} " for c in range(1, mdp.COLS + 1)))


def simulate_trajectory(mdp: GridDoorMDP, policy: np.ndarray,
                       pos_action_name=None, neg_action_name=None,
                       max_steps: int = 50):
    """
    Simulate a trajectory from start state following the policy.
    Returns trajectory list and cumulative discounted return.
    """
    if pos_action_name is None and len(mdp.local_special_actions) >= 2:
        pos_action_name = mdp.local_special_actions[-1].action_name
    if neg_action_name is None and len(mdp.local_special_actions) >= 2:
        neg_action_name = mdp.local_special_actions[0].action_name

    terminal_actions = set()
    if pos_action_name:
        terminal_actions.add(pos_action_name)
    if neg_action_name:
        terminal_actions.add(neg_action_name)

    r_sa = expected_r_sa(mdp.P, mdp.R)

    start_pos = mdp.cfg.start_pos
    r, c, bits = start_pos[0], start_pos[1], mdp.initial_bits
    si = mdp.idx[(r, c, bits)]

    trajectory = []
    visited_si = set()
    total_return = 0.0
    discount = 1.0
    terminal_reason = "max_steps"

    for step in range(max_steps):
        if si in visited_si:
            terminal_reason = "loop"
            break
        visited_si.add(si)

        ai = int(policy[si])
        action = mdp.ACTIONS[ai]
        reward = r_sa[si, ai]
        total_return += discount * reward
        discount *= mdp.gamma

        trajectory.append((r, c, bits, action, reward))

        if action in terminal_actions:
            terminal_reason = f"terminal:{action}"
            break

        # Execute action
        if action in ["N", "E", "S", "W"]:
            nr, nc = mdp.step_cell(r, c, action, bits)
            r, c = nr, nc
        elif action.startswith("TOGGLE"):
            # Toggle the appropriate door
            for sw in mdp.switches:
                if sw.action_name == action and (r, c) in sw.locations:
                    mask = 1 << sw.door_index
                    if sw.close_only:
                        bits = bits & ~mask
                    else:
                        bits = bits ^ mask
                    break

        si = mdp.idx[(r, c, bits)]

    return trajectory, total_return, terminal_reason


def format_trajectory_grid(mdp: GridDoorMDP, trajectory, pos_action_name=None,
                          neg_action_name=None):
    """Format trajectory as a grid string with arrows on visited cells."""
    position_action = {}
    for (r, c, bits, action, reward) in trajectory:
        if (r, c) not in position_action:
            position_action[(r, c)] = action

    lines = []
    for row in range(mdp.ROWS, 0, -1):
        row_str = []
        for col in range(1, mdp.COLS + 1):
            if (row, col) in mdp.WALLS:
                row_str.append(" # ")
            elif (row, col) in position_action:
                action = position_action[(row, col)]
                sym = mdp.ARROW.get(action, action[0])
                row_str.append(f" {sym} ")
            else:
                row_str.append(" . ")
        lines.append("".join(row_str) + f"  row {row}")
    lines.append("col:" + "".join(f" {c} " for c in range(1, mdp.COLS + 1)))
    return "\n".join(lines)


def print_policy_with_trajectory(mdp: GridDoorMDP, policy: np.ndarray,
                                title: str = None, max_steps: int = 50,
                                pos_action_name=None, neg_action_name=None):
    """
    Print trajectory + grid visualization. Generalized version.
    """
    if pos_action_name is None and len(mdp.local_special_actions) >= 2:
        pos_action_name = mdp.local_special_actions[-1].action_name
    if neg_action_name is None and len(mdp.local_special_actions) >= 2:
        neg_action_name = mdp.local_special_actions[0].action_name

    if title:
        print(f"\n{'='*60}")
        print(f"{title}")
        print('='*60)

    trajectory, total_return, terminal_reason = simulate_trajectory(
        mdp, policy, pos_action_name, neg_action_name, max_steps
    )

    print("\nTrajectory:")
    for step, (r, c, bits, action, reward) in enumerate(trajectory):
        door_str = "OPEN" if bits else "CLOSED" if mdp.num_doors > 0 else "N/A"
        print(f"  Step {step}: ({r},{c}) door={door_str} -> {action} (r={reward:.2f})")
        if action.startswith("TOGGLE"):
            print(f"       -> Door toggled")

    print(f"\n  Terminal: {terminal_reason}")
    print(f"  Cumulative return: {total_return:.4f}")

    print("\nTrajectory Grid:")
    print(format_trajectory_grid(mdp, trajectory, pos_action_name, neg_action_name))

    return trajectory, total_return, terminal_reason


def print_state_indices_grid(mdp: GridDoorMDP, bits=None, title=None):
    if bits is None:
        bits = mdp.initial_bits
    if title:
        print(f"\n{title}")
    for r in range(mdp.ROWS, 0, -1):
        row = []
        for c in range(1, mdp.COLS + 1):
            if (r, c) in mdp.WALLS:
                row.append("   ###   ")
            else:
                si = mdp.idx[(r, c, bits)]
                row.append(f"{si:7d}")
        print(" ".join(row))
