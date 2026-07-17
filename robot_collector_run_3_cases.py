

## Build Env.

# Case 1 + MDP_destroy
"""

# ============================================================
#
# Policy 1: "original"   (clean positive, pick_gold negative, movement small negative)
# Policy 2: "close"      (if Q_destroy(s,a) > 0 -> reward = -1 else reward = original)
# Policy 3: "legibility" (dual MILP: discourage -Q_destroy, enforce CLEAN_WATER)
#
# All policies are solved with PuLP (dual/occupancy style),
# and policy 3 uses  MILP structure.
# ============================================================

import numpy as np
import pulp
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional

Coord = Tuple[int, int]

# ============================================================
# 1) ENVIRONMENT core)
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
    start_pos: Optional[Coord] = None
    move_actions: Tuple[str, ...] = ("N", "E", "S", "W")
    verbose: bool = False

ARROW_DEFAULT = {"N": "↑", "E": "→", "S": "↓", "W": "←"}

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
                        self.R[si, ai, sj] = self.cfg.living_reward
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

    # value iteration (only used to compute Q_destroy for cases 2 & 3)
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

    def print_policy_grid(self, policy: np.ndarray, title: Optional[str] = None, bits: Optional[int] = None):
        if bits is None:
            bits = self.initial_bits
        if title:
            print(f"\n{title}")
        for r in range(self.ROWS, 0, -1):
            row = []
            for c in range(1, self.COLS + 1):
                if (r, c) in self.WALLS:
                    row.append("  # ")
                else:
                    si = self.idx[(r, c, bits)]
                    a = self.ACTIONS[int(policy[si])]
                    sym = self.ARROW.get(a, a[0])
                    row.append("  " + sym + "  ")
            print(" ".join(row))

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

# ============================================================
# 2) HELPERS: expected reward + dual occupancy LP
# ============================================================

def expected_r_sa(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    # r[s,a] = sum_{s'} P[s,a,s'] * R[s,a,s']
    return np.einsum("sas,sas->sa", P, R)

def solve_dual_occupancy_policy(P, R, avail, d0, gamma, solver=None):
    """
      x[si][ai] vars, objective sum x*r
      flow: sum_a x[i,a] - gamma * sum_{j,a} x[j,a] P[j,a,i] = d0[i]
    Returns: x_val, pi
    """
    S, A, _ = P.shape
    r_sa = expected_r_sa(P, R)

    prob = pulp.LpProblem("Dual_Occupancy", pulp.LpMaximize)

    x = [[None for _ in range(A)] for _ in range(S)]
    for si in range(S):
        for ai in range(A):
            if avail[si, ai]:
                x[si][ai] = pulp.LpVariable(f"x_{si}_{ai}", lowBound=0.0)
            else:
                x[si][ai] = None

    def xterm(si, ai):
        return x[si][ai] if x[si][ai] is not None else 0.0

    # objective: sum_{s,a} x(s,a) r(s,a)
    prob += pulp.lpSum(float(r_sa[si, ai]) * xterm(si, ai) for si in range(S) for ai in range(A))

    # flow constraints
    for i in range(S):
        inflow = pulp.lpSum(xterm(i, ai) for ai in range(A))
        outflow = pulp.lpSum(
            xterm(j, aj) * float(P[j, aj, i])
            for j in range(S) for aj in range(A)
            if avail[j, aj] and P[j, aj, i] != 0.0
        )
        prob += inflow - gamma * outflow == float(d0[i])

    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)
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
            # fallback: first available
            fb = 0
            for ai in range(A):
                if avail[si, ai]:
                    fb = ai
                    break
            pi[si] = fb

    return x_val, pi

# ============================================================
# 3) BUILD GRID ONCE + BUILD 3 MDP REWARD VERSIONS
# ============================================================

# ----- geometry (same as you) -----
# row 5: a5, c5, d5 ; rows 6–7: d6, d7   (a,b,c,d,e,f -> 1..6)
walls = {
    (1, 5), (2, 5), (3, 5), (3, 6), (3, 8),
    (11, 5), (10, 5), (9, 5), (9, 6), (9, 8)
}
doors = [
    DoorSpec(position=(9, 7), initial_open=True),  # GateA
    DoorSpec(position=(3, 7), initial_open=True),  # GateB
]
# One toggle per gate, both at (6,6), and close_only=True.
switches = [
    SwitchSpec(
        action_name="TOGGLE_D0",      # lever A (closes GateA)
        locations={(6, 6)},
        door_index=0,
        reward=0.0,
        close_only=True
    ),
    SwitchSpec(
        action_name="TOGGLE_D1",      # lever B (closes GateB)
        locations={(6, 6)},
        door_index=1,
        reward=0.0,
        close_only=True
    ),
]

START_POS = (1, 2)   # b1
GAMMA = 0.9
NOISE = 0.0

def _all_bits_in_mdp(mdp: GridDoorMDP):
    # infer all bits that exist from mdp.idx keys
    return sorted({bits for (_, _, bits) in mdp.idx.keys()})

def print_state_indices_all_bits(mdp: GridDoorMDP, title_prefix="si indices"):
    for bits in _all_bits_in_mdp(mdp):
        print_state_indices_grid(mdp, bits=bits, title=f"{title_prefix} | bits={bits}")

def print_policy_all_bits(mdp: GridDoorMDP, policy: np.ndarray, title_prefix="policy"):
    for bits in _all_bits_in_mdp(mdp):
        mdp.print_policy_grid(policy, title=f"{title_prefix} | bits={bits}", bits=bits)

def simulate_policy_mode(mdp: GridDoorMDP, policy: np.ndarray, start_pos, start_bits, max_steps=25):
    """
    Debug: follow policy deterministically by taking the most-likely transition (argmax over P).
    This shows how bits change after TOGGLE and what action is chosen next.
    """
    si = mdp.idx[(start_pos[0], start_pos[1], start_bits)]

    for t in range(max_steps):
        ai = int(policy[si])
        a  = mdp.ACTIONS[ai]
        probs = mdp.P[si, ai]

        if probs.sum() <= 0:
            print(f"t={t:02d}: terminal/no transitions at si={si}, state={mdp.STATES[si]}")
            break

        si2 = int(np.argmax(probs))
        r = float(mdp.R[si, ai])

        print(f"t={t:02d}  state={mdp.STATES[si]}  action={a:>12s}  r={r:7.2f}  ->  state'={mdp.STATES[si2]}")
        si = si2

# ============================================================
# CASE 1: "original"
# - CLEAN_WATER: +10
# - PICK_GOLD:  -10
# - movement:   -1 (living_reward)
# - toggle:     0 (already in switch spec; keep it 0)
# ============================================================

def build_mdp_original():

        #  Local special actions:
    # PICK_GOLD becomes "mail pickup" (from (2,7))
    # CLEAN_WATER becomes "trash pickup" (from (10,7))
    MAIL_REWARD_FOR_Q = 10.0
    localspecs = [
        LocalSpecialActionSpec(
        action_name="PICK_MAIL",
        locations={(2, 7)},          # can pick mail if at (2,7)
        reward=MAIL_REWARD_FOR_Q     # keep the same pattern: Q "likes" it
    ),
    LocalSpecialActionSpec(
        action_name="COLLECT_TRASH",
        locations={(10, 7)},         # can pick trash if at (10,7)
        reward=10.0
    ),
    ]
    cfg = GridDoorMDPConfig(
        rows=11, cols=8, walls=walls,
        doors=doors, switches=switches, local_special_actions=localspecs,
        gamma=GAMMA, noise=NOISE, living_reward=-1.0, start_pos=START_POS
    )
    return GridDoorMDP(cfg)

# ============================================================
# For CASE 2 and CASE 3 we need Q_destroy computed from a "destroy" reward shaping:
# In your setting: make PICK_GOLD attractive (+10), everything else 0 (except moves still have living_reward maybe 0)
# Set movement 0 in destroy, CLEAN_WATER 0, TOGGLE 0, PICK_GOLD +10
# ============================================================

def build_mdp_destroy_for_Q():
    #  Local special actions:
    # PICK_GOLD becomes "mail pickup" (from (2,7))
    # CLEAN_WATER becomes "trash pickup" (from (10,7))
    MAIL_REWARD_FOR_Q = 10.0

    localspecs = [
        LocalSpecialActionSpec(
        action_name="PICK_MAIL",
        locations={(2, 7)},          # can pick mail if at (2,7)
        reward=MAIL_REWARD_FOR_Q     # keep the same pattern: Q "likes" it
    ),
    LocalSpecialActionSpec(
        action_name="COLLECT_TRASH",
        locations={(10, 7)},         # can pick trash if at (10,7)
        reward=0.0
    ),
    ]
    cfg = GridDoorMDPConfig(
        rows=11, cols=8, walls=walls,
        doors=doors, switches = [
    SwitchSpec(
        action_name="TOGGLE_D0",      # lever A (closes GateA)
        locations={(6, 6)},
        door_index=0,
        reward=0.0,
        close_only=True
    ),
    SwitchSpec(
        action_name="TOGGLE_D1",      # lever B (closes GateB)
        locations={(6, 6)},
        door_index=1,
        reward=0.0,
        close_only=True
    ),
]
, local_special_actions=localspecs,
        gamma=GAMMA, noise=NOISE, living_reward=0.0, start_pos=START_POS
    )
    return GridDoorMDP(cfg)

"""# CASE 2"""

# ============================================================
# CASE 2: "close"
# Reward_close(s,a,*) = -1  if Q_destroy(s,a) > 0
#                      = original reward otherwise
#
# ============================================================

def build_close_reward_mdp(mdp_original: GridDoorMDP, Q_destroy: np.ndarray, cutoff=0.0):
    mdp_close = mdp_original  # copy arrays
    # Deep copy P/R/avail/STATES etc. into a new object is annoying; easiest:
    # build a fresh original mdp and then overwrite its R with modified rewards.
    mdp_close = build_mdp_original()

    S, A, _ = mdp_close.P.shape
    R_close = mdp_close.R.copy()

    ai_clean = mdp_close.ACTIONS.index("CLEAN_WATER")  # adjust name if different

    for si in range(S):
      for ai in range(A):
          if not mdp_close.avail[si, ai]:
              continue

          nonzero_next = np.where(mdp_close.P[si, ai, :] > 0)[0]

          # --- 1) BOOST CLEAN so it appears in the optimal policy ---
          if ai == ai_clean:
              for sj in nonzero_next:
                  R_close[si, ai, sj] = 10.0   # pick a value > the cost of reaching the clean cell
              continue  # don't apply destroy-penalty logic to clean

          # --- 2) Your existing "avoid destroy" penalty ---
          if Q_destroy[si, ai] > cutoff:
              for sj in nonzero_next:
                  R_close[si, ai, sj] = -1.0


    mdp_close.R = R_close
    return mdp_close


def solve_close_dual_occupancy_policy(P, R_orig, avail, d0, gamma, Q_destroy, cutoff=0.0, penalty_reward=-1.0, solver=None):
    """
    Case 2: Solve a *separate* dual occupancy LP where the objective uses:
        r_close(s,a) = penalty_reward  if Q_destroy(s,a) > cutoff
                      r_orig(s,a)     otherwise
    No need to build mdp_close or modify any reward tensor.
    Returns: x_val, pi
    """
    S, A, _ = P.shape
    # d0 = np.ones(S, dtype=float) / S


    # expected one-step reward under original rewards
    r_orig_sa = expected_r_sa(P, R_orig)   # shape (S,A)

    # build r_close(s,a)
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

    # objective: sum_{s,a} x(s,a) * r_close(s,a)
    prob += pulp.lpSum(float(r_close_sa[si, ai]) * xterm(si, ai)
                       for si in range(S) for ai in range(A))

    # flow constraints: sum_a x(i,a) - gamma * sum_{j,a} x(j,a) P(j,a,i) = d0(i)
    for i in range(S):
        inflow = pulp.lpSum(xterm(i, ai) for ai in range(A))
        outflow = pulp.lpSum(
            xterm(j, aj) * float(P[j, aj, i])
            for j in range(S) for aj in range(A)
            if avail[j, aj] and P[j, aj, i] != 0.0
        )
        prob += inflow - gamma * outflow == float(d0[i])

    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)

    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Close dual LP not optimal. Status={pulp.LpStatus[status]}")

    # extract occupancy
    x_val = np.zeros((S, A), dtype=float)
    for si in range(S):
        for ai in range(A):
            if avail[si, ai]:
                x_val[si, ai] = float(x[si][ai].value() or 0.0)

    #  policy from occupancy
    pi = np.zeros(S, dtype=int)
    for si in range(S):
        if x_val[si].sum() > 0:
            pi[si] = int(np.argmax(x_val[si]))
        else:
            # fallback: first available action
            for ai in range(A):
                if avail[si, ai]:
                    pi[si] = ai
                    break

    return x_val, pi

"""# Case 3 (MILP)"""

# ============================================================
# CASE 3: "legibility" (your special MILP)
# - compute Q_destroy from destroy mdp (already)
# - then run the MILP that discourages -Q_destroy and enforces CLEAN_WATER via x2 constraint
# ============================================================

def solve_legibility_milp_policy(mdp_for_P: GridDoorMDP, Q_destroy: np.ndarray,
                                TAU=100.0, POS_REWARD=200.0, GAMMA2=0.9,
                                pos_action_name="COLLECT_TRASH",
                                solver=None):
    """
    mdp_for_P: use the transitions/avail/STATES/ACTIONS  ( original mdp)
    Q_destroy: Q array aligned with mdp_for_P state/action ordering
    pos_action_name: the positive action to encourage (default: COLLECT_TRASH)
    """
    S = mdp_for_P.S
    A = mdp_for_P.A
    P = mdp_for_P.P
    avail = mdp_for_P.avail
    STATES = mdp_for_P.STATES
    ACTIONS = mdp_for_P.ACTIONS
    idx = mdp_for_P.idx

    # start-only d1 for x2
    d1 = np.zeros(S, dtype=float)
    si_start = idx[(START_POS[0], START_POS[1], mdp_for_P.initial_bits)]
    d1[si_start] = 1.0

    # uniform d0 for x1 (as in your code)
    d0 = np.ones(S, dtype=float) / S

    # identify action index for positive action
    ai_pos = ACTIONS.index(pos_action_name)

    # find locations where positive action is available from the MDP config
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

    # (C5) water constraint
    pos_reward_expr = pulp.lpSum(
        x2[si][ai] * float(R_pos[si, ai])
        for si in range(S) for ai in range(A)
        if avail[si, ai] and R_pos[si, ai] != 0.0
    )
    prob += pos_reward_expr >= float(TAU)

    # objective: discourage high Q_destroy via -Q
    prob += pulp.lpSum(
        x1[si][ai] * float(-Q_destroy[si, ai])
        for si in range(S) for ai in range(A)
        if avail[si, ai]
    )

    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=600)
    status = prob.solve(solver)
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
            # fallback
            for ai in range(A):
                if avail[si, ai]:
                    chosen = ai
                    break
        pi[si] = int(chosen)

    return pi, float(pulp.value(pos_reward_expr) or 0.0), float(pulp.value(prob.objective) or 0.0)

"""#Run Cases"""

# ============================================================
# 4) RUN ALL 3 CASES
# ============================================================

# Build original mdp (case 1 rewards)
mdp_orig = build_mdp_original()

# Print si mapping (useful debugging)
print_state_indices_grid(mdp_orig, bits=mdp_orig.initial_bits, title="si indices (door OPEN slice)")

# --------------------------
# CASE 1: ORIGINAL policy via dual occupancy
# --------------------------
x1_case1, pi_case1 = solve_dual_occupancy_policy(
    P=mdp_orig.P, R=mdp_orig.R, avail=mdp_orig.avail, d0=mdp_orig.d0, gamma=mdp_orig.gamma
)
mdp_orig.print_policy_grid(pi_case1, title="CASE 1 (original): dual occupancy policy", bits=mdp_orig.initial_bits)

# --------------------------
# Compute Q_destroy (needed for case 2 and 3)
# --------------------------
mdp_destroy = build_mdp_destroy_for_Q()
Vd, pd, Q_destroy = mdp_destroy.value_iteration(max_iters=400, tol=1e-10)

# --------------------------
# CASE 2: CLOSE policy (SEPARATE dual occupancy, no mdp_close)
# --------------------------
x_case2, pi_case2 = solve_close_dual_occupancy_policy(
    P=mdp_orig.P,
    R_orig=mdp_orig.R,
    avail=mdp_orig.avail,
    d0=mdp_orig.d0,
    gamma=mdp_orig.gamma,
    Q_destroy=Q_destroy ,
    cutoff=0.0,
    penalty_reward=-1.0,
    solver=pulp.PULP_CBC_CMD(msg=False)
)

print_policy_all_bits(mdp_orig, pi_case2, title_prefix="CASE 2 (close): separate dual occupancy policy")


si_start = mdp_orig.idx[(START_POS[0], START_POS[1], mdp_orig.initial_bits)]

# --------------------------
# CASE 3: LEGIBILITY policy (your special MILP)
# Use transitions/avail of mdp_orig; Q_destroy is aligned because same state/action ordering in our builds
# --------------------------
pi_case3, water_total, obj_val = solve_legibility_milp_policy(
    mdp_for_P=mdp_orig,
    Q_destroy=Q_destroy,
    TAU=10.0,
    POS_REWARD=200.0,
    GAMMA2=0.8,
    pos_action_name="COLLECT_TRASH",
    solver=pulp.PULP_CBC_CMD(msg=True, timeLimit=600)
)
print_policy_all_bits(mdp_orig, pi_case3, title_prefix="CASE 3 (legibility): special MILP policy")


print("\nCASE 3 diagnostics:")
print("  Total CLEAN_WATER reward (x2 * R_pos):", water_total)
print("  Objective (sum x1 * -Q_destroy):      ", obj_val)

# ============================================================
# 5) (Optional) show chosen actions at start for each case
# ============================================================

bits0 = mdp_orig.initial_bits

print("\nStart state si =", si_start, "state =", (START_POS[0], START_POS[1], bits0))
print("  CASE1 action:", mdp_orig.ACTIONS[int(pi_case1[si_start])])
print("  CASE2 action:", mdp_orig.ACTIONS[int(pi_case2[si_start])])
print("  CASE3 action:", mdp_orig.ACTIONS[int(pi_case3[si_start])])

Q_destroy

"""| bits | binary | door 0 (9,7) | door 1 (3,7) | meaning           |
| ---- | ------ | ------------ | ------------ | ----------------- |
| 0    | `00`   | closed       | closed       | both doors closed |
| 1    | `01`   | open         | closed       | only GateA open   |
| 2    | `10`   | closed       | open         | only GateB open   |
| 3    | `11`   | open         | open         | both doors open   |

"""

print("orig A =", mdp_orig.A, mdp_orig.ACTIONS)
print("dest A =", mdp_destroy.A, mdp_destroy.ACTIONS)

print("orig locals :", mdp_orig.local_action_names)
print("dest locals :", mdp_destroy.local_action_names)

print("orig switches:", mdp_orig.switch_action_names)
print("dest switches:", mdp_destroy.switch_action_names)

"""## Helpers to check the details"""

def expected_r_sa(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    # r[s,a] = sum_{s'} P[s,a,s'] * R[s,a,s']
    return np.einsum("sas,sas->sa", P, R)

def build_close_reward_table(P, R_orig, avail, Q_for_close, cutoff=0.0, penalty_reward=-1.0):
    """
    Returns:
      r_orig_sa : expected one-step reward under original R
      r_close_sa: same, but clamped to penalty_reward when Q_for_close(s,a) > cutoff
    """
    r_orig_sa = expected_r_sa(P, R_orig)
    r_close_sa = r_orig_sa.copy()
    mask = (Q_for_close > cutoff) & avail
    r_close_sa[mask] = float(penalty_reward)
    return r_orig_sa, r_close_sa


def print_x_for_state(mdp: GridDoorMDP,
                      x_val: np.ndarray,
                      si: int,
                      Q: np.ndarray,
                      r_orig_sa: np.ndarray,
                      r_close_sa: np.ndarray,
                      title: str = None,
                      eps: float = 1e-12):
    """
    Prints actions in mdp.ACTIONS order (no sorting) with:
      x(s,a), x/sum, Q(s,a), r_orig(s,a), r_close(s,a)
    """
    if title:
        print(f"\n{title}")

    r, c, bits = mdp.STATES[si]

    # sum_x over available actions
    x_sum = 0.0
    for ai in range(mdp.A):
        if mdp.avail[si, ai]:
            x_sum += float(x_val[si, ai])

    print(f"State si={si}  (r={r}, c={c}, bits={bits})   sum_x={x_sum:.6g}")

    # chosen action = argmax x over available actions (no sorting)
    best_ai = None
    best_x = -1.0
    for ai in range(mdp.A):
        if not mdp.avail[si, ai]:
            continue
        xv = float(x_val[si, ai])
        if xv > best_x:
            best_x = xv
            best_ai = ai

    if best_ai is None:
        print("  (no available actions)")
        return

    if best_x > eps:
        print(f"  chosen (argmax x): ai={best_ai} action={mdp.ACTIONS[best_ai]}  x={best_x:.6g}")
    else:
        print("  chosen (argmax x): (all x≈0 here; your policy extraction will fall back if you coded it that way)")

    header = f"{'ai':>3s} {'action':14s} {'x(s,a)':>12s} {'x/sum':>10s} {'Q(s,a)':>12s} {'r_orig':>10s} {'r_close':>10s}"
    print(header)
    print("-" * len(header))

    for ai, a in enumerate(mdp.ACTIONS):
        if not mdp.avail[si, ai]:
            continue
        xv = float(x_val[si, ai])
        frac = (xv / x_sum) if x_sum > eps else 0.0
        qv = float(Q[si, ai])
        ro = float(r_orig_sa[si, ai])
        rc = float(r_close_sa[si, ai])
        print(f"{ai:3d} {a:14s} {xv:12.6g} {frac:10.4f} {qv:12.6g} {ro:10.6g} {rc:10.6g}")
def si_of(mdp: GridDoorMDP, coord, bits=None):
    if bits is None:
        bits = mdp.initial_bits
    r, c = coord
    return mdp.idx[(r, c, bits)]

def prnt_x_for_key_states(mdp: GridDoorMDP,
                          x_val: np.ndarray,
                          Q: np.ndarray,
                          r_orig_sa: np.ndarray,
                          r_close_sa: np.ndarray,
                          name_to_coord: dict,
                          bits=None):
    if bits is None:
        bits = mdp.initial_bits
    for name, coord in name_to_coord.items():
        si = si_of(mdp, coord, bits=bits)
        print_x_for_state(mdp, x_val, si, Q=Q, r_orig_sa=r_orig_sa, r_close_sa=r_close_sa, title=name)

# Build reward tables for printing (Case 2 shaping uses Q_destroy thresholding)
r_orig_sa, r_close_sa = build_close_reward_table(
    P=mdp_orig.P,
    R_orig=mdp_orig.R,
    avail=mdp_orig.avail,
    Q_for_close=Q_destroy,
    cutoff=0.0,
    penalty_reward=-1.0
)

key_states = {
    "START": START_POS,
    "TOGGLE (3,2)": (3, 2),
    "DOOR (5,2)": (5, 2),
    "GOLD (6,2)": (6, 2),
    "WATER (6,6)": (6, 6),
}

prnt_x_for_key_states(
    mdp=mdp_orig,
    x_val=x_case2,
    Q=Q_destroy,
    r_orig_sa=r_orig_sa,
    r_close_sa=r_close_sa,
    name_to_coord=key_states,
    bits=mdp_orig.initial_bits
)

