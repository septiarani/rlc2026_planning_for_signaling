# experiment_domains.py — Domain definitions for experiments
# 4 domain types x 2 sizes = 8 domain instances

from mdp_core import (
    GridDoorMDP, GridDoorMDPConfig, DoorSpec, SwitchSpec,
    LocalSpecialActionSpec, Policy2Config, Policy3Config
)


def _make_domain(name, size_label, cfg_orig, cfg_destroy,
                 pos_action_name, neg_action_name,
                 policy2_cfg=None, gamma2=0.9, pos_reward=10.0):
    """Helper to build a domain config dict."""
    if policy2_cfg is None:
        policy2_cfg = Policy2Config(cutoff=0.0, penalty_reward=-1.0)
    return {
        "name": f"{name}_{size_label}",
        "display_name": f"{name} ({size_label})",
        "size_label": size_label,
        "domain_type": name,
        "build_mdp_original": lambda c=cfg_orig: GridDoorMDP(c),
        "build_mdp_destroy": lambda c=cfg_destroy: GridDoorMDP(c),
        "pos_action_name": pos_action_name,
        "neg_action_name": neg_action_name,
        "policy2_cfg": policy2_cfg,
        "gamma2": gamma2,
        "pos_reward": pos_reward,
        "grid_size": f"{cfg_orig.rows}x{cfg_orig.cols}",
    }


# ============================================================
# Domain 1: PuddleGrid (no toggle)
# ============================================================

def puddle_grid_small():
    """PuddleGrid 6x6 — no toggle, puddle cells with -1 penalty."""
    walls = {(4, 2), (5, 4), (3, 5)}
    pos_loc = {(6, 6)}
    neg_loc = {(3, 1)}
    start = (1, 1)
    puddles = {(2, 3): -1.0, (2, 4): -1.0, (3, 3): -1.0,
               (4, 4): -1.0, (5, 3): -1.0}

    cfg_orig = GridDoorMDPConfig(
        rows=6, cols=6, walls=walls,
        local_special_actions=[
            LocalSpecialActionSpec("FALL_PIT", neg_loc, reward=-10.0),
            LocalSpecialActionSpec("REACH_GOAL", pos_loc, reward=+10.0),
        ],
        gamma=0.9, noise=0.0, living_reward=0.0, cell_rewards=puddles,
        start_pos=start,
    )
    cfg_destroy = GridDoorMDPConfig(
        rows=6, cols=6, walls=walls,
        local_special_actions=[
            LocalSpecialActionSpec("FALL_PIT", neg_loc, reward=+10.0),
            LocalSpecialActionSpec("REACH_GOAL", pos_loc, reward=0.0),
        ],
        gamma=0.9, noise=0.0, living_reward=0.0, start_pos=start,
    )
    return _make_domain("puddle_grid", "small", cfg_orig, cfg_destroy,
                        "REACH_GOAL", "FALL_PIT", gamma2=0.9)


def puddle_grid_large():
    """PuddleGrid 10x10 — no toggle, puddle cells with -1 penalty."""
    walls = {(5, 2), (7, 3), (9, 6), (7, 8), (5, 9), (4, 5), (3, 7)}
    pos_loc = {(10, 10)}
    neg_loc = {(3, 1)}
    start = (1, 1)
    puddles = {(2, 4): -1.0, (3, 4): -1.0, (3, 5): -1.0, (4, 3): -1.0,
               (6, 4): -1.0, (6, 7): -1.0, (7, 5): -1.0, (8, 5): -1.0,
               (8, 6): -1.0}

    cfg_orig = GridDoorMDPConfig(
        rows=10, cols=10, walls=walls,
        local_special_actions=[
            LocalSpecialActionSpec("FALL_PIT", neg_loc, reward=-10.0),
            LocalSpecialActionSpec("REACH_GOAL", pos_loc, reward=+10.0),
        ],
        gamma=0.9, noise=0.0, living_reward=0.0, cell_rewards=puddles,
        start_pos=start,
    )
    cfg_destroy = GridDoorMDPConfig(
        rows=10, cols=10, walls=walls,
        local_special_actions=[
            LocalSpecialActionSpec("FALL_PIT", neg_loc, reward=+10.0),
            LocalSpecialActionSpec("REACH_GOAL", pos_loc, reward=0.0),
        ],
        gamma=0.9, noise=0.0, living_reward=0.0, start_pos=start,
    )
    return _make_domain("puddle_grid", "large", cfg_orig, cfg_destroy,
                        "REACH_GOAL", "FALL_PIT", gamma2=0.9)


# ============================================================
# Domain 2: SimpleGrid (no toggle)
# ============================================================

def simple_grid_small():
    """SimpleGrid 5x5 — no toggle."""
    walls = {(4, 2), (3, 4)}
    pos_loc = {(5, 5)}
    neg_loc = {(5, 2)}
    start = (1, 1)

    cfg_orig = GridDoorMDPConfig(
        rows=5, cols=5, walls=walls,
        local_special_actions=[
            LocalSpecialActionSpec("TRAP", neg_loc, reward=-10.0),
            LocalSpecialActionSpec("COLLECT", pos_loc, reward=+10.0),
        ],
        gamma=0.9, noise=0.0, living_reward=-1.0, start_pos=start,
    )
    cfg_destroy = GridDoorMDPConfig(
        rows=5, cols=5, walls=walls,
        local_special_actions=[
            LocalSpecialActionSpec("TRAP", neg_loc, reward=+10.0),
            LocalSpecialActionSpec("COLLECT", pos_loc, reward=0.0),
        ],
        gamma=0.9, noise=0.0, living_reward=0.0, start_pos=start,
    )
    return _make_domain("simple_grid", "small", cfg_orig, cfg_destroy,
                        "COLLECT", "TRAP", gamma2=0.9)


def simple_grid_large():
    """SimpleGrid 9x9 — no toggle."""
    walls = {(5, 2), (8, 4), (7, 6), (5, 8), (4, 5)}
    pos_loc = {(9, 9)}
    neg_loc = {(9, 3)}
    start = (1, 1)

    cfg_orig = GridDoorMDPConfig(
        rows=9, cols=9, walls=walls,
        local_special_actions=[
            LocalSpecialActionSpec("TRAP", neg_loc, reward=-10.0),
            LocalSpecialActionSpec("COLLECT", pos_loc, reward=+10.0),
        ],
        gamma=0.9, noise=0.0, living_reward=-1.0, start_pos=start,
    )
    cfg_destroy = GridDoorMDPConfig(
        rows=9, cols=9, walls=walls,
        local_special_actions=[
            LocalSpecialActionSpec("TRAP", neg_loc, reward=+10.0),
            LocalSpecialActionSpec("COLLECT", pos_loc, reward=0.0),
        ],
        gamma=0.9, noise=0.0, living_reward=0.0, start_pos=start,
    )
    return _make_domain("simple_grid", "large", cfg_orig, cfg_destroy,
                        "COLLECT", "TRAP", gamma2=0.9)


# ============================================================
# Domain 3: SafeRoom (1 toggle)
# ============================================================

def safe_room_small():
    """SafeRoom 7x5 — 1 door, 1 toggle (close_only)."""
    walls = {(5, 1), (5, 3), (5, 4), (5, 5)}
    doors = [DoorSpec(position=(5, 2), initial_open=True)]
    switches = [SwitchSpec(action_name="TOGGLE_D0", locations={(3, 2)},
                           door_index=0, reward=0.0, close_only=True)]
    pos_loc = {(2, 5)}
    neg_loc = {(6, 2)}
    start = (1, 2)

    cfg_orig = GridDoorMDPConfig(
        rows=7, cols=5, walls=walls, doors=doors, switches=switches,
        local_special_actions=[
            LocalSpecialActionSpec("TOUCH_WIRE", neg_loc, reward=-10.0),
            LocalSpecialActionSpec("GET_PACKAGE", pos_loc, reward=+10.0),
        ],
        gamma=0.9, noise=0.0, living_reward=-1.0, start_pos=start,
    )
    cfg_destroy = GridDoorMDPConfig(
        rows=7, cols=5, walls=walls, doors=doors,
        switches=[SwitchSpec(action_name="TOGGLE_D0", locations={(3, 2)},
                             door_index=0, reward=0.0, close_only=True)],
        local_special_actions=[
            LocalSpecialActionSpec("TOUCH_WIRE", neg_loc, reward=+10.0),
            LocalSpecialActionSpec("GET_PACKAGE", pos_loc, reward=0.0),
        ],
        gamma=0.9, noise=0.0, living_reward=0.0, start_pos=start,
    )
    return _make_domain("safe_room", "small", cfg_orig, cfg_destroy,
                        "GET_PACKAGE", "TOUCH_WIRE", gamma2=0.9)


def safe_room_large():
    """SafeRoom 12x8 — 1 door, 1 toggle (close_only)."""
    walls = {(8, 1), (8, 2), (8, 3), (8, 5), (8, 6), (8, 7), (8, 8),
             (5, 2), (6, 6)}
    doors = [DoorSpec(position=(8, 4), initial_open=True)]
    switches = [SwitchSpec(action_name="TOGGLE_D0", locations={(4, 4)},
                           door_index=0, reward=0.0, close_only=True)]
    pos_loc = {(2, 8)}
    neg_loc = {(10, 3)}
    start = (1, 1)

    cfg_orig = GridDoorMDPConfig(
        rows=12, cols=8, walls=walls, doors=doors, switches=switches,
        local_special_actions=[
            LocalSpecialActionSpec("TOUCH_WIRE", neg_loc, reward=-10.0),
            LocalSpecialActionSpec("GET_PACKAGE", pos_loc, reward=+10.0),
        ],
        gamma=0.9, noise=0.0, living_reward=-1.0, start_pos=start,
    )
    cfg_destroy = GridDoorMDPConfig(
        rows=12, cols=8, walls=walls, doors=doors,
        switches=[SwitchSpec(action_name="TOGGLE_D0", locations={(4, 4)},
                             door_index=0, reward=0.0, close_only=True)],
        local_special_actions=[
            LocalSpecialActionSpec("TOUCH_WIRE", neg_loc, reward=+10.0),
            LocalSpecialActionSpec("GET_PACKAGE", pos_loc, reward=0.0),
        ],
        gamma=0.9, noise=0.0, living_reward=0.0, start_pos=start,
    )
    return _make_domain("safe_room", "large", cfg_orig, cfg_destroy,
                        "GET_PACKAGE", "TOUCH_WIRE", gamma2=0.9)


# ============================================================
# Domain 4: Corridor (1 toggle)
# ============================================================

def corridor_small():
    """Corridor 8x5 — 1 door, 1 toggle (close_only), debris cells with -1 penalty."""
    walls = {(6, 1), (6, 2), (6, 4), (6, 5)}
    doors = [DoorSpec(position=(6, 3), initial_open=True)]
    switches = [SwitchSpec(action_name="TOGGLE_D0", locations={(4, 3)},
                           door_index=0, reward=0.0, close_only=True)]
    pos_loc = {(2, 5)}
    neg_loc = {(7, 3)}
    start = (1, 1)
    debris = {(3, 2): -1.0, (3, 4): -1.0, (5, 2): -1.0, (5, 4): -1.0}

    cfg_orig = GridDoorMDPConfig(
        rows=8, cols=5, walls=walls, doors=doors, switches=switches,
        local_special_actions=[
            LocalSpecialActionSpec("SPILL_CHEMICAL", neg_loc, reward=-10.0),
            LocalSpecialActionSpec("CLEAN_AREA", pos_loc, reward=+10.0),
        ],
        gamma=0.9, noise=0.0, living_reward=0.0, cell_rewards=debris,
        start_pos=start,
    )
    cfg_destroy = GridDoorMDPConfig(
        rows=8, cols=5, walls=walls, doors=doors,
        switches=[SwitchSpec(action_name="TOGGLE_D0", locations={(4, 3)},
                             door_index=0, reward=0.0, close_only=True)],
        local_special_actions=[
            LocalSpecialActionSpec("SPILL_CHEMICAL", neg_loc, reward=+10.0),
            LocalSpecialActionSpec("CLEAN_AREA", pos_loc, reward=0.0),
        ],
        gamma=0.9, noise=0.0, living_reward=0.0, start_pos=start,
    )
    return _make_domain("corridor", "small", cfg_orig, cfg_destroy,
                        "CLEAN_AREA", "SPILL_CHEMICAL", gamma2=0.9)


def corridor_large():
    """Corridor 14x8 — 1 door, 1 toggle (close_only), debris cells with -1 penalty."""
    walls = {(10, 1), (10, 2), (10, 3), (10, 4), (10, 6), (10, 7), (10, 8),
             (7, 6), (6, 3)}
    doors = [DoorSpec(position=(10, 5), initial_open=True)]
    switches = [SwitchSpec(action_name="TOGGLE_D0", locations={(5, 5)},
                           door_index=0, reward=0.0, close_only=True)]
    pos_loc = {(2, 8)}
    neg_loc = {(12, 4)}
    start = (1, 1)
    debris = {(3, 3): -1.0, (4, 6): -1.0, (7, 3): -1.0, (8, 6): -1.0,
              (9, 3): -1.0, (9, 7): -1.0}

    cfg_orig = GridDoorMDPConfig(
        rows=14, cols=8, walls=walls, doors=doors, switches=switches,
        local_special_actions=[
            LocalSpecialActionSpec("SPILL_CHEMICAL", neg_loc, reward=-10.0),
            LocalSpecialActionSpec("CLEAN_AREA", pos_loc, reward=+10.0),
        ],
        gamma=0.9, noise=0.0, living_reward=0.0, cell_rewards=debris,
        start_pos=start,
    )
    cfg_destroy = GridDoorMDPConfig(
        rows=14, cols=8, walls=walls, doors=doors,
        switches=[SwitchSpec(action_name="TOGGLE_D0", locations={(5, 5)},
                             door_index=0, reward=0.0, close_only=True)],
        local_special_actions=[
            LocalSpecialActionSpec("SPILL_CHEMICAL", neg_loc, reward=+10.0),
            LocalSpecialActionSpec("CLEAN_AREA", pos_loc, reward=0.0),
        ],
        gamma=0.9, noise=0.0, living_reward=0.0, start_pos=start,
    )
    return _make_domain("corridor", "large", cfg_orig, cfg_destroy,
                        "CLEAN_AREA", "SPILL_CHEMICAL", gamma2=0.9)


# ============================================================
# Registry of all domains
# ============================================================

ALL_DOMAINS = [
    puddle_grid_small,
    puddle_grid_large,
    simple_grid_small,
    simple_grid_large,
    safe_room_small,
    safe_room_large,
    corridor_small,
    corridor_large,
]


def get_all_domains():
    """Return list of all domain config dicts."""
    return [fn() for fn in ALL_DOMAINS]
