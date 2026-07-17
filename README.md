# Running Experiments

## Dependencies

```bash
pip install numpy pulp
```

## Files

- **`mdp_core.py`** — Core MDP classes (`GridDoorMDP`, `GridDoorMDPConfig`, etc.) and solvers (dual occupancy LP, close policy, legibility MILP). Shared library used by all other scripts.
- **`experiment_domains.py`** — Defines the 4 domain types x 2 sizes = 8 domain instances used in the experiments.
- **`run_experiments.py`** — Main experiment runner. Runs all domains x 3 policy cases x tau sweep and writes results to `results/`.
- **`robot_cleaner_run_3_cases.py`** — Standalone script for the Robot Cleaner domain (single 7x6 grid, 3 policies).
- **`robot_collector_run_3_cases.py`** — Standalone script for the Robot Collector domain (single grid, 3 policies).

## Running

**Full experiment suite** (all domains, sizes, tau values):

```bash
python run_experiments.py
```

Results are saved to `results/`.

**Single domain scripts** (for quick testing):

```bash
python robot_cleaner_run_3_cases.py
python robot_collector_run_3_cases.py
```

