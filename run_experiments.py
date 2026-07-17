#!/usr/bin/env python3
"""
run_experiments.py — Experiment runner for 4 domains x 2 sizes x 3 cases x tau sweep.

Usage:
    python run_experiments.py

Output:
    results/<domain_name>/  — per-run .txt files
    results/summary.txt     — aggregated table
"""

import os
import sys
import csv
import time
import numpy as np
import pulp

from mdp_core import (
    solve_dual_occupancy_policy,
    solve_close_dual_occupancy_policy,
    solve_legibility_milp_policy,
    expected_r_sa,
    simulate_trajectory,
    format_trajectory_grid,
    format_environment_state,
    print_environment_state,
    print_full_policy_grid,
    print_policy_with_trajectory,
    parse_solver_log,
)
from experiment_domains import get_all_domains

# ============================================================
# Configuration
# ============================================================

TAU_VALUES = [0, 4, 8]
NUM_REPS = 3
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ============================================================
# Helpers
# ============================================================

def format_run_output(domain_cfg, case_label, tau, rep, mdp, policy, trajectory,
                      total_return, terminal_reason, solver_time, q_destroy_time,
                      extra_info=None):
    """Format a single run's output as a string for saving to file."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Domain: {domain_cfg['display_name']}  ({domain_cfg['grid_size']})")
    lines.append(f"Case: {case_label}")
    if tau is not None:
        lines.append(f"Tau: {tau}")
    lines.append(f"Repetition: {rep}")
    lines.append("=" * 60)

    lines.append("")
    lines.append("Environment:")
    env_str = format_environment_state(mdp, title=None,
                                       pos_action_name=domain_cfg["pos_action_name"],
                                       neg_action_name=domain_cfg["neg_action_name"])
    lines.append(env_str)

    lines.append("")
    lines.append("Trajectory:")
    for step, (r, c, bits, action, reward) in enumerate(trajectory):
        if mdp.num_doors > 0:
            door_str = "OPEN" if bits else "CLOSED"
        else:
            door_str = "N/A"
        lines.append(f"  Step {step}: ({r},{c}) door={door_str} -> {action} (r={reward:.2f})")

    lines.append(f"\n  Terminal: {terminal_reason}")
    lines.append(f"  Path length: {len(trajectory)} steps")
    lines.append(f"  Cumulative return: {total_return:.4f}")

    lines.append("")
    lines.append("Trajectory Grid:")
    grid_str = format_trajectory_grid(mdp, trajectory,
                                      domain_cfg["pos_action_name"],
                                      domain_cfg["neg_action_name"])
    lines.append(grid_str)

    lines.append("")
    lines.append("Timing:")
    lines.append(f"  Solver time: {solver_time:.4f} s")
    lines.append(f"  Q_destroy time: {q_destroy_time:.4f} s")
    lines.append(f"  Total time: {solver_time + q_destroy_time:.4f} s")

    if extra_info:
        lines.append("")
        for k, v in extra_info.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)


def save_run(output_dir, filename, content):
    """Save run output to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


# ============================================================
# Main experiment loop
# ============================================================

def run_single_domain(domain_cfg, results_base_dir):
    """Run all cases and reps for a single domain. Returns list of result dicts."""
    domain_name = domain_cfg["name"]
    output_dir = os.path.join(results_base_dir, domain_name)
    os.makedirs(output_dir, exist_ok=True)

    pos_action = domain_cfg["pos_action_name"]
    neg_action = domain_cfg["neg_action_name"]
    p2_cfg = domain_cfg["policy2_cfg"]
    gamma2 = domain_cfg["gamma2"]
    pos_reward = domain_cfg["pos_reward"]

    results = []

    print(f"\n{'#'*60}")
    print(f"# Domain: {domain_cfg['display_name']}  ({domain_cfg['grid_size']})")
    print(f"{'#'*60}")

    # Build MDP once (structure is deterministic)
    mdp_orig = domain_cfg["build_mdp_original"]()

    # Print environment layout
    print_environment_state(mdp_orig, pos_action_name=pos_action, neg_action_name=neg_action,
                           title=f"Environment: {domain_cfg['display_name']}")

    for rep in range(NUM_REPS):
        print(f"\n--- Repetition {rep} ---")

        # ============ Case 1: Dual Occupancy LP ============
        t0 = time.time()
        x_val_c1, pi_case1 = solve_dual_occupancy_policy(
            P=mdp_orig.P, R=mdp_orig.R, avail=mdp_orig.avail,
            d0=mdp_orig.d0, gamma=mdp_orig.gamma,
            solver=pulp.HiGHS(msg=False)
        )
        case1_time = time.time() - t0

        traj_c1, ret_c1, term_c1 = simulate_trajectory(
            mdp_orig, pi_case1, pos_action, neg_action
        )

        content = format_run_output(domain_cfg, "Case 1 (Dual LP)", None, rep,
                                    mdp_orig, pi_case1, traj_c1, ret_c1, term_c1,
                                    case1_time, 0.0)
        save_run(output_dir, f"case1_rep{rep}.txt", content)

        results.append({
            "domain": domain_name, "domain_type": domain_cfg["domain_type"],
            "size": domain_cfg["size_label"], "grid_size": domain_cfg["grid_size"],
            "case": "Case1", "tau": None, "rep": rep,
            "solver_time": case1_time, "q_destroy_time": 0.0,
            "total_return": ret_c1, "path_length": len(traj_c1),
            "terminal": term_c1,
            "lp_status": "Optimal", "sol_status": "Optimal Solution Found",
        })

        print(f"  Case 1: return={ret_c1:.4f}, time={case1_time:.3f}s, {term_c1}")

        # ============ Q_destroy computation (timed separately) ============
        t0 = time.time()
        mdp_destroy = domain_cfg["build_mdp_destroy"]()
        V_destroy, _, Q_destroy = mdp_destroy.value_iteration(max_iters=500, tol=1e-10)
        q_destroy_time = time.time() - t0

        print(f"  Q_destroy: time={q_destroy_time:.3f}s, V range=[{V_destroy.min():.2f}, {V_destroy.max():.2f}]")

        # ============ Case 2: Q_destroy-based reward shaping ============
        # Original algorithm: if Q_destroy(s,a) > cutoff, set reward to penalty_reward
        t0 = time.time()
        x_val_c2, pi_case2 = solve_close_dual_occupancy_policy(
            P=mdp_orig.P, R_orig=mdp_orig.R, avail=mdp_orig.avail,
            d0=mdp_orig.d0, gamma=mdp_orig.gamma,
            Q_destroy=Q_destroy,
            cutoff=p2_cfg.cutoff, penalty_reward=p2_cfg.penalty_reward,
            solver=pulp.HiGHS(msg=False)
        )
        case2_time = time.time() - t0

        traj_c2, ret_c2, term_c2 = simulate_trajectory(
            mdp_orig, pi_case2, pos_action, neg_action
        )

        content = format_run_output(domain_cfg, "Case 2 (Q_destroy shaping)", None, rep,
                                    mdp_orig, pi_case2, traj_c2, ret_c2, term_c2,
                                    case2_time, q_destroy_time)
        save_run(output_dir, f"case2_rep{rep}.txt", content)

        results.append({
            "domain": domain_name, "domain_type": domain_cfg["domain_type"],
            "size": domain_cfg["size_label"], "grid_size": domain_cfg["grid_size"],
            "case": "Case2", "tau": None, "rep": rep,
            "solver_time": case2_time, "q_destroy_time": q_destroy_time,
            "total_return": ret_c2, "path_length": len(traj_c2),
            "terminal": term_c2,
            "lp_status": "Optimal", "sol_status": "Optimal Solution Found",
        })

        print(f"  Case 2: return={ret_c2:.4f}, time={case2_time:.3f}s, {term_c2}")

        # ============ Case 3: MILP x tau sweep ============
        for tau in TAU_VALUES:
            log_file = os.path.join(output_dir, f"case3_tau{tau}_rep{rep}_cbc.log")
            t0 = time.time()
            try:
                pi_case3, water_total, obj_val, solver_info = solve_legibility_milp_policy(
                    mdp_for_P=mdp_orig,
                    Q_destroy=Q_destroy,
                    TAU=float(tau),
                    POS_REWARD=pos_reward,
                    GAMMA2=gamma2,
                    pos_action_name=pos_action,
                    log_path=log_file,
                )
                case3_time = time.time() - t0
                feasible = True
            except RuntimeError as e:
                case3_time = time.time() - t0
                feasible = False
                solver_info = {}
                if os.path.exists(log_file):
                    solver_info = parse_solver_log(log_file)
                print(f"  Case 3 tau={tau}: INFEASIBLE ({e})")

            if feasible:
                traj_c3, ret_c3, term_c3 = simulate_trajectory(
                    mdp_orig, pi_case3, pos_action, neg_action
                )

                extra = {"water_total": f"{water_total:.4f}",
                         "objective": f"{obj_val:.4f}"}
                # PuLP-level status (always available)
                extra["lp_status"] = solver_info.get("lp_status", "N/A")
                extra["sol_status"] = solver_info.get("sol_status", "N/A")
                extra["solver_name"] = solver_info.get("solver_name", "N/A")
                extra["num_variables"] = solver_info.get("num_variables", "N/A")
                extra["num_constraints"] = solver_info.get("num_constraints", "N/A")
                if solver_info.get("solution_time") is not None:
                    extra["solution_time"] = f"{solver_info['solution_time']:.4f}s"
                # Log-parsed diagnostics (may not be available)
                if solver_info.get("first_feasible_time") is not None:
                    extra["first_feasible_time"] = f"{solver_info['first_feasible_time']:.2f}s"
                if solver_info.get("best_solution_time") is not None:
                    extra["best_solution_time"] = f"{solver_info['best_solution_time']:.2f}s"
                if solver_info.get("gap"):
                    extra["gap"] = solver_info["gap"]
                if solver_info.get("nodes") is not None:
                    extra["nodes_enumerated"] = solver_info["nodes"]

                content = format_run_output(domain_cfg, f"Case 3 (MILP, tau={tau})",
                                            tau, rep, mdp_orig, pi_case3,
                                            traj_c3, ret_c3, term_c3,
                                            case3_time, q_destroy_time,
                                            extra_info=extra)
                # Append solver log
                log_text = solver_info.get("log", "") or solver_info.get("log_tail", "")
                if log_text:
                    # Keep last 30 lines
                    log_lines = log_text.strip().splitlines()
                    tail = "\n".join(log_lines[-30:])
                    content += "\n\nSolver Log (last 30 lines):\n"
                    content += tail

                save_run(output_dir, f"case3_tau{tau}_rep{rep}.txt", content)

                results.append({
                    "domain": domain_name, "domain_type": domain_cfg["domain_type"],
                    "size": domain_cfg["size_label"], "grid_size": domain_cfg["grid_size"],
                    "case": "Case3", "tau": tau, "rep": rep,
                    "solver_time": case3_time, "q_destroy_time": q_destroy_time,
                    "total_return": ret_c3, "path_length": len(traj_c3),
                    "terminal": term_c3,
                    "lp_status": solver_info.get("lp_status", ""),
                    "sol_status": solver_info.get("sol_status", ""),
                })

                sol_str = solver_info.get("sol_status", "")
                feas_t = solver_info.get("first_feasible_time")
                feas_str = f", 1st_feas={feas_t:.1f}s" if feas_t else ""
                gap_str = f", gap={solver_info['gap']}" if solver_info.get("gap") else ""
                print(f"  Case 3 tau={tau}: return={ret_c3:.4f}, time={case3_time:.3f}s, "
                      f"water={water_total:.2f}, {term_c3}, [{sol_str}]{feas_str}{gap_str}")
            else:
                results.append({
                    "domain": domain_name, "domain_type": domain_cfg["domain_type"],
                    "size": domain_cfg["size_label"], "grid_size": domain_cfg["grid_size"],
                    "case": "Case3", "tau": tau, "rep": rep,
                    "solver_time": case3_time, "q_destroy_time": q_destroy_time,
                    "total_return": float("nan"), "path_length": 0,
                    "terminal": "infeasible",
                    "lp_status": "Infeasible", "sol_status": "No Solution Exists",
                })

    return results


def write_summary(all_results, results_base_dir):
    """Write aggregated summary table."""
    # Group by (domain, case, tau)
    groups = {}
    for r in all_results:
        key = (r["domain"], r["case"], r["tau"])
        groups.setdefault(key, []).append(r)

    lines = []
    lines.append("=" * 120)
    lines.append("EXPERIMENT SUMMARY")
    lines.append("=" * 120)
    lines.append("")

    header = (f"{'Domain':<25} {'Size':<8} {'Case':<8} {'Tau':<6} "
              f"{'Solver Time':>18} {'Q_destroy Time':>16} "
              f"{'Return':>18} {'Path Len':>10} {'Terminal':<20}")
    lines.append(header)
    lines.append("-" * 120)

    for key in sorted(groups.keys()):
        runs = groups[key]
        domain_name = runs[0]["domain"]
        grid_size = runs[0]["grid_size"]
        case = runs[0]["case"]
        tau = runs[0]["tau"]

        solver_times = [r["solver_time"] for r in runs]
        q_times = [r["q_destroy_time"] for r in runs]
        returns = [r["total_return"] for r in runs if not np.isnan(r["total_return"])]
        path_lens = [r["path_length"] for r in runs if r["path_length"] > 0]
        terminals = [r["terminal"] for r in runs]

        st_mean = np.mean(solver_times)
        st_std = np.std(solver_times)
        qt_mean = np.mean(q_times)

        if returns:
            ret_mean = np.mean(returns)
            ret_std = np.std(returns)
            ret_str = f"{ret_mean:8.4f} +/- {ret_std:.4f}"
        else:
            ret_str = "INFEASIBLE"

        if path_lens:
            pl_str = f"{np.mean(path_lens):.0f}"
        else:
            pl_str = "N/A"

        tau_str = str(tau) if tau is not None else "-"
        term_str = terminals[0] if len(set(terminals)) == 1 else "mixed"

        line = (f"{domain_name:<25} {grid_size:<8} {case:<8} {tau_str:<6} "
                f"{st_mean:8.4f} +/- {st_std:.4f} {qt_mean:12.4f}     "
                f"{ret_str:>18} {pl_str:>10} {term_str:<20}")
        lines.append(line)

    summary_text = "\n".join(lines)

    filepath = os.path.join(results_base_dir, "summary.txt")
    with open(filepath, "w") as f:
        f.write(summary_text)

    print(f"\n\n{summary_text}")
    print(f"\nSummary saved to: {filepath}")
    return filepath


def write_raw_csv(all_results, results_base_dir):
    """Write per-run results to CSV (every individual run as a row)."""
    filepath = os.path.join(results_base_dir, "results_raw.csv")
    fieldnames = ["domain", "domain_type", "size", "grid_size", "case", "tau",
                  "rep", "solver_time", "q_destroy_time", "total_time",
                  "total_return", "path_length", "terminal",
                  "lp_status", "sol_status"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = dict(r)
            row["total_time"] = row["solver_time"] + row["q_destroy_time"]
            row["tau"] = row["tau"] if row["tau"] is not None else ""
            writer.writerow(row)
    print(f"Raw CSV saved to: {filepath}")
    return filepath


def write_summary_csv(all_results, results_base_dir):
    """Write aggregated summary CSV (mean/std over reps) for paper reporting."""
    groups = {}
    for r in all_results:
        key = (r["domain"], r["domain_type"], r["size"], r["grid_size"], r["case"],
               r["tau"] if r["tau"] is not None else "")
        groups.setdefault(key, []).append(r)

    filepath = os.path.join(results_base_dir, "results_summary.csv")
    fieldnames = ["domain", "domain_type", "size", "grid_size", "case", "tau",
                  "return_mean", "return_std", "path_length_mean", "path_length_std",
                  "solver_time_mean", "solver_time_std",
                  "q_destroy_time_mean", "total_time_mean", "total_time_std",
                  "terminal", "lp_status", "sol_status", "n_feasible", "n_reps"]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key in sorted(groups.keys()):
            runs = groups[key]
            domain, domain_type, size, grid_size, case, tau = key

            solver_times = [r["solver_time"] for r in runs]
            q_times = [r["q_destroy_time"] for r in runs]
            total_times = [r["solver_time"] + r["q_destroy_time"] for r in runs]
            returns = [r["total_return"] for r in runs if not np.isnan(r["total_return"])]
            path_lens = [r["path_length"] for r in runs if r["path_length"] > 0]
            terminals = [r["terminal"] for r in runs]
            lp_statuses = [r.get("lp_status", "") for r in runs]
            sol_statuses = [r.get("sol_status", "") for r in runs]

            row = {
                "domain": domain, "domain_type": domain_type,
                "size": size, "grid_size": grid_size,
                "case": case, "tau": tau,
                "return_mean": f"{np.mean(returns):.4f}" if returns else "NaN",
                "return_std": f"{np.std(returns):.4f}" if returns else "NaN",
                "path_length_mean": f"{np.mean(path_lens):.1f}" if path_lens else "NaN",
                "path_length_std": f"{np.std(path_lens):.1f}" if path_lens else "NaN",
                "solver_time_mean": f"{np.mean(solver_times):.4f}",
                "solver_time_std": f"{np.std(solver_times):.4f}",
                "q_destroy_time_mean": f"{np.mean(q_times):.4f}",
                "total_time_mean": f"{np.mean(total_times):.4f}",
                "total_time_std": f"{np.std(total_times):.4f}",
                "terminal": terminals[0] if len(set(terminals)) == 1 else "mixed",
                "lp_status": lp_statuses[0] if len(set(lp_statuses)) == 1 else "mixed",
                "sol_status": sol_statuses[0] if len(set(sol_statuses)) == 1 else "mixed",
                "n_feasible": len(returns),
                "n_reps": len(runs),
            }
            writer.writerow(row)

    print(f"Summary CSV saved to: {filepath}")
    return filepath


def run_all_experiments():
    """Main entry point: run all domains, cases, taus, reps."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    domains = get_all_domains()
    all_results = []

    total_start = time.time()

    for domain_cfg in domains:
        domain_results = run_single_domain(domain_cfg, RESULTS_DIR)
        all_results.extend(domain_results)

    total_time = time.time() - total_start
    print(f"\n\nTotal experiment time: {total_time:.1f}s")

    write_summary(all_results, RESULTS_DIR)
    write_raw_csv(all_results, RESULTS_DIR)
    write_summary_csv(all_results, RESULTS_DIR)


if __name__ == "__main__":
    run_all_experiments()
