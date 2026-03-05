"""Multi-obstacle experiment for NPField D3.

Runs the D3 model on a scenario with multiple dynamic obstacles on an empty
map.  Each obstacle is independently encoded via ``encode_map_footprint``;
the resulting embeddings are stacked and passed to the MPC robot model which
sums the predicted potentials (additive superposition of repulsive fields).

Usage examples
--------------
    # Run the default 3-obstacle scenario shipped in scenario_multi_obstacle.json
    python test_solver_multi.py --test-episode --save-potential-gif

    # Run from a custom scenario config (editable for debugging)
    python test_solver_multi.py --scenario-json scenario_multi_obstacle.json

    # Re-generate the default scenario JSON (useful after parameter changes)
    python test_solver_multi.py --dump-scenario
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from matplotlib import colors
from matplotlib.path import Path as MplPath
from math import cos, sin
from scipy.ndimage import distance_transform_edt

import create_solver_GPT as create_solver
from model_nn_GPT import GPTConfig, GPT
from mpc_params import (
    CTRL_A_MAX,
    CTRL_A_MIN,
    CTRL_W_MAX,
    CTRL_W_MIN,
    MAP_SCALE,
    N_HORIZON,
    OBSTACLE_FOOTPRINT_ANGLE,
    OBSTACLE_FOOTPRINT_RADIUS,
    V_MAX,
    TIME_STEPS,
    POTENTIAL_RESOLUTION,
    SOLVER_BASE_TF,
    OBSTACLE_PRED_DT,
    OBSTACLE_STEP_M,
    TF_MIN_BUFFER_SEC,
    TF_TIME_SLACK,
    TF_TURN_WEIGHT,
)

OBSTACLE_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_paths(finetune_checkpoint: str = "") -> tuple[Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    npfield_dir = script_dir.parent
    repo_root = npfield_dir.parent
    dataset_root = Path(
        os.getenv("NPFIELD_DATASET_DIR", repo_root / "NPField" / "dataset" / "dataset1000")
    )
    default_checkpoint = Path(
        os.getenv(
            "NPFIELD_CHECKPOINT",
            repo_root / "NPField" / "dataset" / "trained-models" / "NPField_D3_finetune.pth",

            
        )
    )
    checkpoint_path = (
        Path(finetune_checkpoint).expanduser() if finetune_checkpoint else default_checkpoint
    )
    output_dir = npfield_dir / "output"
    return dataset_root, checkpoint_path, output_dir


# ---------------------------------------------------------------------------
# Scenario construction & persistence
# ---------------------------------------------------------------------------

def build_obstacle_trajectory(obstacle_path, steps=TIME_STEPS):
    (x0, y0), (x1, y1) = obstacle_path
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        theta = 0.0
        dir_x, dir_y = 1.0, 0.0
    else:
        theta = math.atan2(dy, dx)
        dir_x, dir_y = dx / norm, dy / norm
    step_idx = np.arange(steps, dtype=float)
    xs = x0 + dir_x * OBSTACLE_STEP_M * step_idx
    ys = y0 + dir_y * OBSTACLE_STEP_M * step_idx
    thetas = np.full(steps, theta)
    return np.stack((xs, ys, thetas), axis=1)


def load_scenario_list(scenario_json_path=None):
    """Load the scenario JSON and return a list of individual scenario dicts.

    Supports two formats:
    - **Multi**: ``{"scenarios": [ {...}, {...}, ... ]}``
    - **Legacy single**: ``{"agent": ..., "obstacles": ..., ...}``
    """
    if scenario_json_path is None:
        scenario_json_path = Path(__file__).resolve().parent / "scenario_multi_obstacle.json"
    with open(scenario_json_path) as f:
        raw = json.load(f)
    if "scenarios" in raw:
        return raw["scenarios"]
    return [raw]


def _parse_single_scenario(config):
    """Extract agent path + obstacle trajectories from one scenario dict."""
    x_ref_points = config["agent"]["x_path"]
    y_ref_points = config["agent"]["y_path"]
    theta_0 = config["agent"]["theta_0"]

    obstacle_trajs = []
    for obst in config["obstacles"]:
        start = tuple(obst["start"])
        end = tuple(obst["end"])
        traj = build_obstacle_trajectory((start, end), steps=TIME_STEPS)
        obstacle_trajs.append(traj)

    return x_ref_points, y_ref_points, theta_0, obstacle_trajs


def build_multi_obstacle_config(scenario_json_path=None, scenario_id=None):
    """Build a multi-obstacle scenario from JSON.

    If *scenario_id* is ``None`` the first (or only) scenario is returned.
    """
    scenarios = load_scenario_list(scenario_json_path)
    if scenario_id is not None:
        config = scenarios[scenario_id]
    else:
        config = scenarios[0]
    x_ref, y_ref, theta_0, trajs = _parse_single_scenario(config)
    return config, x_ref, y_ref, theta_0, trajs


def save_scenario_config(config, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Scenario config saved to {output_path}")


# ---------------------------------------------------------------------------
# Empty map & footprint helpers
# ---------------------------------------------------------------------------

def make_empty_map_data():
    """Create a minimal empty map_data array: shape ``(1, TIME_STEPS+1, 50, 50)``."""
    return np.zeros((1, TIME_STEPS + 1, 50, 50))


def _draw_obstacle_on_map(base_map, x_obst, y_obst, theta_obst):
    modified = base_map.copy().astype(float)
    R = OBSTACLE_FOOTPRINT_RADIUS
    A = OBSTACLE_FOOTPRINT_ANGLE
    corners_world = np.array([
        [x_obst + R * cos(theta_obst - A), y_obst + R * sin(theta_obst - A)],
        [x_obst + R * cos(theta_obst + A), y_obst + R * sin(theta_obst + A)],
        [x_obst - R * cos(theta_obst - A), y_obst - R * sin(theta_obst - A)],
        [x_obst - R * cos(theta_obst + A), y_obst - R * sin(theta_obst + A)],
    ])
    grid_col = corners_world[:, 0] * 10.0
    grid_row = (5.0 - corners_world[:, 1]) * 10.0
    verts = np.column_stack((grid_col, grid_row))
    verts = np.vstack([verts, verts[:1]])
    poly = MplPath(verts)
    cols_g, rows_g = np.meshgrid(
        np.arange(50, dtype=float) + 0.5,
        np.arange(50, dtype=float) + 0.5,
    )
    pts = np.column_stack((cols_g.ravel(), rows_g.ravel()))
    mask = poly.contains_points(pts).reshape(50, 50)
    modified[mask] = 100.0
    return modified


def _build_combined_obstacle_map(base_map, obstacle_trajs, timestep=0):
    """Draw ALL obstacles at ``timestep`` onto *base_map* (accumulated)."""
    combined = base_map.copy().astype(float)
    for traj in obstacle_trajs:
        combined = _draw_obstacle_on_map(
            combined, traj[timestep, 0], traj[timestep, 1], traj[timestep, 2],
        )
    return combined


def _make_map_inp_row(combined_map, footprint, obst_traj_t0):
    """Build a single ``(5003,)`` map-input row.

    *combined_map* already has **all** obstacles imprinted (50x50, values 0/100).
    *obst_traj_t0* is ``(x, y, theta)`` of the *specific* obstacle at t=0 —
    this goes into the last 3 dynamic-info slots exactly like D3.
    """
    row = torch.zeros(5003)
    fp_flat = torch.tensor(footprint.flatten(), dtype=torch.float32)
    row[:2500] = torch.tensor(combined_map.flatten(), dtype=torch.float32)
    row[2500:5000] = fp_flat
    row[:5000] /= 100.0
    row[-3] = float(obst_traj_t0[0])
    row[-2] = float(obst_traj_t0[1])
    row[-1] = float(obst_traj_t0[2])
    return row.cuda()


# ---------------------------------------------------------------------------
# Multi-obstacle embedding
# ---------------------------------------------------------------------------

def compute_multi_embedding(model_loaded, base_map, footprint, obstacle_trajs):
    """Encode each obstacle and stack embeddings ``(N_obst, emb_dim)``.

    The **combined** map (all obstacles at t=0 imprinted) is used for every
    embedding so the network sees the full scene.  The per-obstacle dynamic
    info (dyn_x, dyn_y, dyn_theta) tells the model *which* obstacle to
    compute the potential for — exactly matching the D3 single-obstacle
    format.
    """
    combined_map_t0 = _build_combined_obstacle_map(base_map, obstacle_trajs, timestep=0)

    embeddings = []
    for i, traj in enumerate(obstacle_trajs):
        map_inp_row = _make_map_inp_row(combined_map_t0, footprint, traj[0])
        with torch.no_grad():
            encoded = model_loaded.encode_map_footprint(map_inp_row)
        embeddings.append(encoded.detach().cpu().numpy())
        norm = np.linalg.norm(embeddings[-1])
        print(f"  Obstacle {i} embedding L2 norm: {norm:.4f}, "
              f"dyn_coords=({traj[0,0]:.2f}, {traj[0,1]:.2f}, {math.degrees(traj[0,2]):.0f}°)")

    return np.vstack(embeddings)


def debug_check_potentials(model_loaded, multi_embedding, obstacle_trajs,
                           agent_start, theta=0.0):
    """Evaluate potentials at key locations to verify the field is non-trivial."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_obst = multi_embedding.shape[0]

    test_points, labels = [], []
    for i, traj in enumerate(obstacle_trajs):
        test_points.append([traj[0, 0], traj[0, 1]])
        labels.append(f"obst_{i}_pos")
    test_points.append(list(agent_start))
    labels.append("agent_start")
    test_points.append([2.5, 2.5])
    labels.append("centre")
    test_points.append([0.5, 0.5])
    labels.append("far_corner")

    print("\n--- Potential field sanity check (summed over obstacles, t=0) ---")
    for pt, label in zip(test_points, labels):
        total = np.zeros(TIME_STEPS)
        coords = torch.tensor([[pt[0], pt[1]]], dtype=torch.float32, device=device)
        th = torch.tensor([[theta]], dtype=torch.float32, device=device)
        for oi in range(n_obst):
            emb = torch.tensor(
                multi_embedding[oi:oi + 1], dtype=torch.float32, device=device,
            ).expand(1, -1)
            inp = torch.cat([emb, coords, th], dim=1)
            with torch.no_grad():
                out = model_loaded(inp)
            total += out.cpu().numpy()[0]
        print(f"  {label:16s} ({pt[0]:.2f},{pt[1]:.2f})  "
              f"t0={total[0]:.4f}  mean={total.mean():.4f}  max={total.max():.4f}")
    print("--- end check ---\n")


# ---------------------------------------------------------------------------
# Potential-field grid inference (for visualisation)
# ---------------------------------------------------------------------------

def infer_multi_potential_grid(model_loaded, multi_embedding, angle, device, chunk_size):
    """Evaluate the *summed* potential field across the 5x5m grid.

    For each grid point, the model evaluates potentials for every obstacle
    embedding and sums them, giving a single scalar per timestep.
    """
    xs = np.linspace(0.0, 5.0, POTENTIAL_RESOLUTION, endpoint=False)
    ys = np.linspace(0.0, 5.0, POTENTIAL_RESOLUTION, endpoint=False)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)

    n_obst = multi_embedding.shape[0]
    emb_dim = multi_embedding.shape[1]
    n_points = coords.shape[0]

    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    theta_t = torch.full((n_points, 1), angle, dtype=torch.float32, device=device)

    all_potentials = torch.zeros(n_points, TIME_STEPS, device=device)
    for oi in range(n_obst):
        emb = torch.tensor(
            multi_embedding[oi : oi + 1], dtype=torch.float32, device=device
        ).expand(n_points, -1)
        input_batch = torch.hstack((emb, coords_t, theta_t))
        outputs = []
        with torch.no_grad():
            for start in range(0, n_points, chunk_size):
                chunk = input_batch[start : start + chunk_size]
                outputs.append(model_loaded(chunk))
        all_potentials += torch.cat(outputs, dim=0)

    result = all_potentials.cpu().numpy()
    return (
        result.reshape(POTENTIAL_RESOLUTION, POTENTIAL_RESOLUTION, TIME_STEPS)
        .transpose(2, 0, 1)
    )


# ---------------------------------------------------------------------------
# MPC trajectory solver
# ---------------------------------------------------------------------------

def test_solver(
    acados_solver,
    x_ref_points,
    y_ref_points,
    theta_0,
    num_map,
    ax1,
):
    def _wrap_to_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _compute_feasible_tf(path_length, total_turn_rad):
        v_cruise = max(1e-3, float(V_MAX))
        a_acc = max(1e-3, float(CTRL_A_MAX))
        a_dec = max(1e-3, float(-CTRL_A_MIN))
        w_max = max(1e-3, float(CTRL_W_MAX))
        d_acc = (v_cruise ** 2) / (2.0 * a_acc)
        d_dec = (v_cruise ** 2) / (2.0 * a_dec)
        if path_length >= (d_acc + d_dec):
            t_min = v_cruise / a_acc + (path_length - d_acc - d_dec) / v_cruise + v_cruise / a_dec
        else:
            v_peak = math.sqrt(max(0.0, 2.0 * path_length / (1.0 / a_acc + 1.0 / a_dec)))
            t_min = v_peak / a_acc + v_peak / a_dec
        t_turn = TF_TURN_WEIGHT * (total_turn_rad / w_max)
        return max(path_length / v_cruise, TF_TIME_SLACK * (t_min + t_turn) + TF_MIN_BUFFER_SEC)

    nx, nu = 5, 3
    ny = nx + nu
    N = N_HORIZON
    yref = np.zeros([N, ny + 1])
    v_0, v_e = 0, 0

    theta = [theta_0]
    theta_ref = [theta_0]
    len_segments = []
    num_segment = len(x_ref_points) - 1
    length_path = 0.0
    for i in range(num_segment):
        seg = math.hypot(x_ref_points[i + 1] - x_ref_points[i],
                         y_ref_points[i + 1] - y_ref_points[i])
        length_path += seg
        theta.append(math.atan2(y_ref_points[i + 1] - y_ref_points[i],
                                x_ref_points[i + 1] - x_ref_points[i]))
        len_segments.append(seg)

    step_line = length_path / N
    print("length path", length_path)

    total_turn_rad = sum(abs(_wrap_to_pi(theta[i + 1] - theta[i])) for i in range(len(theta) - 1))
    base_desired_tf = _compute_feasible_tf(length_path, total_turn_rad)

    x_ref = [x_ref_points[0]]
    y_ref = [y_ref_points[0]]
    k = 0
    for i in range(N + 1):
        x_ref.append(x_ref[-1] + step_line * math.cos(theta[k + 1]))
        y_ref.append(y_ref[-1] + step_line * math.sin(theta[k + 1]))
        theta_ref.append(theta[k + 1])
        d = math.hypot(x_ref[-1] - x_ref_points[k], y_ref[-1] - y_ref_points[k])
        if d > len_segments[k] and k < num_segment - 1:
            k += 1
            x_ref[i] = x_ref_points[k]
            y_ref[i] = y_ref_points[k]
        elif k > num_segment - 1:
            break

    x0 = np.array([x_ref_points[0], y_ref_points[0], v_0, theta_0, 0])
    init_x = np.array(x_ref[: N + 1])
    init_y = np.array(y_ref[: N + 1])
    init_theta = np.array(theta_ref[: N + 1])
    yref[:, 0] = init_x[:N]
    yref[:, 1] = init_y[:N]
    yref[:, 2] = V_MAX
    yref[:, 3] = init_theta[:N]

    simX = np.zeros((N + 1, 5))
    simU = np.zeros((N, nu))
    t = time.perf_counter()

    tf_growth_factors = (1.1, 1.3)
    goal_reach_tol_m = 0.18
    best_key = None
    best_candidate = None
    a_ref = np.zeros(1)

    for growth in tf_growth_factors:
        desired_tf = base_desired_tf * growth
        yref[:, 4] = np.linspace(0, desired_tf, N, endpoint=False)
        x_goal = np.array([init_x[-1], init_y[-1], v_e, init_theta[-1], desired_tf])
        yref_e = np.concatenate([x_goal, a_ref])
        x_traj_init = np.column_stack([yref[:, 0], yref[:, 1], yref[:, 2], yref[:, 3], yref[:, 4]])
        time_scale = desired_tf / SOLVER_BASE_TF

        for i in range(N):
            acados_solver.set(i, "y_ref", yref[i])
            acados_solver.set(i, "x", x_traj_init[i])
            acados_solver.set(i, "u", np.array([0.0, 0.0, time_scale]))
            acados_solver.set(i, "lbu", np.array([CTRL_A_MIN, CTRL_W_MIN, time_scale]))
            acados_solver.set(i, "ubu", np.array([CTRL_A_MAX, CTRL_W_MAX, time_scale]))
        acados_solver.set(N, "y_ref", yref_e)
        acados_solver.set(N, "x", x_goal)
        acados_solver.set(0, "lbx", x0)
        acados_solver.set(0, "ubx", x0)

        status = 1
        for _ in range(3):
            status = acados_solver.solve()
            if status == 0:
                break

        cand_simX = np.zeros((N + 1, 5))
        for i in range(N + 1):
            cand_simX[i, :] = acados_solver.get(i, "x")
        cand_goal_err = math.hypot(cand_simX[-1, 0] - x_ref_points[-1],
                                   cand_simX[-1, 1] - y_ref_points[-1])
        cand_simU = np.zeros((N, nu))
        for i in range(N):
            cand_simU[i, :] = acados_solver.get(i, "u")
        cand_cost = float(acados_solver.get_cost())
        key = (0 if cand_goal_err < goal_reach_tol_m else 1, cand_cost)
        if best_key is None or key < best_key:
            best_key = key
            best_candidate = dict(
                status=status, goal_error_m=cand_goal_err, cost=cand_cost,
                simX=cand_simX.copy(), simU=cand_simU.copy(),
                desired_tf=desired_tf, growth=growth,
            )
        print(f"TF candidate {desired_tf:.2f}s (x{growth:.2f}) -> "
              f"status={status}, goal_err={cand_goal_err:.3f}m, cost={cand_cost:.3f}")

    if best_candidate is None:
        raise RuntimeError("Failed to evaluate any TF candidate.")

    simX[:, :] = best_candidate["simX"]
    simU[:, :] = best_candidate["simU"]
    cost = best_candidate["cost"]
    status = best_candidate["status"]
    selected_tf = best_candidate["desired_tf"]
    print(f"Selected TF={selected_tf:.2f}s, status={status}, "
          f"goal_err={best_candidate['goal_error_m']:.3f}m, cost={cost:.3f}")
    if status != 0:
        print("WARNING: solver did not fully converge; trajectory may be suboptimal.")

    elapsed = 1000 * (time.perf_counter() - t)
    print(f"Trajectory solve time (ms): {elapsed:.2f}")

    for i in range(N + 1):
        simX[i, 0] *= MAP_SCALE
        simX[i, 1] *= MAP_SCALE
        simX[i, 4] = round(simX[i, 4], 1)

    ax1.plot(simX[:, 0], simX[:, 1], linewidth=2, color="r", label="NPField path")
    ax1.plot([init_x[0] * MAP_SCALE], [init_y[0] * MAP_SCALE], marker="x", color="r", markersize=8)
    ax1.plot([init_x[-1] * MAP_SCALE], [init_y[-1] * MAP_SCALE], marker="o", color="r", markersize=6)

    return simX, elapsed


# ---------------------------------------------------------------------------
# GIF generation for multiple obstacles
# ---------------------------------------------------------------------------

def _obstacle_polygon(x_obst, y_obst, theta_obst):
    R, A = OBSTACLE_FOOTPRINT_RADIUS, OBSTACLE_FOOTPRINT_ANGLE
    xs = [
        x_obst + R * cos(theta_obst - A),
        x_obst + R * cos(theta_obst + A),
        x_obst - R * cos(theta_obst - A),
        x_obst - R * cos(theta_obst + A),
    ]
    ys = [
        y_obst + R * sin(theta_obst - A),
        y_obst + R * sin(theta_obst + A),
        y_obst - R * sin(theta_obst - A),
        y_obst - R * sin(theta_obst + A),
    ]
    xs.append(xs[0])
    ys.append(ys[0])
    return [x * MAP_SCALE for x in xs], [y * MAP_SCALE for y in ys]


def _robot_polygon(x_pos, y_pos, theta):
    s = MAP_SCALE
    x0, y0 = x_pos + s * 0.6 * cos(theta - 0.59), y_pos + s * 0.6 * sin(theta - 0.59)
    x5, y5 = x_pos + s * 0.6 * cos(theta + 0.59), y_pos + s * 0.6 * sin(theta + 0.59)
    x6, y6 = x_pos - s * 0.6 * cos(theta - 0.59), y_pos - s * 0.6 * sin(theta - 0.59)
    x7, y7 = x_pos - s * 0.6 * cos(theta + 0.59), y_pos - s * 0.6 * sin(theta + 0.59)
    return [x0, x5, x6, x7, x0], [y0, y5, y6, y7, y0]


def _interp_pose(path_array, t):
    times = path_array[:, 4]
    if t <= times[0]:
        return path_array[0, 0], path_array[0, 1], path_array[0, 3]
    if t >= times[-1]:
        return path_array[-1, 0], path_array[-1, 1], path_array[-1, 3]
    idx = int(np.searchsorted(times, t, side="right"))
    idx = max(1, min(idx, len(times) - 1))
    t0, t1 = times[idx - 1], times[idx]
    if abs(t1 - t0) < 1e-6:
        return path_array[idx, 0], path_array[idx, 1], path_array[idx, 3]
    alpha = (t - t0) / (t1 - t0)
    x = path_array[idx - 1, 0] + alpha * (path_array[idx, 0] - path_array[idx - 1, 0])
    y = path_array[idx - 1, 1] + alpha * (path_array[idx, 1] - path_array[idx - 1, 1])
    th = path_array[idx - 1, 3] + alpha * (path_array[idx, 3] - path_array[idx - 1, 3])
    return x, y, th


def gif_generate_multi(
    path,
    map_data,
    num_map,
    id_map,
    output_dir,
    x_ref_points,
    y_ref_points,
    obstacle_trajs,
    config,
    potential_frames=None,
):
    """Generate an animated GIF showing the robot and all dynamic obstacles."""
    bg_map = map_data[num_map][0]
    map_h, map_w = bg_map.shape
    robot_path_x, robot_path_y = path[:, 0], path[:, 1]
    cfg_start_x = float(x_ref_points[0]) * MAP_SCALE
    cfg_start_y = float(y_ref_points[0]) * MAP_SCALE
    cfg_goal_x = float(x_ref_points[-1]) * MAP_SCALE
    cfg_goal_y = float(y_ref_points[-1]) * MAP_SCALE

    obst_colors = [
        config["obstacles"][i].get("color", OBSTACLE_COLORS[i % len(OBSTACLE_COLORS)])
        for i in range(len(obstacle_trajs))
    ]

    obst_paths_x, obst_paths_y = [], []
    for traj in obstacle_trajs:
        obst_paths_x.append(traj[:, 0] * MAP_SCALE)
        obst_paths_y.append(traj[:, 1] * MAP_SCALE)

    total_time = max(15.0, float(path[-1, 4]))
    frames = []

    obst_step_indices = [0] * len(obstacle_trajs)
    t_obst_next = OBSTACLE_PRED_DT

    current_obst_pos = []
    for traj in obstacle_trajs:
        current_obst_pos.append((traj[0, 0], traj[0, 1], traj[0, 2]))

    for t_frame in np.arange(0.0, total_time + 1e-6, 0.05):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        if potential_frames is not None:
            pred_idx = min(int(max(t_frame, 0.0) / OBSTACLE_PRED_DT), TIME_STEPS - 1)
            ax.imshow(
                potential_frames[pred_idx].T,
                origin="lower",
                extent=(0.0, map_w, 0.0, map_h),
                cmap="viridis",
                alpha=0.95,
                interpolation="nearest",
            )
            ax.pcolor(
                bg_map[::-1],
                cmap=colors.ListedColormap(["white", "black"]),
                edgecolors="none",
                alpha=0.12,
            )
        else:
            cmap_bg = colors.ListedColormap(["white", "black"])
            ax.pcolor(bg_map[::-1], cmap=cmap_bg, edgecolors="w", linewidths=0.1)

        ax.plot(robot_path_x, robot_path_y, color="r", linewidth=1.2)
        ax.plot(cfg_start_x, cfg_start_y, marker="x", color="r", markersize=8)
        ax.plot(cfg_goal_x, cfg_goal_y, marker="o", color="r", markersize=6)

        for oi, traj in enumerate(obstacle_trajs):
            c = obst_colors[oi]
            ax.plot(obst_paths_x[oi], obst_paths_y[oi], color=c, linewidth=1, alpha=0.5)
            ax.plot(obst_paths_x[oi][0], obst_paths_y[oi][0], marker="x", color=c, markersize=6)

            ox, oy, oth = current_obst_pos[oi]
            poly_x, poly_y = _obstacle_polygon(ox, oy, oth)
            ax.fill(poly_x, poly_y, color=c, alpha=0.35)
            ax.plot(poly_x, poly_y, color=c, linewidth=1.5)

        interp_x, interp_y, interp_th = _interp_pose(path, t_frame)
        rpx, rpy = _robot_polygon(interp_x, interp_y, interp_th)
        ax.fill(rpx, rpy, color="red", alpha=0.25)
        ax.plot(rpx, rpy, color="k", linewidth=1.5)

        ax.set_xlim(0, map_w)
        ax.set_ylim(0, map_h)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t_frame:.2f}s", fontsize=10)

        fig.canvas.draw()
        data = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(data)
        plt.close(fig)

        if round(t_frame, 2) >= t_obst_next:
            for oi, traj in enumerate(obstacle_trajs):
                if obst_step_indices[oi] < TIME_STEPS - 1:
                    obst_step_indices[oi] += 1
                    si = obst_step_indices[oi]
                    current_obst_pos[oi] = (traj[si, 0], traj[si, 1], traj[si, 2])
            t_obst_next += OBSTACLE_PRED_DT

    output_dir.mkdir(parents=True, exist_ok=True)
    n_obst = len(obstacle_trajs)
    output_path = output_dir / f"D3_multi_{n_obst}obst_ep{num_map}_ID_{id_map}.gif"
    imageio.mimsave(str(output_path), frames, format="GIF", fps=20, loop=0)
    print(f"GIF saved to {output_path}")


# ---------------------------------------------------------------------------
# Metrics (multi-obstacle)
# ---------------------------------------------------------------------------

def compute_trajectory_metrics_multi(
    path_mpc,
    map_data,
    num_map,
    obstacle_trajs,
    total_elapsed_ms,
    goal_xy=None,
):
    N = path_mpc.shape[0] - 1
    xs = path_mpc[:, 0] / MAP_SCALE
    ys = path_mpc[:, 1] / MAP_SCALE
    thetas = path_mpc[:, 3]
    times = path_mpc[:, 4]
    velocities = path_mpc[:, 2]

    points = np.column_stack((xs, ys))
    filtered = [points[0]]
    for p in points[1:]:
        if np.hypot(p[0] - filtered[-1][0], p[1] - filtered[-1][1]) > 1e-9:
            filtered.append(p)
    filtered = np.array(filtered, dtype=float)

    path_length = float(np.sum(np.hypot(np.diff(filtered[:, 0]), np.diff(filtered[:, 1])))) if len(filtered) >= 2 else 0.0

    if len(filtered) >= 3:
        seg_dx, seg_dy = np.diff(filtered[:, 0]), np.diff(filtered[:, 1])
        yaws = np.arctan2(seg_dy, seg_dx)
        dtheta = np.diff(yaws)
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    else:
        dtheta = np.array([], dtype=float)

    smoothness = float(np.sum(dtheta ** 2) / max(path_length, 1e-6))
    aol = float(np.sum(np.abs(dtheta)) / max(path_length, 1e-6))

    if goal_xy is not None:
        straight_line = math.hypot(goal_xy[0] - xs[0], goal_xy[1] - ys[0])
    else:
        straight_line = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    path_efficiency = min(straight_line / max(path_length, 1e-6), 1.0)

    mean_speed = float(np.mean(np.abs(velocities)))
    velocity_utilization = mean_speed / float(V_MAX) if float(V_MAX) > 0 else 0.0

    dt_arr = np.diff(times)
    dt_arr = np.where(np.abs(dt_arr) < 1e-9, 1e-9, dt_arr)
    acc_est = np.diff(velocities) / dt_arr
    dtheta_ctrl = np.diff(thetas)
    dtheta_ctrl = (dtheta_ctrl + np.pi) % (2 * np.pi) - np.pi
    omega_est = dtheta_ctrl / dt_arr
    control_energy = float(np.sum((acc_est ** 2 + omega_est ** 2) * dt_arr))

    if len(acc_est) >= 2:
        dt_inner = np.where(np.abs(dt_arr[:-1]) < 1e-9, 1e-9, dt_arr[:-1])
        jerk = float(np.sqrt(np.mean((np.diff(acc_est) / dt_inner) ** 2 + (np.diff(omega_est) / dt_inner) ** 2)))
    else:
        jerk = 0.0

    base_map = map_data[num_map][0]
    fp_params = [
        (0.6, -0.59), (0.6, 0.59), (-0.6, -0.59), (-0.6, 0.59),
        (0.75, -0.16), (0.75, 0.16),
    ]
    min_safety = float("inf")
    per_wp_clearance = []

    for k in range(N + 1):
        rx, ry, rtheta, t_val = xs[k], ys[k], thetas[k], times[k]
        corners = [(rx + r * cos(rtheta + a), ry + r * sin(rtheta + a)) for r, a in fp_params]
        corners.append((rx, ry))

        wp_min = float("inf")
        for traj in obstacle_trajs:
            obst_idx = min(int(max(t_val, 0.0) / OBSTACLE_PRED_DT), len(traj) - 1)
            combined = _draw_obstacle_on_map(base_map, traj[obst_idx, 0], traj[obst_idx, 1], traj[obst_idx, 2])
            binary_occ = (combined > 50).astype(np.float64)
            sdf_grid = distance_transform_edt(1 - binary_occ) * 0.1
            for cx, cy in corners:
                gi = max(0, min(49, int(round((5 - cy) / 0.1))))
                gj = max(0, min(49, int(round(cx / 0.1))))
                wp_min = min(wp_min, sdf_grid[gi, gj])

        min_safety = min(min_safety, wp_min)
        per_wp_clearance.append(wp_min)

    mean_clearance = float(np.mean(per_wp_clearance))
    collision = bool(min_safety <= 0)

    goal_error = math.hypot(xs[-1] - goal_xy[0], ys[-1] - goal_xy[1]) if goal_xy else float("nan")
    success = (not collision) and (goal_error <= 0.18)

    if path_length > 0 and straight_line > 0:
        spl = float(success) * (straight_line / max(path_length, straight_line))
    else:
        spl = 0.0

    return {
        "time_ms": round(float(total_elapsed_ms), 2),
        "path_length_m": round(path_length, 4),
        "smoothness": round(smoothness, 6),
        "aol": round(aol, 6),
        "safety_distance_m": round(float(min_safety), 4),
        "goal_error_m": round(float(goal_error), 4),
        "collision": collision,
        "success": success,
        "path_efficiency": round(path_efficiency, 4),
        "mean_clearance_m": round(mean_clearance, 4),
        "velocity_utilization": round(velocity_utilization, 4),
        "control_energy": round(control_energy, 4),
        "jerk": round(jerk, 4),
        "spl": round(spl, 4),
        "num_obstacles": len(obstacle_trajs),
    }


# ---------------------------------------------------------------------------
# Model loading (same as D3)
# ---------------------------------------------------------------------------

def _unwrap_state_dict(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Checkpoint payload is not a valid state_dict dict.")


def _infer_model_args_from_state_dict(state_dict):
    n_embd = 576
    if "x_encode.weight" in state_dict:
        n_embd = int(state_dict["x_encode.weight"].shape[0])
    layer_ids = set()
    for key in state_dict:
        if key.startswith("transformer.h."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_ids.add(int(parts[2]))
    n_layer = (max(layer_ids) + 1) if layer_ids else 4
    n_head = 4
    if n_embd % n_head != 0:
        n_head = 1
    return dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                block_size=1024, bias=True, vocab_size=1024, dropout=0.1)


def load_model(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location="cpu")
    pretrained_dict = _unwrap_state_dict(payload)
    model_args = _infer_model_args_from_state_dict(pretrained_dict)
    print("Inferred model args:", model_args)

    gptconf = GPTConfig(**model_args)
    model_loaded = GPT(gptconf)
    model_state = model_loaded.state_dict()
    matched = {k: v for k, v in pretrained_dict.items()
               if k in model_state and model_state[k].shape == v.shape}
    model_loaded.load_state_dict(matched, strict=False)
    print(f"Checkpoint: loaded {len(matched)} / {len(model_state)} params")

    model_loaded.to(device)
    model_loaded.eval()
    return model_loaded


def load_datasets(dataset_root):
    sub_maps = pickle.load(open(dataset_root / "dataset_1000_maps_0_100_all.pkl", "rb"))
    d_footprint = pickle.load(open(dataset_root / "data_footprint.pkl", "rb"))
    dyn_obst_info = pickle.load(open(dataset_root / "dataset_initial_position_dynamic_obst.pkl", "rb"))
    obst_motion_info = pickle.load(open(dataset_root / "dataset_1000_maps_obst_motion.pkl", "rb"))
    costmaps = pickle.load(open(dataset_root / "dataset_1000_costmaps.pkl", "rb"))
    return (
        sub_maps["submaps"],
        d_footprint["footprint_husky"],
        dyn_obst_info,
        obst_motion_info,
        costmaps["costmap"],
    )


# ---------------------------------------------------------------------------
# Main simulation driver
# ---------------------------------------------------------------------------

def run_multi_obstacle_simulation(
    model_loaded,
    footprint,
    config,
    x_ref_points,
    y_ref_points,
    theta_0,
    obstacle_trajs,
    output_dir,
    map_data=None,
    num_map=0,
    save_potential_gif_flag=False,
    potential_chunk_size=4096,
    allow_backward=False,
    save_gif=True,
    id_map=0,
):
    t_total = time.perf_counter()

    if map_data is None:
        print("WARNING: no dataset map_data provided — using empty map "
              "(the model may not produce meaningful potentials).")
        map_data = make_empty_map_data()
        num_map = 0

    map_data = map_data.copy()
    base_map = map_data[num_map][0].copy()

    combined_map_t0 = _build_combined_obstacle_map(base_map, obstacle_trajs, timestep=0)
    map_data[num_map][0] = combined_map_t0

    print(f"\n{'='*60}")
    print(f"Multi-obstacle scenario: {len(obstacle_trajs)} dynamic obstacles")
    print(f"Agent: ({x_ref_points[0]}, {y_ref_points[0]}) -> ({x_ref_points[-1]}, {y_ref_points[-1]})")
    for i, traj in enumerate(obstacle_trajs):
        label = config["obstacles"][i].get("label", f"obst_{i}")
        print(f"  Obstacle {i} [{label}]: ({traj[0,0]:.1f}, {traj[0,1]:.1f}) "
              f"-> heading {math.degrees(traj[0,2]):.0f}deg")
    print(f"{'='*60}\n")

    print("Computing multi-obstacle embeddings...")
    multi_embedding = compute_multi_embedding(model_loaded, base_map, footprint, obstacle_trajs)
    print(f"Stacked embedding shape: {multi_embedding.shape}")

    debug_check_potentials(
        model_loaded, multi_embedding, obstacle_trajs,
        agent_start=(x_ref_points[0], y_ref_points[0]), theta=theta_0,
    )

    print("Creating solver with summed potential field...")
    acados_solver = create_solver.create_solver(
        model_loaded, multi_embedding, allow_backward=allow_backward
    )

    potential_frames = None
    if save_potential_gif_flag and save_gif:
        print("Computing potential field grid for visualisation...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        potential_frames = infer_multi_potential_grid(
            model_loaded, multi_embedding, theta_0, device, potential_chunk_size,
        )

    fig, ax = plt.subplots(figsize=(1, 1))
    path_mpc, solve_elapsed = test_solver(
        acados_solver, x_ref_points, y_ref_points, theta_0, num_map, ax,
    )
    plt.close(fig)

    total_elapsed_ms = 1000 * (time.perf_counter() - t_total)
    print(f"Trajectory solve time (ms): {solve_elapsed:.2f}")
    print(f"Total planning time (ms): {total_elapsed_ms:.2f}")

    goal_xy = (x_ref_points[-1], y_ref_points[-1])
    metrics = compute_trajectory_metrics_multi(
        path_mpc, map_data, num_map, obstacle_trajs, total_elapsed_ms, goal_xy,
    )
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if save_gif:
        gif_generate_multi(
            path_mpc, map_data, num_map, id_map, output_dir,
            x_ref_points, y_ref_points, obstacle_trajs, config,
            potential_frames=potential_frames,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    config_copy_path = output_dir / f"D3_multi_scenario_{id_map}.json"
    save_scenario_config(config, config_copy_path)

    metrics_path = output_dir / f"D3_multi_metrics_{id_map}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return path_mpc, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Multi-obstacle NPField experiment (D3).")
    p.add_argument("--scenario-json", type=str, default="",
                   help="Path to a scenario JSON config file.")
    p.add_argument("--test-episode", action="store_true",
                   help="Run all scenarios from scenario_multi_obstacle.json.")
    p.add_argument("--scenario-id", type=int, default=None,
                   help="Run only this scenario index (0-based). Default: run all.")
    p.add_argument("--save-potential-gif", action="store_true",
                   help="Overlay the summed potential field in the GIF.")
    p.add_argument("--potential-chunk-size", type=int, default=4096)
    p.add_argument("--finetune-checkpoint", type=str, default="")
    p.add_argument("--allow-backward", action="store_true")
    p.add_argument("--no-gif", action="store_true", help="Skip GIF generation.")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_root, checkpoint_path, output_dir = resolve_paths(args.finetune_checkpoint)
    map_data, footprint, _, _, _ = load_datasets(dataset_root)

    map_data = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loaded = load_model(checkpoint_path, device)

    scenario_path = args.scenario_json or (
        str(Path(__file__).resolve().parent / "scenario_multi_obstacle.json")
        if args.test_episode else ""
    )
    if not scenario_path:
        print("ERROR: Provide --test-episode or --scenario-json <path>.")
        return

    scenarios = load_scenario_list(scenario_path)
    if args.scenario_id is not None:
        run_ids = [args.scenario_id]
    else:
        run_ids = list(range(len(scenarios)))

    print(f"Loaded {len(scenarios)} scenario(s); will run: {run_ids}\n")

    multi_output_dir = output_dir / "benchmark_D3_multi"
    all_metrics = {}
    for sid in run_ids:
        cfg = scenarios[sid]
        x_ref, y_ref, theta_0, obstacle_trajs = _parse_single_scenario(cfg)
        desc = cfg.get("description", f"scenario {sid}")
        n_obst = len(obstacle_trajs)
        print(f"\n{'#'*70}")
        print(f"# Scenario {sid}/{len(scenarios)-1}: {desc}")
        print(f"# {n_obst} obstacle(s)")
        print(f"{'#'*70}")

        _, metrics = run_multi_obstacle_simulation(
            model_loaded=model_loaded,
            footprint=footprint,
            config=cfg,
            x_ref_points=x_ref,
            y_ref_points=y_ref,
            theta_0=theta_0,
            obstacle_trajs=obstacle_trajs,
            output_dir=multi_output_dir,
            map_data=map_data,
            num_map=cfg.get("map_id", 0),
            save_potential_gif_flag=args.save_potential_gif,
            potential_chunk_size=args.potential_chunk_size,
            allow_backward=args.allow_backward,
            save_gif=not args.no_gif,
            id_map=sid,
        )
        all_metrics[sid] = metrics

    summary_path = multi_output_dir / "all_metrics.json"
    multi_output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n{'='*70}")
    print(f"All {len(run_ids)} scenario(s) complete.  Summary: {summary_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
