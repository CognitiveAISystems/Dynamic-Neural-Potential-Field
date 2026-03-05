import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pickle
import torch
from matplotlib import colors
from matplotlib.path import Path as MplPath
from math import cos, sin
from scipy.ndimage import distance_transform_edt

import create_solver
from generate_MPC_config import generate_config
from model_nn import Autoencoder_path
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
            repo_root / "NPField" / "dataset" / "trained-models" / "NPField_Dynamic_10_A100.pth",
        )
    )
    checkpoint_path = Path(finetune_checkpoint).expanduser() if finetune_checkpoint else default_checkpoint
    output_dir = npfield_dir / "output"
    return dataset_root, checkpoint_path, output_dir


def test_solver(
    acados_solver,
    x_ref_points,
    y_ref_points,
    theta_0,
    num_map,
    ax1,
    map_inp,
    dyn_obst_info,
    obstacle_traj=None,
):
    def _wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _compute_feasible_tf(path_length: float, total_turn_rad: float) -> float:
        """Minimum feasible travel time for 1D speed profile with accel/decel limits."""
        v_cruise = max(1e-3, float(V_MAX))
        a_acc = max(1e-3, float(CTRL_A_MAX))
        a_dec = max(1e-3, float(-CTRL_A_MIN))
        w_max = max(1e-3, float(CTRL_W_MAX))

        d_acc = (v_cruise * v_cruise) / (2.0 * a_acc)
        d_dec = (v_cruise * v_cruise) / (2.0 * a_dec)

        if path_length >= (d_acc + d_dec):
            t_min = (v_cruise / a_acc) + ((path_length - d_acc - d_dec) / v_cruise) + (v_cruise / a_dec)
        else:
            # Triangular profile (never reaches V_MAX).
            v_peak = math.sqrt(max(0.0, 2.0 * path_length / (1.0 / a_acc + 1.0 / a_dec)))
            t_min = (v_peak / a_acc) + (v_peak / a_dec)

        t_turn = TF_TURN_WEIGHT * (total_turn_rad / w_max)
        # Add robust slack + buffer so horizon does not truncate unfinished episodes.
        return max(path_length / v_cruise, TF_TIME_SLACK * (t_min + t_turn) + TF_MIN_BUFFER_SEC)

    nx = 5
    nu = 3
    ny = nx + nu
    N = N_HORIZON
    yref = np.zeros([N, ny + 1])

    v_0 = 0
    v_e = 0

    x_ref = []
    y_ref = []
    theta = []
    theta_ref = []
    init_x = []
    init_y = []
    init_theta = []
    len_segments = []
    theta = np.append(theta, theta_0)  # current orientation robot
    theta_ref = np.append(theta_ref, theta_0)
    num_segment = len(x_ref_points) - 1
    length_path = 0
    for i in range(num_segment):
        length_path = length_path + math.sqrt(
            (x_ref_points[i + 1] - x_ref_points[i]) ** 2
            + (y_ref_points[i + 1] - y_ref_points[i]) ** 2
        )
        theta = np.append(
            theta,
            math.atan2(
                y_ref_points[i + 1] - y_ref_points[i],
                x_ref_points[i + 1] - x_ref_points[i],
            ),
        )
        len_segments = np.append(
            len_segments,
            math.sqrt(
                (x_ref_points[i + 1] - x_ref_points[i]) ** 2
                + (y_ref_points[i + 1] - y_ref_points[i]) ** 2
            ),
        )

    step_line = length_path / N

    print("length path", length_path)

    total_turn_rad = 0.0
    for i in range(len(theta) - 1):
        total_turn_rad += abs(_wrap_to_pi(theta[i + 1] - theta[i]))
    base_desired_tf = _compute_feasible_tf(length_path, total_turn_rad)

    k = 0
    x_ref = np.append(x_ref, x_ref_points[0])
    y_ref = np.append(y_ref, y_ref_points[0])
    for i in range(N + 1):
        x_ref = np.append(x_ref, x_ref[i] + step_line * math.cos(theta[k + 1]))
        y_ref = np.append(y_ref, y_ref[i] + step_line * math.sin(theta[k + 1]))
        theta_ref = np.append(theta_ref, theta[k + 1])
        d = math.sqrt((x_ref[-1] - x_ref_points[k]) ** 2 + (y_ref[-1] - y_ref_points[k]) ** 2)
        if d > len_segments[k] and k < (num_segment - 1):
            k = k + 1
            x_ref[i] = x_ref_points[k]
            y_ref[i] = y_ref_points[k]
        elif k > (num_segment - 1):
            break
    x0 = np.array([x_ref_points[0], y_ref_points[0], v_0, theta_0, 0])

    init_x = x_ref[0 : N + 1]
    init_y = y_ref[0 : N + 1]
    init_theta = theta_ref[0 : N + 1]
    yref[:, 0] = init_x[0:N]
    yref[:, 1] = init_y[0:N]
    yref[:, 2] = V_MAX
    yref[:, 3] = init_theta[0:N]

    a = np.zeros(1)

    simX = np.zeros((N + 1, 5))
    simU = np.zeros((N, nu))

    t = time.perf_counter()
    status = 1
    max_attempts = 3
    tf_growth_factors = (1.1, 1.3)
    goal_reach_tol_m = 0.18
    final_goal_error_m = float("inf")
    selected_tf = base_desired_tf
    selected_growth = 1.0
    best_key = None
    best_candidate = None
    for growth in tf_growth_factors:
        desired_tf = base_desired_tf * growth
        # Stage references should align with stage times [0, tf) (terminal uses tf).
        yref[:, 4] = np.linspace(0, desired_tf, N, endpoint=False)
        x_goal = np.array([init_x[-1], init_y[-1], v_e, init_theta[-1], desired_tf])
        yref_e = np.concatenate([x_goal, a])
        x_traj_init = np.transpose([yref[:, 0], yref[:, 1], yref[:, 2], yref[:, 3], yref[:, 4]])
        time_scale = desired_tf / SOLVER_BASE_TF

        for i in range(N):
            acados_solver.set(i, "y_ref", yref[i])
            acados_solver.set(i, "x", x_traj_init[i])
            acados_solver.set(i, "u", np.array([0.0, 0.0, time_scale]))
            # Keep a fixed time-scale factor so physical horizon matches desired_tf.
            acados_solver.set(i, "lbu", np.array([CTRL_A_MIN, CTRL_W_MIN, time_scale]))
            acados_solver.set(i, "ubu", np.array([CTRL_A_MAX, CTRL_W_MAX, time_scale]))
        acados_solver.set(N, "y_ref", yref_e)
        acados_solver.set(N, "x", x_goal)
        acados_solver.set(0, "lbx", x0)
        acados_solver.set(0, "ubx", x0)

        status = 1
        for _ in range(max_attempts):
            status = acados_solver.solve()
            if status == 0:
                break

        candidate_simX = np.zeros((N + 1, 5))
        for i in range(N + 1):
            x = acados_solver.get(i, "x")
            candidate_simX[i, 0] = x[0]
            candidate_simX[i, 1] = x[1]
            candidate_simX[i, 2] = x[2]
            candidate_simX[i, 3] = x[3]
            candidate_simX[i, 4] = x[4]
        candidate_goal_error_m = math.hypot(
            candidate_simX[-1, 0] - x_ref_points[-1], candidate_simX[-1, 1] - y_ref_points[-1]
        )
        candidate_simU = np.zeros((N, nu))
        for i in range(N):
            candidate_simU[i, :] = acados_solver.get(i, "u")
        candidate_cost = float(acados_solver.get_cost())
        candidate_key = (0 if candidate_goal_error_m < goal_reach_tol_m else 1, candidate_cost)
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_candidate = {
                "status": status,
                "goal_error_m": candidate_goal_error_m,
                "cost": candidate_cost,
                "simX": candidate_simX.copy(),
                "simU": candidate_simU.copy(),
                "desired_tf": desired_tf,
                "growth": growth,
            }

        print(
            f"TF candidate {desired_tf:.2f}s (x{growth:.2f}) -> "
            f"status={status}, terminal goal error={candidate_goal_error_m:.3f} m, cost={candidate_cost:.3f}"
        )
    if best_candidate is None:
        raise RuntimeError("Failed to evaluate any TF candidate.")
    status = best_candidate["status"]
    final_goal_error_m = best_candidate["goal_error_m"]
    selected_tf = best_candidate["desired_tf"]
    selected_growth = best_candidate["growth"]
    simX[:, :] = best_candidate["simX"]
    simU[:, :] = best_candidate["simU"]
    cost = float(best_candidate["cost"])
    print(
        f"Selected TF candidate {selected_tf:.2f}s (x{selected_growth:.2f}) with "
        f"status={status}, terminal goal error={final_goal_error_m:.3f} m, cost={cost:.3f}"
    )
    print("status", status)
    if status != 0:
        print(
            f"WARNING: acados did not fully converge after {max_attempts} attempts; "
            "trajectory may be suboptimal."
        )
    if final_goal_error_m > goal_reach_tol_m:
        print(
            f"WARNING: terminal error is {final_goal_error_m:.3f} m with tf={selected_tf:.2f}s; "
            "episode may visually stop before fully reaching the goal."
        )
    elapsed = 1000 * (time.perf_counter() - t)
    print(f"Trajectory solve time (ms): {elapsed:.2f}")
    ROB_x = np.zeros([N + 1, 9])
    ROB_y = np.zeros([N + 1, 9])
    for i in range(N + 1):
        simX[i, 4] = round(simX[i, 4], 1)
        ROB_x[i, 0] = simX[i, 0] + 0.6 * cos(simX[i, 3] - 0.59)
        ROB_x[i, 1] = simX[i, 0] + 0.514 * cos(simX[i, 3] - 0.24)
        ROB_x[i, 2] = simX[i, 0] + 0.75 * cos(simX[i, 3] - 0.16)
        ROB_x[i, 3] = simX[i, 0] + 0.75 * cos(simX[i, 3] + 0.16)
        ROB_x[i, 4] = simX[i, 0] + 0.514 * cos(simX[i, 3] + 0.24)
        ROB_x[i, 5] = simX[i, 0] + 0.6 * cos(simX[i, 3] + 0.59)
        ROB_x[i, 6] = simX[i, 0] - 0.6 * cos(simX[i, 3] - 0.59)
        ROB_x[i, 7] = simX[i, 0] - 0.6 * cos(simX[i, 3] + 0.59)
        ROB_x[i, 8] = simX[i, 0] + 0.6 * cos(simX[i, 3] - 0.59)
        ROB_y[i, 0] = simX[i, 1] + 0.6 * sin(simX[i, 3] - 0.59)
        ROB_y[i, 1] = simX[i, 1] + 0.514 * sin(simX[i, 3] - 0.24)
        ROB_y[i, 2] = simX[i, 1] + 0.75 * sin(simX[i, 3] - 0.16)
        ROB_y[i, 3] = simX[i, 1] + 0.75 * sin(simX[i, 3] + 0.16)
        ROB_y[i, 4] = simX[i, 1] + 0.514 * sin(simX[i, 3] + 0.24)
        ROB_y[i, 5] = simX[i, 1] + 0.6 * sin(simX[i, 3] + 0.59)
        ROB_y[i, 6] = simX[i, 1] - 0.6 * sin(simX[i, 3] - 0.59)
        ROB_y[i, 7] = simX[i, 1] - 0.6 * sin(simX[i, 3] + 0.59)
        ROB_y[i, 8] = simX[i, 1] + 0.6 * sin(simX[i, 3] - 0.59)

    initial_path = np.zeros((N + 1, 3))

    for i in range(N + 1):
        initial_path[i, 0] = init_x[i]
        initial_path[i, 1] = init_y[i]
        initial_path[i, 2] = init_theta[i]

    print("cost", cost)

    if num_map == -1:
        ax1.plot(simX[:, 0], simX[:, 1], linewidth=4, marker="o")
        ax1.plot(init_x, init_y, marker="o")
        ax1.set_aspect("equal", "box")
        ax1.plot([3.5, 5.1, 5.1, 3.5, 3.5], [1.52, 1.52, 3.12, 3.12, 1.52], linewidth=2)
        ax1.plot([1.8, 2.6, 2.6, 1.8, 1.8], [3.92, 3.92, 4.72, 4.72, 3.92], linewidth=2)
        ax1.plot([0.6, 2, 2, 0.6, 0.6], [1.12, 1.12, 2.52, 2.52, 1.12], linewidth=2)
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        for i in range(N + 1):
            simX[i, 0] = simX[i, 0] * MAP_SCALE
            simX[i, 1] = simX[i, 1] * MAP_SCALE
            init_x[i] = init_x[i] * MAP_SCALE
            init_y[i] = init_y[i] * MAP_SCALE
            ROB_x[i, 0] = ROB_x[i, 0] * MAP_SCALE
            ROB_x[i, 1] = ROB_x[i, 1] * MAP_SCALE
            ROB_x[i, 2] = ROB_x[i, 2] * MAP_SCALE
            ROB_x[i, 3] = ROB_x[i, 3] * MAP_SCALE
            ROB_x[i, 4] = ROB_x[i, 4] * MAP_SCALE
            ROB_x[i, 5] = ROB_x[i, 5] * MAP_SCALE
            ROB_x[i, 6] = ROB_x[i, 6] * MAP_SCALE
            ROB_x[i, 7] = ROB_x[i, 7] * MAP_SCALE
            ROB_x[i, 8] = ROB_x[i, 8] * MAP_SCALE
            ROB_y[i, 0] = ROB_y[i, 0] * MAP_SCALE
            ROB_y[i, 1] = ROB_y[i, 1] * MAP_SCALE
            ROB_y[i, 2] = ROB_y[i, 2] * MAP_SCALE
            ROB_y[i, 3] = ROB_y[i, 3] * MAP_SCALE
            ROB_y[i, 4] = ROB_y[i, 4] * MAP_SCALE
            ROB_y[i, 5] = ROB_y[i, 5] * MAP_SCALE
            ROB_y[i, 6] = ROB_y[i, 6] * MAP_SCALE
            ROB_y[i, 7] = ROB_y[i, 7] * MAP_SCALE
            ROB_y[i, 8] = ROB_y[i, 8] * MAP_SCALE

        ax1.plot(simX[:, 0], simX[:, 1], linewidth=2, label="NPField path")
        ax1.plot(init_x, init_y, linestyle="dashed", linewidth=2, label="Initial path")
        ax1.legend()
        scale_x = MAP_SCALE
        scale_y = MAP_SCALE
        ticks_x = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale_x))
        ax1.xaxis.set_major_formatter(ticks_x)

        ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale_y))
        ax1.yaxis.set_major_formatter(ticks_y)

    colors_list = [
        "k",
        "k",
        "r",
        "r",
        "g",
        "g",
        "b",
        "b",
        "c",
        "c",
        "m",
        "m",
        "y",
        "y",
        "k",
        "k",
        "r",
        "r",
        "g",
        "g",
        "b",
        "b",
        "c",
        "c",
        "m",
        "m",
        "y",
        "y",
        "k",
        "k",
    ]
    for i in range(0, N, 2):
        ax1.plot(
            [ROB_x[i, 0], ROB_x[i, 5], ROB_x[i, 6], ROB_x[i, 7], ROB_x[i, 0]],
            [ROB_y[i, 0], ROB_y[i, 5], ROB_y[i, 6], ROB_y[i, 7], ROB_y[i, 0]],
            color=colors_list[i],
        )
        ax1.text(i * 2.5, 1, str(round(simX[i, 4], 1)) + "  ,  ", color=colors_list[i])

    if obstacle_traj is not None:
        x_obst = obstacle_traj[0, 0]
        y_obst = obstacle_traj[0, 1]
        theta_obst = obstacle_traj[0, 2]
    else:
        x_obst = dyn_obst_info["initial_position"][num_map, 0]
        y_obst = dyn_obst_info["initial_position"][num_map, 1]
        theta_obst = dyn_obst_info["initial_position"][num_map, 2]
    OBST_x = np.zeros([TIME_STEPS, 4])
    OBST_y = np.zeros([TIME_STEPS, 4])
    OBST_x[0, 0] = x_obst + OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
    OBST_x[0, 1] = x_obst + OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
    OBST_x[0, 2] = x_obst - OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
    OBST_x[0, 3] = x_obst - OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
    OBST_y[0, 0] = y_obst + OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
    OBST_y[0, 1] = y_obst + OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
    OBST_y[0, 2] = y_obst - OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
    OBST_y[0, 3] = y_obst - OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
    ax1.plot(
        [
            OBST_x[0, 0] * MAP_SCALE,
            OBST_x[0, 1] * MAP_SCALE,
            OBST_x[0, 2] * MAP_SCALE,
            OBST_x[0, 3] * MAP_SCALE,
            OBST_x[0, 0] * MAP_SCALE,
        ],
        [
            OBST_y[0, 0] * MAP_SCALE,
            OBST_y[0, 1] * MAP_SCALE,
            OBST_y[0, 2] * MAP_SCALE,
            OBST_y[0, 3] * MAP_SCALE,
            OBST_y[0, 0] * MAP_SCALE,
        ],
        color="k",
    )
    colors_list = ["r", "g", "b", "c", "m", "y", "k", "r", "g"]
    ax1.text(0, 5, str(0) + ", ", color=colors_list[0])
    for i in range(TIME_STEPS - 1):
        if obstacle_traj is not None:
            x_obst = obstacle_traj[i + 1, 0]
            y_obst = obstacle_traj[i + 1, 1]
            theta_obst = obstacle_traj[i + 1, 2]
        else:
            x_obst = x_obst + OBSTACLE_STEP_M * cos(theta_obst)
            y_obst = y_obst + OBSTACLE_STEP_M * sin(theta_obst)
        OBST_x[i + 1, 0] = x_obst + OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
        OBST_x[i + 1, 1] = x_obst + OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
        OBST_x[i + 1, 2] = x_obst - OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
        OBST_x[i + 1, 3] = x_obst - OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
        OBST_y[i + 1, 0] = y_obst + OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
        OBST_y[i + 1, 1] = y_obst + OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
        OBST_y[i + 1, 2] = y_obst - OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
        OBST_y[i + 1, 3] = y_obst - OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
        ax1.plot(
            [
                OBST_x[i + 1, 0] * MAP_SCALE,
                OBST_x[i + 1, 1] * MAP_SCALE,
                OBST_x[i + 1, 2] * MAP_SCALE,
                OBST_x[i + 1, 3] * MAP_SCALE,
                OBST_x[i + 1, 0] * MAP_SCALE,
            ],
            [
                OBST_y[i + 1, 0] * MAP_SCALE,
                OBST_y[i + 1, 1] * MAP_SCALE,
                OBST_y[i + 1, 2] * MAP_SCALE,
                OBST_y[i + 1, 3] * MAP_SCALE,
                OBST_y[i + 1, 0] * MAP_SCALE,
            ],
            color=colors_list[i],
        )
        ax1.text(
            1.5 + i * 2.5,
            5,
            str(round(OBSTACLE_PRED_DT + i * OBSTACLE_PRED_DT, 1)) + "  ,  ",
            color=colors_list[i],
        )
    path_mpc = simX

    return path_mpc, elapsed, ROB_x, ROB_y


def gif_generate(
    path,
    ROB_x,
    ROB_y,
    num_map,
    id_map,
    map_data,
    dyn_obst_info,
    output_dir: Path,
    x_ref_points,
    y_ref_points,
    obstacle_traj=None,
    potential_frames=None,
):
    def _interp_pose(path_array, t):
        times = path_array[:, 4]
        if t <= times[0]:
            return path_array[0, 0], path_array[0, 1], path_array[0, 3]
        if t >= times[-1]:
            return path_array[-1, 0], path_array[-1, 1], path_array[-1, 3]
        idx = int(np.searchsorted(times, t, side="right"))
        idx = max(1, min(idx, len(times) - 1))
        t0 = times[idx - 1]
        t1 = times[idx]
        if abs(t1 - t0) < 1e-6:
            return path_array[idx, 0], path_array[idx, 1], path_array[idx, 3]
        alpha = (t - t0) / (t1 - t0)
        x = path_array[idx - 1, 0] + alpha * (path_array[idx, 0] - path_array[idx - 1, 0])
        y = path_array[idx - 1, 1] + alpha * (path_array[idx, 1] - path_array[idx - 1, 1])
        theta = path_array[idx - 1, 3] + alpha * (path_array[idx, 3] - path_array[idx - 1, 3])
        return x, y, theta

    def _robot_polygon(x_pos, y_pos, theta):
        scale = MAP_SCALE
        x0 = x_pos + scale * 0.6 * cos(theta - 0.59)
        y0 = y_pos + scale * 0.6 * sin(theta - 0.59)
        x5 = x_pos + scale * 0.6 * cos(theta + 0.59)
        y5 = y_pos + scale * 0.6 * sin(theta + 0.59)
        x6 = x_pos - scale * 0.6 * cos(theta - 0.59)
        y6 = y_pos - scale * 0.6 * sin(theta - 0.59)
        x7 = x_pos - scale * 0.6 * cos(theta + 0.59)
        y7 = y_pos - scale * 0.6 * sin(theta + 0.59)
        return [x0, x5, x6, x7, x0], [y0, y5, y6, y7, y0]

    if obstacle_traj is not None:
        x_obst = obstacle_traj[0, 0]
        y_obst = obstacle_traj[0, 1]
        theta_obst = obstacle_traj[0, 2]
    else:
        x_obst = dyn_obst_info["initial_position"][num_map, 0]
        y_obst = dyn_obst_info["initial_position"][num_map, 1]
        theta_obst = dyn_obst_info["initial_position"][num_map, 2]
    robot_path_x = path[:, 0]
    robot_path_y = path[:, 1]
    cfg_start_x = float(x_ref_points[0]) * MAP_SCALE
    cfg_start_y = float(y_ref_points[0]) * MAP_SCALE
    cfg_goal_x = float(x_ref_points[-1]) * MAP_SCALE
    cfg_goal_y = float(y_ref_points[-1]) * MAP_SCALE
    obst_path_x = [x_obst * MAP_SCALE]
    obst_path_y = [y_obst * MAP_SCALE]
    if obstacle_traj is not None:
        for i in range(1, obstacle_traj.shape[0]):
            obst_path_x.append(obstacle_traj[i, 0] * MAP_SCALE)
            obst_path_y.append(obstacle_traj[i, 1] * MAP_SCALE)
    else:
        for _ in range(TIME_STEPS - 1):
            x_obst = x_obst + OBSTACLE_STEP_M * cos(theta_obst)
            y_obst = y_obst + OBSTACLE_STEP_M * sin(theta_obst)
            obst_path_x.append(x_obst * MAP_SCALE)
            obst_path_y.append(y_obst * MAP_SCALE)
    frames = []
    t_obst = OBSTACLE_PRED_DT
    k_obst = 0
    k_robot = 0

    OBST_x = np.zeros([4])
    OBST_y = np.zeros([4])
    if obstacle_traj is not None:
        x_obst = obstacle_traj[0, 0]
        y_obst = obstacle_traj[0, 1]
        theta_obst = obstacle_traj[0, 2]
    else:
        x_obst = dyn_obst_info["initial_position"][num_map, 0]
        y_obst = dyn_obst_info["initial_position"][num_map, 1]
    OBST_x[0] = x_obst + OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
    OBST_x[1] = x_obst + OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
    OBST_x[2] = x_obst - OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
    OBST_x[3] = x_obst - OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
    OBST_y[0] = y_obst + OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
    OBST_y[1] = y_obst + OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
    OBST_y[2] = y_obst - OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
    OBST_y[3] = y_obst - OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
    total_time = max(15.0, float(path[-1, 4]))
    if obstacle_traj is not None:
        bg_map = _draw_obstacle_on_map(
            map_data[num_map][0],
            obstacle_traj[0, 0], obstacle_traj[0, 1], obstacle_traj[0, 2],
        )
    else:
        bg_map = map_data[num_map][0]
    map_h, map_w = bg_map.shape
    for i in np.arange(0.0, total_time + 1e-6, 0.05):
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        if potential_frames is not None:
            pred_idx = min(int(max(i, 0.0) / OBSTACLE_PRED_DT), TIME_STEPS - 1)
            ax2.imshow(
                potential_frames[pred_idx].T,
                origin="lower",
                extent=(0.0, map_w, 0.0, map_h),
                cmap="viridis",
                alpha=0.95,
                interpolation="nearest",
            )
            ax2.pcolor(
                bg_map[::-1],
                cmap=colors.ListedColormap(["white", "black"]),
                edgecolors="none",
                alpha=0.12,
            )
        else:
            cmap = colors.ListedColormap(["white", "black"])
            ax2.pcolor(bg_map[::-1], cmap=cmap, edgecolors="w", linewidths=0.1)
        ax2.plot(robot_path_x, robot_path_y, color="r", linewidth=1)
        # Show configured start/goal from generate_config, not optimized endpoints.
        ax2.plot(cfg_start_x, cfg_start_y, marker="x", color="r", markersize=6)
        ax2.plot(cfg_goal_x, cfg_goal_y, marker="o", color="r", markersize=4)
        ax2.plot(obst_path_x, obst_path_y, color="b", linewidth=1)
        ax2.plot(obst_path_x[0], obst_path_y[0], marker="x", color="b", markersize=6)
        ax2.plot(obst_path_x[-1], obst_path_y[-1], marker="o", color="b", markersize=4)
        ax2.plot(
            [
                OBST_x[0] * MAP_SCALE,
                OBST_x[1] * MAP_SCALE,
                OBST_x[2] * MAP_SCALE,
                OBST_x[3] * MAP_SCALE,
                OBST_x[0] * MAP_SCALE,
            ],
            [
                OBST_y[0] * MAP_SCALE,
                OBST_y[1] * MAP_SCALE,
                OBST_y[2] * MAP_SCALE,
                OBST_y[3] * MAP_SCALE,
                OBST_y[0] * MAP_SCALE,
            ],
            color="k",
        )
        interp_x, interp_y, interp_theta = _interp_pose(path, i)
        poly_x, poly_y = _robot_polygon(interp_x, interp_y, interp_theta)
        ax2.plot(poly_x, poly_y, color="k")
        if k_robot < len(path) - 1 and i > path[k_robot, 4]:
            k_robot += 1
        if round(i, 1) >= t_obst and k_obst < (TIME_STEPS - 1):
            if obstacle_traj is not None:
                x_obst = obstacle_traj[k_obst + 1, 0]
                y_obst = obstacle_traj[k_obst + 1, 1]
                theta_obst = obstacle_traj[k_obst + 1, 2]
            else:
                x_obst = x_obst + OBSTACLE_STEP_M * cos(theta_obst)
                y_obst = y_obst + OBSTACLE_STEP_M * sin(theta_obst)
            OBST_x[0] = x_obst + OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
            OBST_x[1] = x_obst + OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
            OBST_x[2] = x_obst - OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
            OBST_x[3] = x_obst - OBSTACLE_FOOTPRINT_RADIUS * cos(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
            OBST_y[0] = y_obst + OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
            OBST_y[1] = y_obst + OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
            OBST_y[2] = y_obst - OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst - OBSTACLE_FOOTPRINT_ANGLE)
            OBST_y[3] = y_obst - OBSTACLE_FOOTPRINT_RADIUS * sin(theta_obst + OBSTACLE_FOOTPRINT_ANGLE)
            t_obst = t_obst + OBSTACLE_PRED_DT
            k_obst += 1
        fig2.canvas.draw()
        data = np.asarray(fig2.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(data)
        plt.close(fig2)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"D2_MAP_ep{num_map}_ID_{id_map}.gif"
    imageio.mimsave(str(output_path), frames, format="GIF", fps=20)


def _draw_obstacle_on_map(base_map, x_obst, y_obst, theta_obst):
    """Render the obstacle footprint onto a copy of the 50x50 base map.

    Grid mapping: col = x / 0.1, row = (5 - y) / 0.1.
    Occupied cells are set to 100.
    """
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


def fill_map_inp(
    num_map,
    map_data,
    footprint,
    obst_initial_position,
    use_static_base=False,
):
    map_inp = torch.zeros((TIME_STEPS, 5003))
    fp_flat = torch.tensor(footprint.flatten(), dtype=torch.float32)

    for n in range(TIME_STEPS):
        if use_static_base:
            frame = _draw_obstacle_on_map(
                map_data[num_map][0],
                obst_initial_position[n][0],
                obst_initial_position[n][1],
                obst_initial_position[n][2],
            )
            map_inp[n, :2500] = torch.tensor(frame.flatten(), dtype=torch.float32)
        else:
            map_inp[n, :2500] = torch.tensor(
                map_data[num_map][n + 1].flatten(), dtype=torch.float32,
            )
        map_inp[n, 2500:5000] = fp_flat

    map_inp[:, :5000] /= 100.0
    for i in range(TIME_STEPS):
        map_inp[i, -3] = obst_initial_position[i][0]
        map_inp[i, -2] = obst_initial_position[i][1]
        map_inp[i, -1] = obst_initial_position[i][2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return map_inp.to(device)


_SOLVER_CACHE = {}
_MAP_INP_CACHE = {}


def compute_embedding(model_loaded, map_inp):
    # map_inp[0] is the first obstacle-conditioned step for this episode.
    encoded, _ = model_loaded.encode_map_footprint(map_inp[0])
    return encoded.detach().cpu().numpy()


def encode_map_for_potential(model_loaded, map_inp_row):
    if map_inp_row.dim() == 1:
        map_inp_row = map_inp_row.unsqueeze(0)
    with torch.no_grad():
        return model_loaded.encode_map_footprint(map_inp_row)


def build_obstacle_trajectory(obstacle_path, steps=TIME_STEPS):
    (x0, y0), (x1, y1) = obstacle_path
    dx = x1 - x0
    dy = y1 - y0
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


def build_test_episode_config():
    x_ref_points = [1.0, 4.0]
    y_ref_points = [2.5, 2.5]
    theta_0 = -1.0
    obstacle_path = ((3.0, 1.6), (1.0, 1.6))
    obstacle_traj = build_obstacle_trajectory(obstacle_path, steps=TIME_STEPS)
    return x_ref_points, y_ref_points, theta_0, obstacle_traj


def add_static_circle_obstacle(map_data, num_map, center=(2.5, 2.5), radius=1.0):
    cx, cy = center
    rows, cols = map_data[num_map][0].shape
    x_coords = np.arange(cols) * 0.1
    y_coords = 5.0 - np.arange(rows) * 0.1
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    mask = (grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= radius**2
    for layer in range(map_data[num_map].shape[0]):
        map_data[num_map][layer][mask] = 100.0


def sample_theta_from_path(path_mpc, steps=TIME_STEPS):
    sample_times = np.arange(steps, dtype=float) * OBSTACLE_PRED_DT
    path_times = path_mpc[:, 4].astype(float)
    path_theta = path_mpc[:, 3].astype(float)
    if path_times.size == 0:
        return np.zeros(steps, dtype=float)
    # Avoid interpolation artifacts when heading wraps around +/-pi.
    theta_unwrapped = np.unwrap(path_theta)
    sampled_unwrapped = np.interp(
        sample_times,
        path_times,
        theta_unwrapped,
        left=theta_unwrapped[0],
        right=theta_unwrapped[-1],
    )
    return (sampled_unwrapped + math.pi) % (2.0 * math.pi) - math.pi


def infer_potential_grid(model_loaded, encoded, heading_samples, device, chunk_size):
    xs = np.linspace(0.0, 5.0, POTENTIAL_RESOLUTION, endpoint=False)
    ys = np.linspace(0.0, 5.0, POTENTIAL_RESOLUTION, endpoint=False)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)

    if encoded.dim() == 1:
        encoded = encoded.unsqueeze(0)
    encoded = encoded.to(device=device, dtype=torch.float32)
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    encoded_rep = encoded.repeat(coords_t.shape[0], 1)
    potential_frames = np.zeros((TIME_STEPS, POTENTIAL_RESOLUTION, POTENTIAL_RESOLUTION), dtype=np.float32)
    with torch.no_grad():
        for pred_idx, angle in enumerate(heading_samples):
            theta = torch.full((coords_t.shape[0], 1), float(angle), dtype=torch.float32, device=device)
            input_batch = torch.hstack((encoded_rep, coords_t, theta))
            outputs = []
            for start in range(0, input_batch.shape[0], chunk_size):
                chunk = input_batch[start : start + chunk_size]
                outputs.append(model_loaded.encode_map_pos(chunk).cpu())
            output = torch.cat(outputs, dim=0).numpy()
            potential_frames[pred_idx] = output[:, pred_idx].reshape(POTENTIAL_RESOLUTION, POTENTIAL_RESOLUTION)
    return potential_frames


def save_potential_gif(
    map_data,
    num_map,
    id_map,
    output_dir,
    potential_frames,
):
    frames = []
    for tme in range(TIME_STEPS):
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, figsize=(12, 5), gridspec_kw={"wspace": 0.3, "hspace": 0.1}
        )
        ax1.imshow(map_data[num_map][tme])
        ax2.imshow(potential_frames[tme].T, origin="lower")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"NPField_D2_potential_ep{num_map}_ID_{id_map}.gif"
    # Set loop=0 for infinite looping of the gif
    imageio.mimsave(str(output_path), frames, format="GIF", loop=0)


def get_acados_solver(model_loaded, map_inp, cache_key, allow_backward=False):
    if cache_key in _SOLVER_CACHE:
        return _SOLVER_CACHE[cache_key]
    embedding_values = compute_embedding(model_loaded, map_inp)
    acados_solver = create_solver.create_solver(
        model_loaded, embedding_values, allow_backward=allow_backward
    )
    _SOLVER_CACHE[cache_key] = acados_solver
    return acados_solver


def run_simulation(
    num_map,
    x_ref_points,
    y_ref_points,
    theta_0,
    obst_motion_info,
    map_data,
    footprint,
    model_loaded,
    dyn_obst_info,
    output_dir,
    save_potential_gif_flag,
    potential_chunk_size,
    id_map=0,
    obst_initial_position=None,
    obstacle_traj=None,
    save_gif=True,
    interactive=True,
    allow_backward=False,
):
    t_total = time.perf_counter()
    id_dyn = 0
    if obst_initial_position is None:
        obst_initial_position = obst_motion_info[
            "motion_dynamic_obst"
        ][num_map, id_dyn * TIME_STEPS : (id_dyn + 1) * TIME_STEPS]

    cache_key = (num_map, id_dyn, allow_backward) if obst_initial_position is None else None
    if cache_key is not None and cache_key in _MAP_INP_CACHE:
        map_inp = _MAP_INP_CACHE[cache_key]
    else:
        map_inp = fill_map_inp(
            num_map,
            map_data,
            footprint,
            obst_initial_position,
            use_static_base=(obstacle_traj is not None),
        )
        if cache_key is not None:
            _MAP_INP_CACHE[cache_key] = map_inp

    if cache_key is not None:
        acados_solver = get_acados_solver(
            model_loaded, map_inp, cache_key=cache_key, allow_backward=allow_backward
        )
    else:
        embedding_values = compute_embedding(model_loaded, map_inp)
        acados_solver = create_solver.create_solver(
            model_loaded, embedding_values, allow_backward=allow_backward
        )

    if interactive:
        cmap = colors.ListedColormap(["white", "black"])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_box_aspect(1)
        ax.pcolor(map_data[num_map][0][::-1], cmap=cmap, edgecolors="w", linewidths=0.1)
    else:
        fig, ax = plt.subplots(figsize=(1, 1))

    path_mpc, elapsed, ROB_x, ROB_y = test_solver(
        acados_solver,
        x_ref_points,
        y_ref_points,
        theta_0,
        num_map,
        ax,
        map_inp,
        dyn_obst_info,
        obstacle_traj=obstacle_traj,
    )
    total_elapsed_ms = 1000 * (time.perf_counter() - t_total)
    print(f"Trajectory time without solver init (ms): {elapsed:.2f}")
    print(f"Total planning time (ms): {total_elapsed_ms:.2f}")
    plt.close(fig)

    potential_frames = None
    if save_potential_gif_flag and save_gif:
        device = map_inp.device.type
        encoded, _ = encode_map_for_potential(model_loaded, map_inp[0])
        heading_samples = sample_theta_from_path(path_mpc, steps=TIME_STEPS)
        potential_frames = infer_potential_grid(
            model_loaded=model_loaded,
            encoded=encoded,
            heading_samples=heading_samples,
            device=device,
            chunk_size=potential_chunk_size,
        )

    if save_gif:
        gif_generate(
            path_mpc,
            ROB_x,
            ROB_y,
            num_map,
            id_map,
            map_data,
            dyn_obst_info,
            output_dir,
            x_ref_points,
            y_ref_points,
            obstacle_traj=obstacle_traj,
            potential_frames=potential_frames,
        )

    if interactive:
        fig, ax2 = plt.subplots(1)
        ax2.plot(path_mpc[:, 4], path_mpc[:, 2])
        ax2.grid()
        ax2.set_ylim([0, 1.1])
        plt.setp(ax2, ylabel="v (m/sec)")
        plt.show(block=False)

    return path_mpc, elapsed, total_elapsed_ms


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


def load_model(checkpoint_path, device):
    model_loaded = Autoencoder_path(mode="k")
    pretrained_dict = torch.load(checkpoint_path, map_location="cpu")
    model_loaded.load_state_dict(pretrained_dict)
    model_loaded.to(device)
    model_loaded.eval()
    return model_loaded


def compute_trajectory_metrics(path_mpc, map_data, num_map, obstacle_traj, total_elapsed_ms, goal_xy=None):
    """Compute evaluation metrics for trajectory quality.

    Original metrics (all lower-is-better except safety_distance):
        time_ms           – end-to-end planning time (encoding + MPC solve)
        path_length_m     – Euclidean path length in metres
        smoothness        – Σ(Δθ²) / path_length
        aol               – Σ|Δθ| / path_length  (angle-over-length)
        safety_distance_m – min SDF at robot footprint, incl. dynamic obstacle
        goal_error_m      – Euclidean distance from final pose to goal
        collision / success

    Additional metrics:
        path_efficiency       – straight_line / path_length ∈ (0,1], higher = more direct
        mean_clearance_m      – mean per-waypoint clearance, higher = safer overall
        velocity_utilization  – mean(|v|) / V_MAX ∈ [0,1], higher = faster traversal
        control_energy        – Σ(â²+ω̂²)·Δt, lower = less actuator effort
        jerk                  – RMS of (Δâ/Δt, Δω̂/Δt), lower = smoother control
        spl                   – Success weighted by Path Length (Anderson et al. 2018)
    """
    N = path_mpc.shape[0] - 1
    xs = path_mpc[:, 0] / MAP_SCALE
    ys = path_mpc[:, 1] / MAP_SCALE
    thetas = path_mpc[:, 3]
    times = path_mpc[:, 4]
    velocities = path_mpc[:, 2]

    # --- Path length & heading changes (AOL per bench-mr) ---
    points = np.column_stack((xs, ys))
    filtered_points = [points[0]]
    for p in points[1:]:
        if np.hypot(p[0] - filtered_points[-1][0], p[1] - filtered_points[-1][1]) > 1e-9:
            filtered_points.append(p)
    filtered_points = np.array(filtered_points, dtype=float)

    if filtered_points.shape[0] >= 2:
        segments = np.sqrt(np.diff(filtered_points[:, 0]) ** 2 + np.diff(filtered_points[:, 1]) ** 2)
        path_length = float(np.sum(segments))
    else:
        path_length = 0.0

    if filtered_points.shape[0] >= 3:
        seg_dx = np.diff(filtered_points[:, 0])
        seg_dy = np.diff(filtered_points[:, 1])
        yaws = np.arctan2(seg_dy, seg_dx)
        dtheta = np.diff(yaws)
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    else:
        dtheta = np.array([], dtype=float)

    smoothness = float(np.sum(dtheta**2) / max(path_length, 1e-6))
    aol = float(np.sum(np.abs(dtheta)) / max(path_length, 1e-6))

    # --- Path efficiency ---
    if goal_xy is not None:
        straight_line = math.hypot(goal_xy[0] - xs[0], goal_xy[1] - ys[0])
    else:
        straight_line = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    path_efficiency = min(straight_line / max(path_length, 1e-6), 1.0)

    # --- Velocity utilization ---
    mean_speed = float(np.mean(np.abs(velocities)))
    velocity_utilization = mean_speed / float(V_MAX) if float(V_MAX) > 0 else 0.0

    # --- Control energy & jerk (estimated from state trajectory) ---
    dt_arr = np.diff(times)
    dt_arr = np.where(np.abs(dt_arr) < 1e-9, 1e-9, dt_arr)
    acc_est = np.diff(velocities) / dt_arr
    dtheta_ctrl = np.diff(thetas)
    dtheta_ctrl = (dtheta_ctrl + np.pi) % (2 * np.pi) - np.pi
    omega_est = dtheta_ctrl / dt_arr

    control_energy = float(np.sum((acc_est ** 2 + omega_est ** 2) * dt_arr))

    if len(acc_est) >= 2:
        dt_inner = dt_arr[:-1]
        dt_inner = np.where(np.abs(dt_inner) < 1e-9, 1e-9, dt_inner)
        jerk_a = np.diff(acc_est) / dt_inner
        jerk_w = np.diff(omega_est) / dt_inner
        jerk = float(np.sqrt(np.mean(jerk_a ** 2 + jerk_w ** 2)))
    else:
        jerk = 0.0

    # --- Safety clearance (min and mean per-waypoint) ---
    # Build per-timestep SDF grids with the dynamic obstacle drawn at its
    # actual position, using the same rendering as the GIF generator.
    base_map = map_data[num_map][0]
    sdf_per_step = []
    for step in range(len(obstacle_traj)):
        combined = _draw_obstacle_on_map(
            base_map,
            obstacle_traj[step, 0], obstacle_traj[step, 1], obstacle_traj[step, 2],
        )
        binary_occ = (combined > 50).astype(np.float64)
        sdf_per_step.append(distance_transform_edt(1 - binary_occ) * 0.1)

    fp_params = [
        (0.6, -0.59), (0.6, 0.59), (-0.6, -0.59), (-0.6, 0.59),
        (0.75, -0.16), (0.75, 0.16),
    ]

    min_safety = float("inf")
    per_wp_clearance = []
    for k in range(N + 1):
        rx, ry, rtheta = xs[k], ys[k], thetas[k]
        t_val = times[k]

        corners = [
            (rx + r * cos(rtheta + a), ry + r * sin(rtheta + a))
            for r, a in fp_params
        ]
        corners.append((rx, ry))

        obst_idx = min(int(max(t_val, 0.0) / OBSTACLE_PRED_DT), len(sdf_per_step) - 1)
        sdf_grid = sdf_per_step[obst_idx]

        wp_min = float("inf")
        for cx, cy in corners:
            gi = max(0, min(49, int(round((5 - cy) / 0.1))))
            gj = max(0, min(49, int(round(cx / 0.1))))
            wp_min = min(wp_min, sdf_grid[gi, gj])

        min_safety = min(min_safety, wp_min)
        per_wp_clearance.append(wp_min)

    mean_clearance = float(np.mean(per_wp_clearance))
    collision = bool(min_safety <= 0)

    if goal_xy is not None:
        goal_error = math.hypot(xs[-1] - goal_xy[0], ys[-1] - goal_xy[1])
    else:
        goal_error = float("nan")

    success = (not collision) and (goal_error <= 0.18)

    # --- SPL (Success weighted by Path Length, Anderson et al. 2018) ---
    if path_length > 0 and straight_line > 0:
        spl = float(success) * (straight_line / max(path_length, straight_line))
    else:
        spl = 0.0

    return {
        "time_ms": round(float(total_elapsed_ms), 2),
        "path_length_m": round(float(path_length), 4),
        "smoothness": round(float(smoothness), 6),
        "aol": round(float(aol), 6),
        "safety_distance_m": round(float(min_safety), 4),
        "goal_error_m": round(float(goal_error), 4),
        "collision": bool(collision),
        "success": bool(success),
        "path_efficiency": round(float(path_efficiency), 4),
        "mean_clearance_m": round(float(mean_clearance), 4),
        "velocity_utilization": round(float(velocity_utilization), 4),
        "control_energy": round(float(control_energy), 4),
        "jerk": round(float(jerk), 4),
        "spl": round(float(spl), 4),
    }


def run_benchmark(
    benchmark_json_path,
    model_loaded,
    map_data,
    footprint,
    dyn_obst_info,
    obst_motion_info,
    costmap,
    output_dir,
    save_potential_gif_flag=False,
    potential_chunk_size=4096,
    allow_backward=False,
):
    # Ensure Tera renderer is available; if missing, auto-download without prompting (e.g. Docker / CI)
    try:
        from acados_template.utils import get_tera, get_tera_exec_path

        tera_path = get_tera_exec_path()
        if not (os.path.exists(tera_path) and os.access(tera_path, os.X_OK)):
            get_tera(force_download=True)
    except Exception as e:
        print(f"Warning: could not pre-install Tera renderer: {e}")

    with open(benchmark_json_path) as f:
        benchmark = json.load(f)
    scenarios = benchmark["scenarios"]
    benchmark_output_dir = output_dir / "benchmark"
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = benchmark_output_dir / "benchmark_metrics.csv"
    json_path = benchmark_output_dir / "benchmark_metrics.json"
    print(f"\n=== Benchmark: {len(scenarios)} scenarios from {benchmark_json_path} ===\n")

    csv_columns = [
        "id",
        "map_id",
        "time_ms",
        "path_length_m",
        "smoothness",
        "aol",
        "safety_distance_m",
        "goal_error_m",
        "path_efficiency",
        "mean_clearance_m",
        "velocity_utilization",
        "control_energy",
        "jerk",
        "spl",
        "collision",
        "success",
        "status",
        "error",
    ]
    numeric_cols = [
        "time_ms", "path_length_m", "smoothness", "aol",
        "safety_distance_m", "goal_error_m",
        "path_efficiency", "mean_clearance_m", "velocity_utilization",
        "control_energy", "jerk", "spl",
    ]

    results = []
    completed = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)
        f.flush()
        try:
            for scenario in scenarios:
                sid = scenario["id"]
                map_id = scenario["map_id"]
                x_path = scenario["x_path"]
                y_path = scenario["y_path"]
                theta_initial = scenario["theta_initial"]
                obstacle_start = tuple(scenario["obstacle_start"])
                obstacle_end = tuple(scenario["obstacle_end"])
                obstacle_traj = build_obstacle_trajectory(
                    (obstacle_start, obstacle_end), steps=TIME_STEPS
                )

                print(f"--- Scenario {sid} (map={map_id}) ---")
                try:
                    path_mpc, _solve_elapsed, total_elapsed = run_simulation(
                        num_map=map_id,
                        x_ref_points=x_path,
                        y_ref_points=y_path,
                        theta_0=theta_initial,
                        obst_motion_info=obst_motion_info,
                        map_data=map_data,
                        footprint=footprint,
                        model_loaded=model_loaded,
                        dyn_obst_info=dyn_obst_info,
                        output_dir=benchmark_output_dir,
                        save_potential_gif_flag=save_potential_gif_flag,
                        potential_chunk_size=potential_chunk_size,
                        id_map=sid,
                        obst_initial_position=obstacle_traj,
                        obstacle_traj=obstacle_traj,
                        save_gif=True,
                        interactive=False,
                        allow_backward=allow_backward,
                    )
                    goal_xy = (x_path[-1], y_path[-1])
                    metrics = compute_trajectory_metrics(
                        path_mpc, map_data, map_id, obstacle_traj, total_elapsed, goal_xy
                    )
                    metrics["scenario_id"] = sid
                    metrics["map_id"] = map_id
                    metrics["status"] = "ok"
                    metrics["error"] = ""
                except Exception as e:
                    print(f"  FAILED: {e}")
                    metrics = {
                        "scenario_id": sid,
                        "map_id": map_id,
                        "status": "failed",
                        "error": str(e),
                        "time_ms": "",
                        "path_length_m": "",
                        "smoothness": "",
                        "aol": "",
                        "safety_distance_m": "",
                        "goal_error_m": "",
                        "path_efficiency": "",
                        "mean_clearance_m": "",
                        "velocity_utilization": "",
                        "control_energy": "",
                        "jerk": "",
                        "spl": "",
                        "collision": "",
                        "success": False,
                    }
                results.append(metrics)
                completed += 1
                writer.writerow(
                    [
                        metrics["scenario_id"],
                        metrics["map_id"],
                        metrics["time_ms"],
                        metrics["path_length_m"],
                        metrics["smoothness"],
                        metrics["aol"],
                        metrics["safety_distance_m"],
                        metrics["goal_error_m"],
                        metrics["path_efficiency"],
                        metrics["mean_clearance_m"],
                        metrics["velocity_utilization"],
                        metrics["control_energy"],
                        metrics["jerk"],
                        metrics["spl"],
                        metrics["collision"],
                        metrics["success"],
                        metrics["status"],
                        metrics["error"],
                    ]
                )
                # Persist partial benchmark table even for very long runs.
                f.flush()
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user. Partial CSV and JSON were saved.")

        evaluated = [r for r in results if r.get("status") == "ok"]
        successful = [r for r in evaluated if r.get("success", False)]
        n_eval = len(evaluated)
        n_success_for_avg = len(successful)
        if n_success_for_avg > 0:
            avg_row = [
                "AVERAGE (successful only)",
                "",
                round(np.mean([r["time_ms"] for r in successful]), 2),
                round(np.mean([r["path_length_m"] for r in successful]), 4),
                round(np.mean([r["smoothness"] for r in successful]), 6),
                round(np.mean([r["aol"] for r in successful]), 6),
                round(np.mean([r["safety_distance_m"] for r in successful]), 4),
                round(np.mean([r["goal_error_m"] for r in successful]), 4),
                round(np.mean([r["path_efficiency"] for r in successful]), 4),
                round(np.mean([r["mean_clearance_m"] for r in successful]), 4),
                round(np.mean([r["velocity_utilization"] for r in successful]), 4),
                round(np.mean([r["control_energy"] for r in successful]), 4),
                round(np.mean([r["jerk"] for r in successful]), 4),
                round(np.mean([r["spl"] for r in successful]), 4),
                0,  # no collisions in successful episodes
                f"{n_success_for_avg}/{n_eval}",
                "summary",
                "",
            ]
            writer.writerow(avg_row)
            f.flush()

    def _to_serializable(obj):
        """Convert numpy/types to native Python for JSON."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(
            {
                "num_scenarios": len(scenarios),
                "num_completed": completed,
                "per_scenario": [_to_serializable(r) for r in results],
            },
            f,
            indent=2,
        )

    n_success = sum(1 for r in results if r.get("success", False))
    n_collision = sum(1 for r in results if r.get("collision", False))
    print(f"\n{'='*60}")
    print(f"Benchmark complete: {completed}/{len(scenarios)} scenarios evaluated")
    print(f"  Success rate: {n_success}/{len(results)}")
    print(f"  Collisions:   {n_collision}/{len(results)}")
    if successful:
        print(f"{'='*60}")
        print(f"  {'Metric':<22s} {'Mean (successful only)':>22s}")
        print(f"  {'-'*22} {'-'*22}")
        for col in numeric_cols:
            vals = [r[col] for r in successful]
            print(f"  {col:<22s} {np.mean(vals):>22.4f}")
    print(f"\nCSV saved to {csv_path}")
    print(f"JSON saved to {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run NPField solver.")
    parser.add_argument("--map-id", type=int, default=3, help="Map index to use.")
    parser.add_argument("--num-orientation", type=int, default=0, help="Orientation index.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of runs.")
    parser.add_argument(
        "--easy-config",
        action="store_true",
        help="Use easier deterministic agent start/goal pairs when possible.",
    )
    parser.add_argument(
        "--save-potential-gif",
        action="store_true",
        help="Overlay potential field on the trajectory GIF.",
    )
    parser.add_argument(
        "--potential-chunk-size",
        type=int,
        default=4096,
        help="Batch size for potential grid inference.",
    )
    parser.add_argument(
        "--test-episode",
        action="store_true",
        help=(
            "Run a deterministic scenario: agent (1.0,2.5)->(4.0,2.5), "
            "dynamic obstacle starts at (3.0,2.5) moving toward the agent, "
            "plus a static circular obstacle (r=1.0m) at (2.5,2.5)."
        ),
    )
    parser.add_argument(
        "--finetune-checkpoint",
        type=str,
        default="",
        help="Optional path to a finetuned D2 checkpoint; overrides NPFIELD_CHECKPOINT.",
    )
    parser.add_argument(
        "--benchmark-json",
        type=str,
        default="",
        help="Path to JSON file with benchmark scenarios. Runs all scenarios and saves metrics.",
    )
    parser.add_argument(
        "--allow-backward",
        action="store_true",
        help="Allow negative velocity (backward motion) in MPC trajectory planning.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root, checkpoint_path, output_dir = resolve_paths(args.finetune_checkpoint)
    map_data, footprint, dyn_obst_info, obst_motion_info, costmap = load_datasets(dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loaded = load_model(checkpoint_path, device)

    if args.benchmark_json:
        run_benchmark(
            benchmark_json_path=args.benchmark_json,
            model_loaded=model_loaded,
            map_data=map_data,
            footprint=footprint,
            dyn_obst_info=dyn_obst_info,
            obst_motion_info=obst_motion_info,
            costmap=costmap,
            output_dir=output_dir,
            save_potential_gif_flag=args.save_potential_gif,
            potential_chunk_size=args.potential_chunk_size,
            allow_backward=args.allow_backward,
        )
        return

    if args.test_episode:
        # Copy map tensor to keep deterministic test obstacle local to this run.
        map_data = np.array(map_data, copy=True)
        #add_static_circle_obstacle(map_data, args.map_id, center=(2.5, 2.5), radius=0.2)
        x_ref_points, y_ref_points, theta_0, obstacle_traj = build_test_episode_config()
        print("Running deterministic --test-episode scenario.")

    for i in range(args.episodes):
        if not args.test_episode:
            x_ref_points, y_ref_points, theta_0, obstacle_path = generate_config(
                costmap,
                args.map_id,
                args.num_orientation,
                return_obstacle=True,
                easy_config=args.easy_config,
            )
            obstacle_traj = build_obstacle_trajectory(obstacle_path, steps=TIME_STEPS)
        run_simulation(
            num_map=args.map_id,
            x_ref_points=x_ref_points,
            y_ref_points=y_ref_points,
            theta_0=theta_0,
            obst_motion_info=obst_motion_info,
            map_data=map_data,
            footprint=footprint,
            model_loaded=model_loaded,
            dyn_obst_info=dyn_obst_info,
            output_dir=output_dir,
            save_potential_gif_flag=args.save_potential_gif,
            potential_chunk_size=args.potential_chunk_size,
            id_map=i,
            obst_initial_position=obstacle_traj,
            obstacle_traj=obstacle_traj,
            save_gif=True,
            allow_backward=args.allow_backward,
        )


if __name__ == "__main__":
    main()