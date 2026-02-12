import argparse
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
from math import cos, sin

import create_solver_GPT as create_solver
from generate_MPC_config import generate_config
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
            repo_root / "NPField" / "dataset" / "trained-models" / "NPField_onlyGPT_predmap9.pth",
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
    tf_growth_factors = (1.0, 2.0, 3.0)
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
        candidate_key = (0 if status == 0 else 1, candidate_goal_error_m, candidate_cost)
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
    map_h, map_w = map_data[num_map][0].shape
    for i in np.arange(0.0, total_time + 1e-6, 0.05):
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        if potential_frames is not None:
            # Keep the NN potential-map timing consistent with robot_model:
            # each prediction slice is active for one OBSTACLE_PRED_DT bin.
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
                map_data[num_map][0][::-1],
                cmap=colors.ListedColormap(["white", "black"]),
                edgecolors="none",
                alpha=0.12,
            )
        else:
            cmap = colors.ListedColormap(["white", "black"])
            ax2.pcolor(map_data[num_map][0][::-1], cmap=cmap, edgecolors="w", linewidths=0.1)
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
    output_path = output_dir / f"D3_MAP_ep{num_map}_ID_{id_map}.gif"
    imageio.mimsave(str(output_path), frames, format="GIF", fps=20)


def fill_map_inp(
    num_map,
    map_data,
    footprint,
    obst_initial_position,
    use_static_base=False,
):
    map_inp = torch.zeros((TIME_STEPS, 5003))
    k = 0
    for n in range(TIME_STEPS):
        layer_idx = 0 if use_static_base else (n + 1)
        for i in range(50):
            for j in range(50):
                map_inp[n][k] = map_data[num_map][layer_idx][i, j]
                k = k + 1
        k = 0

    k = 0
    for n in range(TIME_STEPS):
        for i in range(50):
            for j in range(50):
                map_inp[n][2500 + k] = footprint[i, j]
                k = k + 1
        k = 0

    map_inp[:, :5000] /= 100.0
    for i in range(TIME_STEPS):
        map_inp[i, -3] = obst_initial_position[i][0]
        map_inp[i, -2] = obst_initial_position[i][1]
        map_inp[i, -1] = obst_initial_position[i][2]
    return map_inp.cuda()


_SOLVER_CACHE = {}
_MAP_INP_CACHE = {}


def compute_embedding(model_loaded, map_inp):
    # map_inp[0] is the first obstacle-conditioned step for this episode.
    encoded = model_loaded.encode_map_footprint(map_inp[0])
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


def infer_potential_grid(model_loaded, encoded, angle, device, chunk_size):
    xs = np.linspace(0.0, 5.0, POTENTIAL_RESOLUTION, endpoint=False)
    ys = np.linspace(0.0, 5.0, POTENTIAL_RESOLUTION, endpoint=False)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)

    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    theta = torch.full((coords_t.shape[0], 1), angle, dtype=torch.float32, device=device)
    encoded_rep = encoded.repeat(coords_t.shape[0], 1)
    input_batch = torch.hstack((encoded_rep, coords_t, theta))

    outputs = []
    with torch.no_grad():
        for start in range(0, input_batch.shape[0], chunk_size):
            chunk = input_batch[start : start + chunk_size]
            outputs.append(model_loaded(chunk).cpu())
    output = torch.cat(outputs, dim=0)
    return (
        output.reshape(POTENTIAL_RESOLUTION, POTENTIAL_RESOLUTION, TIME_STEPS)
        .permute(2, 0, 1)
        .numpy()
    )


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
    output_path = output_dir / f"NPField_D3_potential_ep{num_map}_ID_{id_map}.gif"
    # Set loop=0 for infinite looping of the gif
    imageio.mimsave(str(output_path), frames, format="GIF", loop=0)


def get_acados_solver(model_loaded, map_inp, cache_key):
    if cache_key in _SOLVER_CACHE:
        return _SOLVER_CACHE[cache_key]
    embedding_values = compute_embedding(model_loaded, map_inp)
    acados_solver = create_solver.create_solver(model_loaded, embedding_values)
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
):
    id_dyn = 0
    if obst_initial_position is None:
        obst_initial_position = obst_motion_info[
            "motion_dynamic_obst"
        ][num_map, id_dyn * TIME_STEPS : (id_dyn + 1) * TIME_STEPS]

    cache_key = (num_map, id_dyn) if obst_initial_position is None else None
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
        acados_solver = get_acados_solver(model_loaded, map_inp, cache_key=cache_key)
    else:
        embedding_values = compute_embedding(model_loaded, map_inp)
        acados_solver = create_solver.create_solver(model_loaded, embedding_values)

    potential_frames = None
    if save_potential_gif_flag:
        device = map_inp.device.type
        encoded = encode_map_for_potential(model_loaded, map_inp[0])
        potential_frames = infer_potential_grid(
            model_loaded=model_loaded,
            encoded=encoded,
            angle=theta_0,
            device=device,
            chunk_size=potential_chunk_size,
        )

    cmap = colors.ListedColormap(["white", "black"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_box_aspect(1)
    ax.pcolor(map_data[num_map][0][::-1], cmap=cmap, edgecolors="w", linewidths=0.1)

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
    print(f"Trajectory time without solver init (ms): {elapsed:.2f}")
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
    if save_potential_gif_flag:
        save_potential_gif(
            map_data=map_data,
            num_map=num_map,
            id_map=id_map,
            output_dir=output_dir,
            potential_frames=potential_frames,
        )

    fig, ax2 = plt.subplots(1)
    ax2.plot(path_mpc[:, 4], path_mpc[:, 2])
    ax2.grid()
    ax2.set_ylim([0, 1.1])
    plt.setp(ax2, ylabel="v (m/sec)")
    plt.show(block=False)

    return path_mpc, elapsed


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


def _unwrap_state_dict(payload):
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Checkpoint payload is not a valid state_dict dict.")


def _infer_model_args_from_state_dict(state_dict):
    n_embd = 576
    if "x_encode.weight" in state_dict and hasattr(state_dict["x_encode.weight"], "shape"):
        n_embd = int(state_dict["x_encode.weight"].shape[0])

    layer_ids = set()
    for key in state_dict.keys():
        if key.startswith("transformer.h."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_ids.add(int(parts[2]))
    n_layer = (max(layer_ids) + 1) if layer_ids else 4

    n_head = 4
    if n_embd % n_head != 0:
        n_head = 1

    return dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=1024,
        bias=True,
        vocab_size=1024,
        dropout=0.1,
    )


def _load_partial_state_dict(model, state_dict):
    model_state = model.state_dict()
    matched = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped.append(f"{key}: ckpt{tuple(value.shape)} != model{tuple(model_state[key].shape)}")
            continue
        matched[key] = value
    model.load_state_dict(matched, strict=False)
    return matched, skipped


def load_model(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location="cpu")
    pretrained_dict = _unwrap_state_dict(payload)
    model_args = _infer_model_args_from_state_dict(pretrained_dict)
    print("Inferred model args:", model_args)

    gptconf = GPTConfig(**model_args)
    model_loaded = GPT(gptconf)
    matched, skipped = _load_partial_state_dict(model_loaded, pretrained_dict)
    print(
        "Checkpoint load summary:",
        f"loaded={len(matched)}",
        f"skipped_shape_mismatch={len(skipped)}",
    )
    if skipped:
        print("First shape mismatches:")
        for msg in skipped[:20]:
            print("  -", msg)
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    model_loaded.to(device)
    model_loaded.eval()
    return model_loaded


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
        help="Save potential field GIF for each run.",
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
        help="Optional path to a finetuned D3 checkpoint; overrides NPFIELD_CHECKPOINT.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root, checkpoint_path, output_dir = resolve_paths(args.finetune_checkpoint)
    map_data, footprint, dyn_obst_info, obst_motion_info, costmap = load_datasets(dataset_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loaded = load_model(checkpoint_path, device)

    if args.test_episode:
        # Copy map tensor to keep deterministic test obstacle local to this run.
        map_data = np.array(map_data, copy=True)
        # add_static_circle_obstacle(map_data, args.map_id, center=(2.5, 2.5), radius=0.2)
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
        )


if __name__ == "__main__":
    main()
