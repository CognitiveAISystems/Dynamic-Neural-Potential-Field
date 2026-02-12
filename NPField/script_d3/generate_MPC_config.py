import argparse
import os
import random
from math import atan2, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import colors

MAP_MIN = 0.8
MAP_MAX = 4.2
GRID_RESOLUTION = 0.10
AGENT_MAX_RADIUS = 1.2
FREE_THRESHOLD = 1.8
MIN_AGENT_GOAL_DIST = 1.0
MAX_AGENT_GOAL_DIST = 3.0
WALL_CLEARANCE = 0.6
MIN_PATH_EPS = 1.2


def _grid_index(x, y):
    i = int(round((5 - y) / GRID_RESOLUTION))
    j = int(round(x / GRID_RESOLUTION))
    return i, j


def _in_bounds(x, y):
    return MAP_MIN <= x <= MAP_MAX and MAP_MIN <= y <= MAP_MAX


def _is_free(costmaps, num_map, num_orientation, x, y, threshold=FREE_THRESHOLD, layer_offset=6):
    i, j = _grid_index(x, y)
    if i < 0 or j < 0:
        return False
    if i >= costmaps[num_map][0].shape[0] or j >= costmaps[num_map][0].shape[1]:
        return False
    layer = (num_orientation * 10) + layer_offset
    return costmaps[num_map][layer][i][j] > threshold


def _distance(x0, y0, x1, y1):
    return sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def _valid_agent_path(x_start, y_start, x_goal, y_goal):
    vals = np.array([x_start, y_start, x_goal, y_goal], dtype=float)
    if not np.all(np.isfinite(vals)):
        return False
    d = _distance(x_start, y_start, x_goal, y_goal)
    if d <= MIN_PATH_EPS:
        return False
    if d < MIN_AGENT_GOAL_DIST or d > MAX_AGENT_GOAL_DIST:
        return False
    return True


def _wall_clearance(x, y):
    return min(x - MAP_MIN, MAP_MAX - x, y - MAP_MIN, MAP_MAX - y)


def _point_segment_distance(px, py, x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy
    if denom < 1e-9:
        return _distance(px, py, x0, y0)
    t = ((px - x0) * dx + (py - y0) * dy) / denom
    t = max(0.0, min(1.0, t))
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    return _distance(px, py, proj_x, proj_y)


def _segment_distance(a0, a1, b0, b1):
    ax0, ay0 = a0
    ax1, ay1 = a1
    bx0, by0 = b0
    bx1, by1 = b1
    return min(
        _point_segment_distance(ax0, ay0, bx0, by0, bx1, by1),
        _point_segment_distance(ax1, ay1, bx0, by0, bx1, by1),
        _point_segment_distance(bx0, by0, ax0, ay0, ax1, ay1),
        _point_segment_distance(bx1, by1, ax0, ay0, ax1, ay1),
    )


def _max_step_to_bounds(x, y, dir_x, dir_y):
    limits = []
    if abs(dir_x) > 1e-9:
        for bound in (MAP_MIN, MAP_MAX):
            t = (bound - x) / dir_x
            if t > 0:
                limits.append(t)
    if abs(dir_y) > 1e-9:
        for bound in (MAP_MIN, MAP_MAX):
            t = (bound - y) / dir_y
            if t > 0:
                limits.append(t)
    if not limits:
        return 0.0
    return min(limits)


def _sample_obstacle_path(
    costmaps,
    num_map,
    num_orientation,
    agent_start,
    agent_goal,
    agent_radius=AGENT_MAX_RADIUS,
    attempts=15000,
    length_range=(1.4, 3.2),
):
    ax0, ay0 = agent_start
    ax1, ay1 = agent_goal
    dx = ax1 - ax0
    dy = ay1 - ay0
    dist = sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        return None
    dir_x = dx / dist
    dir_y = dy / dist

    margin = min(0.3, agent_radius / dist)
    t_min = margin
    t_max = 1.0 - margin
    if t_min >= t_max:
        t_min = 0.2
        t_max = 0.8

    for _ in range(attempts):
        t = np.random.uniform(t_min, t_max)
        ix = ax0 + t * dx
        iy = ay0 + t * dy
        angle = np.random.uniform(0, 2 * np.pi)
        dir2_x = np.cos(angle)
        dir2_y = np.sin(angle)
        if abs(dir2_x * dir_x + dir2_y * dir_y) > 0.995:
            continue

        max_pos = _max_step_to_bounds(ix, iy, dir2_x, dir2_y)
        max_neg = _max_step_to_bounds(ix, iy, -dir2_x, -dir2_y)
        max_half = min(max_pos, max_neg)
        if max_half <= 0.0:
            continue

        min_half = 0.5 * length_range[0]
        max_half = min(max_half, 0.5 * length_range[1])
        if max_half < min_half:
            continue
        half = np.random.uniform(min_half, max_half)
        ox0 = ix + dir2_x * half
        oy0 = iy + dir2_y * half
        ox1 = ix - dir2_x * half
        oy1 = iy - dir2_y * half

        if not (_in_bounds(ox0, oy0) and _in_bounds(ox1, oy1)):
            continue
        if not (_is_free(costmaps, num_map, num_orientation, ox0, oy0) and _is_free(costmaps, num_map, num_orientation, ox1, oy1)):
            continue
        if _distance(ax0, ay0, ox0, oy0) < agent_radius:
            continue
        if _distance(ax1, ay1, ox1, oy1) < agent_radius:
            continue

        return (ox0, oy0), (ox1, oy1)

    return None


def generate_config(
    costmaps,
    num_map,
    num_orientation,
    return_obstacle=False,
    max_retries=200,
    easy_config=False,
):

    if easy_config:
        candidate_pairs = [
            ((1.0, 1.0), (4.0, 4.0)),
            ((1.0, 4.0), (4.0, 1.0)),
            ((1.2, 1.2), (3.8, 3.8)),
            ((1.2, 3.8), (3.8, 1.2)),
        ]
        for (x_start, y_start), (x_goal, y_goal) in candidate_pairs:
            if not _valid_agent_path(x_start, y_start, x_goal, y_goal):
                continue
            if _wall_clearance(x_start, y_start) < WALL_CLEARANCE:
                continue
            if _wall_clearance(x_goal, y_goal) < WALL_CLEARANCE:
                continue
            if not _is_free(costmaps, num_map, num_orientation, x_start, y_start, layer_offset=1):
                continue
            if not _is_free(costmaps, num_map, num_orientation, x_goal, y_goal, layer_offset=10):
                continue
            x_middle = (x_start + x_goal) / 2
            y_middle = (y_start + y_goal) / 2
            if _wall_clearance(x_middle, y_middle) < WALL_CLEARANCE:
                continue
            if not _is_free(
                costmaps,
                num_map,
                num_orientation,
                x_middle,
                y_middle,
                threshold=2.0,
                layer_offset=6,
            ):
                continue

            x_path = [x_start, x_goal]
            y_path = [y_start, y_goal]
            obstacle_path = _sample_obstacle_path(
                costmaps,
                num_map,
                num_orientation,
                (x_start, y_start),
                (x_goal, y_goal),
            )
            if obstacle_path is None:
                continue
            if _wall_clearance(obstacle_path[0][0], obstacle_path[0][1]) < WALL_CLEARANCE:
                continue
            if _wall_clearance(obstacle_path[1][0], obstacle_path[1][1]) < WALL_CLEARANCE:
                continue
            if (
                _segment_distance(
                    (x_start, y_start),
                    (x_goal, y_goal),
                    obstacle_path[0],
                    obstacle_path[1],
                )
                < AGENT_MAX_RADIUS
            ):
                continue

            theta_initial = atan2(y_path[1] - y_path[0], x_path[1] - x_path[0])
            print("num_map = ", num_map, "num_orientaion = ", num_orientation)

            if not _valid_agent_path(x_path[0], y_path[0], x_path[1], y_path[1]):
                continue

            if return_obstacle:
                return x_path, y_path, theta_initial, obstacle_path

            return x_path, y_path, theta_initial

    for _ in range(max_retries):
        x_path = []
        y_path = []
        found = False

        for _ in range(4000):
            x_start = np.random.uniform(MAP_MIN, MAP_MAX)
            y_start = np.random.uniform(MAP_MIN, MAP_MAX)
            i = round((5 - y_start) / GRID_RESOLUTION)
            j = round(x_start / GRID_RESOLUTION)
            if costmaps[num_map][(num_orientation * 10) + 1][i][j] <= FREE_THRESHOLD:
                continue

            for _ in range(2000):
                x_goal = np.random.uniform(MAP_MIN, MAP_MAX)
                y_goal = np.random.uniform(MAP_MIN, MAP_MAX)
                i_g = round((5 - y_goal) / GRID_RESOLUTION)
                j_g = round(x_goal / GRID_RESOLUTION)
                if costmaps[num_map][(num_orientation * 10) + 10][i_g][j_g] <= FREE_THRESHOLD:
                    continue
                if not _valid_agent_path(x_start, y_start, x_goal, y_goal):
                    continue

                x_middle = (x_start + x_goal) / 2
                y_middle = (y_start + y_goal) / 2
                i_middle = round((5 - y_middle) / GRID_RESOLUTION)
                j_middle = round(x_middle / GRID_RESOLUTION)
                if costmaps[num_map][(num_orientation * 10) + 6][i_middle][j_middle] <= FREE_THRESHOLD:
                    continue

                x_path = [x_start, x_goal]
                y_path = [y_start, y_goal]
                found = True
                break
            if found:
                break

        if not found:
            continue

        agent_start = (x_path[0], y_path[0])
        agent_goal = (x_path[1], y_path[1])
        obstacle_path = _sample_obstacle_path(
            costmaps,
            num_map,
            num_orientation,
            agent_start,
            agent_goal,
        )
        if obstacle_path is None:
            continue

        theta_initial = atan2(y_path[1] - y_path[0], x_path[1] - x_path[0])
        print("num_map = ", num_map, "num_orientaion = ", num_orientation)

        if not _valid_agent_path(x_path[0], y_path[0], x_path[1], y_path[1]):
            continue

        if return_obstacle:
            return x_path, y_path, theta_initial, obstacle_path

        return x_path, y_path, theta_initial

    raise RuntimeError("Could not find a valid configuration after retries.")


def _resolve_paths():
    script_dir = Path(__file__).resolve().parent
    npfield_dir = script_dir.parent
    repo_root = npfield_dir.parent
    dataset_root = Path(
        os.getenv("NPFIELD_DATASET_DIR", repo_root / "NPField" / "dataset" / "dataset1000")
    )
    output_dir = npfield_dir / "output"
    return dataset_root, output_dir


def _load_datasets(dataset_root):
    sub_maps = pickle.load(open(dataset_root / "dataset_1000_maps_0_100_all.pkl", "rb"))
    costmaps = pickle.load(open(dataset_root / "dataset_1000_costmaps.pkl", "rb"))
    return sub_maps["submaps"], costmaps["costmap"]


def _plot_config(map_data, num_map, agent_path, obstacle_path, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    cmap = colors.ListedColormap(["white", "black"])
    ax.pcolor(map_data[num_map][0][::-1], cmap=cmap, edgecolors="w", linewidths=0.1)

    agent_x = np.array(agent_path[0]) * 10.0
    agent_y = np.array(agent_path[1]) * 10.0
    obstacle_x = np.array([obstacle_path[0][0], obstacle_path[1][0]]) * 10.0
    obstacle_y = np.array([obstacle_path[0][1], obstacle_path[1][1]]) * 10.0

    ax.plot(agent_x, agent_y, color="r", linewidth=2, label="Agent path")
    ax.plot(agent_x[0], agent_y[0], marker="x", color="r", markersize=6)
    ax.plot(agent_x[-1], agent_y[-1], marker="o", color="r", markersize=4)
    ax.plot(obstacle_x, obstacle_y, color="b", linewidth=2, label="Obstacle path")
    ax.plot(obstacle_x[0], obstacle_y[0], marker="x", color="b", markersize=6)
    ax.plot(obstacle_x[-1], obstacle_y[-1], marker="o", color="b", markersize=4)
    ax.set_aspect("equal", "box")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MPC configs with obstacle crossings.")
    parser.add_argument("--num-configs", "-n", type=int, default=1, help="Number of configs to generate.")
    parser.add_argument(
        "--easy-config",
        action="store_true",
        help="Use easier deterministic agent start/goal pairs when possible.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root, output_dir = _resolve_paths()
    map_data, costmaps = _load_datasets(dataset_root)

    created = 0
    attempts = 0
    max_attempts = args.num_configs * 20
    while created < args.num_configs and attempts < max_attempts:
        attempts += 1
        num_map = random.randint(0, len(costmaps) - 1)
        num_orientation = random.randint(0, 2)
        try:
            x_path, y_path, theta_initial, obstacle_path = generate_config(
                costmaps,
                num_map,
                num_orientation,
                return_obstacle=True,
                easy_config=args.easy_config,
            )
        except RuntimeError:
            continue
        output_path = output_dir / f"MPC_config_{num_map}_{created}.png"
        _plot_config(map_data, num_map, (x_path, y_path), obstacle_path, output_path)
        created += 1

    if created < args.num_configs:
        raise RuntimeError(
            f"Only generated {created}/{args.num_configs} configs after {attempts} attempts."
        )


if __name__ == "__main__":
    main()