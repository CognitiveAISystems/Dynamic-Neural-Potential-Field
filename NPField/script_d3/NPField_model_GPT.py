import argparse
import os
import pickle
from pathlib import Path

import imageio
import math
import matplotlib.pyplot as plt
import numpy as np
import torch

from model_nn_GPT import GPT, GPTConfig


TIME_STEPS = 10
RESOLUTION = 100
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_paths() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    npfield_dir = script_dir.parent
    repo_root = npfield_dir.parent

    dataset_root = Path(
        os.getenv("NPFIELD_DATASET_DIR", repo_root / "NPField" / "dataset" / "dataset1000")
    )
    checkpoint_path = Path(
        os.getenv(
            "NPFIELD_CHECKPOINT",
            repo_root / "NPField" / "dataset" / "trained-models" / "NPField_onlyGPT_predmap9.pth",
        )
    )
    return dataset_root, checkpoint_path


def load_dataset(dataset_root: Path) -> dict:
    required = {
        "sub_maps_all": dataset_root / "dataset_1000_maps_0_100_all.pkl",
        "footprints": dataset_root / "data_footprint.pkl",
        "dyn_obst_info": dataset_root / "dataset_initial_position_dynamic_obst.pkl",
        "obst_motion_info": dataset_root / "dataset_1000_maps_obst_motion.pkl",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset files not found. Set NPFIELD_DATASET_DIR to the dataset1000 directory. "
            f"Missing: {', '.join(missing)}"
        )

    return {name: pickle.load(open(path, "rb")) for name, path in required.items()}


def build_model(checkpoint_path: Path, device: str) -> GPT:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Set NPFIELD_CHECKPOINT to the model .pth file."
        )

    model_args = dict(
        n_layer=4,
        n_head=4,
        n_embd=576,
        block_size=1024,
        bias=True,
        vocab_size=1024,
        dropout=0.1,
    )
    gptconf = GPTConfig(**model_args)
    model_gpt = GPT(gptconf)

    pretrained_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = model_gpt.state_dict()
    rejected_keys = [k for k in model_dict if k not in pretrained_dict]
    print("REJECTED KEYS test GPT:", rejected_keys)

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_gpt.load_state_dict(model_dict)

    model_gpt.to(device)
    model_gpt.eval()
    return model_gpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Dyn-NPField GPT inference and save a GIF."
    )
    parser.add_argument("episode", nargs="?", type=int, default=0, help="Episode index.")
    parser.add_argument("id_dyn", nargs="?", type=int, default=0, help="Dynamic obstacle id.")
    parser.add_argument("angle", nargs="?", type=float, default=0.0, help="Angle in degrees.")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Batch size for grid inference to limit memory use.",
    )
    return parser.parse_args()


def encode_map(model_gpt: GPT, test_map: np.ndarray, test_footprint: np.ndarray, d_info: np.ndarray, device: str) -> torch.Tensor:
    map_inp = (
        torch.tensor(
            np.hstack((test_map.flatten(), test_footprint.flatten())),
            dtype=torch.float32,
            device=device,
        )
        .unsqueeze(0)
        / 100.0
    )
    d_info_tensor = torch.tensor(d_info, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return model_gpt.encode_map_footprint(torch.hstack((map_inp, d_info_tensor)))


def infer_grid(
    model_gpt: GPT,
    encoded: torch.Tensor,
    angle: float,
    device: str,
    chunk_size: int,
) -> np.ndarray:
    print(f"Infer angle (rad): {angle:.6f}")
    xs = np.linspace(0.0, 5.0, RESOLUTION, endpoint=False)
    ys = np.linspace(0.0, 5.0, RESOLUTION, endpoint=False)
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
            outputs.append(model_gpt(chunk).cpu())
    output = torch.cat(outputs, dim=0)

    res = output.reshape(RESOLUTION, RESOLUTION, TIME_STEPS).permute(2, 0, 1).numpy()
    return res


def save_gif(frames: list[np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, format="GIF", loop=65535)
    print(str(output_path))


def main() -> None:
    args = parse_args()
    dataset_root, checkpoint_path = resolve_paths()
    data = load_dataset(dataset_root)

    angle_rad = math.radians(args.angle)
    print(
        f"Episode ID: {args.episode} ID_dyn: {args.id_dyn} "
        f"Angle: {args.angle} deg ({angle_rad:.6f} rad)"
    )
    print(f"Using dataset root: {dataset_root}")
    print(f"Using checkpoint: {checkpoint_path}")

    model_gpt = build_model(checkpoint_path, args.device)

    test_map = data["sub_maps_all"]["submaps"][args.episode, args.id_dyn]
    test_footprint = data["footprints"]["footprint_husky"]
    d_info = data["obst_motion_info"]["motion_dynamic_obst"][args.episode, args.id_dyn]

    encoded = encode_map(model_gpt, test_map, test_footprint, d_info, args.device)
    res_array = infer_grid(model_gpt, encoded, angle_rad, args.device, args.chunk_size)

    frames = []
    for tme in range(TIME_STEPS):
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, figsize=(15, 5), gridspec_kw={"wspace": 0.3, "hspace": 0.1}
        )
        ax1.imshow(data["sub_maps_all"]["submaps"][args.episode, tme])
        ax2.imshow(np.rot90(res_array[tme]))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    angle_tag = f"{args.angle:.1f}deg".replace(".", "p")
    npfield_dir = Path(__file__).resolve().parent.parent
    output_dir = npfield_dir / "output"
    gif_name = f"NPField_D3_ep{args.episode}_dyn{args.id_dyn}_angle_{angle_tag}.gif"
    save_gif(frames, output_dir / gif_name)


if __name__ == "__main__":
    main()