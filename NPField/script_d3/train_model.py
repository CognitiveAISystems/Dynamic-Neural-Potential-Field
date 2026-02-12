from __future__ import annotations

import argparse
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model_nn_GPT import GPT, GPTConfig


def _try_create_comet_experiment(
    project_name: str,
    workspace: str,
) -> Optional[object]:
    """
    Comet is optional:
    - If `comet_ml` is unavailable, training still runs.
    - If `COMET_API_KEY` is not set, training still runs.
    """
    from comet_ml import start  # type: ignore

    api_key = os.getenv("COMET_API_KEY")

    return start(api_key=api_key, project_name=project_name, workspace=workspace)


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def _first_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _resolve_paths() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    dataset_override = os.getenv("NPFIELD_DATASET_DIR")
    dataset_candidates = (
        [Path(dataset_override)]
        if dataset_override
        else [
            repo_root / "NPField" / "dataset" / "dataset1000",
            repo_root / "dataset" / "dataset1000",
            repo_root / "NPField" / "dataset",
        ]
    )
    dataset_root = _first_existing_path(dataset_candidates)

    trained_models_override = os.getenv("NPFIELD_TRAINED_MODELS_DIR")
    trained_models_candidates = (
        [Path(trained_models_override)]
        if trained_models_override
        else [
            repo_root / "trained-models",
            repo_root / "NPField" / "dataset" / "trained-models",
            repo_root / "NPField" / "trained-models",
        ]
    )
    trained_models_dir = _first_existing_path(trained_models_candidates)
    trained_models_dir.mkdir(parents=True, exist_ok=True)

    return dataset_root, trained_models_dir


def flat_to_indices(flat_idx: int, n_positions: int, n_theta: int) -> tuple[int, int, int]:
    per_map = n_positions * n_theta
    map_id = flat_idx // per_map
    rem = flat_idx % per_map
    pos_id = rem // n_theta
    theta_id = rem % n_theta
    return map_id, pos_id, theta_id


def _infer_num_dyn_from_motion(motion: np.ndarray, steps_per_dyn: int) -> Optional[int]:
    if motion.ndim == 4:
        # Most likely [N, num_dyn, steps, 3]
        if motion.shape[2] >= 1:
            return int(motion.shape[1])
    if motion.ndim == 3:
        # Most likely [N, num_dyn * steps, 3]
        if motion.shape[1] >= steps_per_dyn and motion.shape[1] % steps_per_dyn == 0:
            return int(motion.shape[1] // steps_per_dyn)
        if motion.shape[1] >= 1:
            return int(motion.shape[1])
    return None


def _motion_state_for_dyn(
    motion: np.ndarray,
    *,
    map_id: int,
    dyn_id: int,
    steps_per_dyn: int,
    time_offset: int,
) -> tuple[float, float, float]:
    if motion.ndim == 4:
        # [N, num_dyn, steps, xyz]
        if motion.shape[1] > dyn_id and motion.shape[2] > 0:
            x, y, theta = motion[map_id, dyn_id, 0, :3]
            return float(x), float(y), float(theta)
        # [N, steps, num_dyn, xyz]
        if motion.shape[2] > dyn_id and motion.shape[1] > 0:
            x, y, theta = motion[map_id, 0, dyn_id, :3]
            return float(x), float(y), float(theta)

    if motion.ndim == 3:
        # [N, num_dyn * steps, xyz]
        packed_idx = dyn_id * steps_per_dyn
        if motion.shape[1] > packed_idx:
            x, y, theta = motion[map_id, packed_idx, :3]
            return float(x), float(y), float(theta)

        # [N, time, xyz] aligned with map time axis (shift by time_offset).
        time_idx = dyn_id * steps_per_dyn + max(0, time_offset - 1)
        if motion.shape[1] > time_idx:
            x, y, theta = motion[map_id, time_idx, :3]
            return float(x), float(y), float(theta)

        # [N, num_dyn, xyz]
        if motion.shape[1] > dyn_id:
            x, y, theta = motion[map_id, dyn_id, :3]
            return float(x), float(y), float(theta)

        x, y, theta = motion[map_id, 0, :3]
        return float(x), float(y), float(theta)

    raise ValueError(f"Unsupported motion tensor shape: {motion.shape}")


class PotentialDatasetD3(Dataset):
    """
    D3 dataset sample:
      input: (map+footprint, x, y, theta, dyn_x, dyn_y, dyn_theta)
      output: future potential sequence (10 steps by default)
    """

    def __init__(
        self,
        mapp: np.ndarray,
        potential: np.ndarray,
        footprint: np.ndarray,
        motion_dynamic_obst: np.ndarray,
        *,
        num_dyn: Optional[int] = None,
        steps_per_dyn: int = 10,
        time_offset: int = 1,
        clamp_max: float = 30.0,
    ) -> None:
        self.map = mapp
        self.potential = potential
        self.footprint = footprint
        self.motion = motion_dynamic_obst

        self.n_positions = int(self.potential.shape[2])
        self.n_theta = int(self.potential.shape[3])
        self.steps_per_dyn = int(steps_per_dyn)
        self.time_offset = int(time_offset)
        self.clamp_max = float(clamp_max)

        candidates: list[int] = []
        pot_t = int(self.potential.shape[1])
        map_t = int(self.map.shape[1])
        if pot_t > self.time_offset and (pot_t - self.time_offset) % self.steps_per_dyn == 0:
            candidates.append((pot_t - self.time_offset) // self.steps_per_dyn)
        if map_t > self.time_offset and (map_t - self.time_offset) % self.steps_per_dyn == 0:
            candidates.append((map_t - self.time_offset) // self.steps_per_dyn)
        motion_dyn = _infer_num_dyn_from_motion(self.motion, self.steps_per_dyn)
        if motion_dyn is not None:
            candidates.append(int(motion_dyn))

        inferred_num_dyn = min(candidates) if candidates else 1
        if num_dyn is not None:
            inferred_num_dyn = min(inferred_num_dyn, int(num_dyn))
        self.num_dyn = max(1, int(inferred_num_dyn))

    def __len__(self) -> int:
        return int(self.potential.shape[0] * self.n_positions * self.n_theta)

    def __getitem__(self, flat_idx: int):
        map_id, pos_id, theta_id = flat_to_indices(flat_idx, self.n_positions, self.n_theta)
        dyn_id = random.randrange(self.num_dyn)

        t0 = dyn_id * self.steps_per_dyn + self.time_offset
        t_slice = slice(t0, t0 + self.steps_per_dyn)

        map_inp = (
            torch.tensor(np.stack((self.map[map_id][t0], self.footprint)), dtype=torch.float32) / 100.0
        )

        input_x = torch.tensor(self.potential[map_id, t0, pos_id, theta_id, 0], dtype=torch.float32)
        input_y = torch.tensor(self.potential[map_id, t0, pos_id, theta_id, 1], dtype=torch.float32)
        theta = torch.tensor(self.potential[map_id, t0, pos_id, theta_id, 2], dtype=torch.float32)

        dyn_x, dyn_y, dyn_theta = _motion_state_for_dyn(
            self.motion,
            map_id=map_id,
            dyn_id=dyn_id,
            steps_per_dyn=self.steps_per_dyn,
            time_offset=self.time_offset,
        )

        output = (
            torch.clamp(
                torch.tensor(self.potential[map_id, t_slice, pos_id, theta_id, 3], dtype=torch.float32),
                min=0.0,
                max=self.clamp_max,
            )
            / self.clamp_max
        )

        return {
            "input": (
                map_inp,
                input_x.unsqueeze(0),
                input_y.unsqueeze(0),
                theta.unsqueeze(0),
                torch.tensor([dyn_x], dtype=torch.float32),
                torch.tensor([dyn_y], dtype=torch.float32),
                torch.tensor([dyn_theta], dtype=torch.float32),
            ),
            "output": output,
            "meta": {
                "map_id": int(map_id),
                "dyn_id": int(dyn_id),
                "t0": int(t0),
                "pos_id": int(pos_id),
                "theta_id": int(theta_id),
            },
        }


@dataclass(frozen=True)
class TrainConfig:
    # Data split
    train_split: int = 990

    # Training
    epochs: int = 4
    lr: float = 1e-4
    grad_clip: float = 10.0

    # Batching
    batch_size: int = 256
    val_batch_size: int = 16
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False

    # Sequence and normalization
    steps_per_dyn: int = 10
    time_offset: int = 1
    potential_clip_max: float = 30.0

    # Checkpointing and validation
    checkpoint_every_steps: int = 1000
    val_every_steps: int = 1000
    max_val_batches: int = 100
    checkpoint_name: str = "NPField_D3_GPT_train.pth"

    # Comet (optional; requires COMET_API_KEY env var)
    comet_project: str = "npfield-d3-training"
    comet_workspace: str = "aleksei2"


def _make_dataloader(ds: Dataset, *, batch_size: int, shuffle: bool, cfg: TrainConfig) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        num_workers=max(0, int(cfg.num_workers)),
    )
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = max(1, int(cfg.prefetch_factor))
        kwargs["persistent_workers"] = bool(cfg.persistent_workers)
    return DataLoader(ds, **kwargs)


def _count_params(module: nn.Module, *, trainable_only: bool) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _format_int(n: int) -> str:
    return f"{n:,}"


def describe_model(model: nn.Module) -> str:
    total = _count_params(model, trainable_only=False)
    trainable = _count_params(model, trainable_only=True)
    return (
        f"Model class: {model.__class__.__name__}\n"
        f"Total params: {_format_int(total)}\n"
        f"Trainable params: {_format_int(trainable)}"
    )


def _require_key(blob: dict, key: str, *, source_name: str) -> np.ndarray:
    if key not in blob:
        raise KeyError(f"Missing key '{key}' in {source_name}")
    return np.asarray(blob[key])


def _load_partial_state_dict(model: nn.Module, payload: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Load only checkpoint tensors that match existing model keys and shapes.
    Returns (loaded_keys, skipped_shape_mismatch, missing_in_checkpoint).
    """
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    loaded_keys: list[str] = []
    skipped_shape: list[str] = []

    for key, tensor in payload.items():
        if key not in model_state:
            continue
        if model_state[key].shape != tensor.shape:
            skipped_shape.append(
                f"{key}: ckpt{tuple(tensor.shape)} != model{tuple(model_state[key].shape)}"
            )
            continue
        filtered[key] = tensor
        loaded_keys.append(key)

    missing_in_checkpoint = [k for k in model_state.keys() if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    return loaded_keys, skipped_shape, missing_in_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NPField D3 GPT model")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--val-batch-size", type=int, default=TrainConfig.val_batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-split", type=int, default=TrainConfig.train_split)
    parser.add_argument("--steps-per-dyn", type=int, default=TrainConfig.steps_per_dyn)
    parser.add_argument("--time-offset", type=int, default=TrainConfig.time_offset)
    parser.add_argument("--num-dyn", type=int, default=0, help="0 means infer from dataset")
    parser.add_argument("--potential-clip-max", type=float, default=TrainConfig.potential_clip_max)
    parser.add_argument("--checkpoint-every-steps", type=int, default=TrainConfig.checkpoint_every_steps)
    parser.add_argument("--val-every-steps", type=int, default=TrainConfig.val_every_steps)
    parser.add_argument("--max-val-batches", type=int, default=TrainConfig.max_val_batches)
    parser.add_argument("--checkpoint-name", type=str, default=TrainConfig.checkpoint_name)
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--no-comet", action="store_true", help="Disable Comet logging")
    parser.add_argument("--no-map-loss", action="store_true", help="Disable decoded map loss term")
    parser.add_argument("--use-map-loss", action="store_true", help="(deprecated) Enable map loss")
    parser.add_argument("--map-loss-weight", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=576)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    cfg = TrainConfig(
        train_split=int(args.train_split),
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        val_batch_size=int(args.val_batch_size),
        num_workers=int(args.num_workers),
        steps_per_dyn=int(args.steps_per_dyn),
        time_offset=int(args.time_offset),
        potential_clip_max=float(args.potential_clip_max),
        checkpoint_every_steps=int(args.checkpoint_every_steps),
        val_every_steps=int(args.val_every_steps),
        max_val_batches=int(args.max_val_batches),
        checkpoint_name=str(args.checkpoint_name),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch:", torch.__version__, "device:", device)

    dataset_root, trained_models_dir = _resolve_paths()
    print("dataset_root:", dataset_root)
    print("trained_models_dir:", trained_models_dir)

    potential_path = _first_existing_path(
        [
            dataset_root / "potentials" / "dataset_1000_potential_husky.pkl",
            dataset_root / "dataset_1000_potential_husky.pkl",
        ]
    )
    sub_maps_blob = _load_pickle(dataset_root / "dataset_1000_maps_0_100_all.pkl")
    footprint_blob = _load_pickle(dataset_root / "data_footprint.pkl")
    motion_blob = _load_pickle(dataset_root / "dataset_1000_maps_obst_motion.pkl")
    potential_blob = _load_pickle(potential_path)

    maps = _require_key(sub_maps_blob, "submaps", source_name="dataset_1000_maps_0_100_all.pkl")
    footprint = _require_key(footprint_blob, "footprint_husky", source_name="data_footprint.pkl")
    motion_dynamic = _require_key(
        motion_blob, "motion_dynamic_obst", source_name="dataset_1000_maps_obst_motion.pkl"
    )
    position_potential = _require_key(
        potential_blob, "position_potential", source_name=potential_path.name
    )

    if maps.shape[0] < 2:
        raise ValueError(f"Need at least 2 maps for train/val split, got {maps.shape[0]}")
    split = min(max(1, int(cfg.train_split)), int(maps.shape[0]) - 1)
    if split != int(cfg.train_split):
        print(f"Adjusted train_split from {cfg.train_split} to {split} to fit dataset size.")

    train_maps = maps[:split]
    val_maps = maps[split:]
    train_motion = motion_dynamic[:split]
    val_motion = motion_dynamic[split:]
    train_pot = position_potential[:split]
    val_pot = position_potential[split:]

    dataset_train = PotentialDatasetD3(
        train_maps,
        train_pot,
        footprint,
        train_motion,
        num_dyn=(None if int(args.num_dyn) <= 0 else int(args.num_dyn)),
        steps_per_dyn=cfg.steps_per_dyn,
        time_offset=cfg.time_offset,
        clamp_max=cfg.potential_clip_max,
    )
    dataset_val = PotentialDatasetD3(
        val_maps,
        val_pot,
        footprint,
        val_motion,
        num_dyn=(None if int(args.num_dyn) <= 0 else int(args.num_dyn)),
        steps_per_dyn=cfg.steps_per_dyn,
        time_offset=cfg.time_offset,
        clamp_max=cfg.potential_clip_max,
    )
    print(
        "dataset_train len:",
        len(dataset_train),
        "| dataset_val len:",
        len(dataset_val),
        "| num_dyn used:",
        dataset_train.num_dyn,
    )

    loader = _make_dataloader(dataset_train, batch_size=cfg.batch_size, shuffle=True, cfg=cfg)
    loader_val = _make_dataloader(dataset_val, batch_size=cfg.val_batch_size, shuffle=False, cfg=cfg)

    gptconf = GPTConfig(
        n_layer=int(args.n_layer),
        n_head=int(args.n_head),
        n_embd=int(args.n_embd),
        block_size=1024,
        vocab_size=1024,
        dropout=float(args.dropout),
        bias=True,
    )
    model = GPT(gptconf).to(device)
    if hasattr(model, "device"):
        model.device = device
    print(describe_model(model))

    if args.resume_from:
        ckpt_resume = Path(args.resume_from)
        payload = torch.load(ckpt_resume, map_location=device)
        if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]
        if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            payload = payload["model"]
        if not isinstance(payload, dict):
            raise TypeError(
                f"Unsupported checkpoint format at {ckpt_resume}. Expected a dict-like state_dict."
            )
        loaded_keys, skipped_shape, missing_keys = _load_partial_state_dict(model, payload)
        print("resumed from:", ckpt_resume)
        print(
            "partial load summary:",
            f"loaded={len(loaded_keys)}",
            f"skipped_shape={len(skipped_shape)}",
            f"missing_in_checkpoint={len(missing_keys)}",
        )
        if skipped_shape:
            print("first shape mismatches:")
            for msg in skipped_shape[:20]:
                print("  -", msg)
            if len(skipped_shape) > 20:
                print(f"  ... and {len(skipped_shape) - 20} more")

    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    use_amp = bool(args.amp and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    use_map_loss = (not bool(args.no_map_loss)) or bool(args.use_map_loss)

    experiment = None
    if not args.no_comet:
        experiment = _try_create_comet_experiment(cfg.comet_project, cfg.comet_workspace)

    global_step = 0
    ckpt_path = trained_models_dir / cfg.checkpoint_name

    def run_validation(step: int, epoch: int) -> None:
        model.eval()
        val_total_loss_sum = 0.0
        val_pot_loss_sum = 0.0
        val_map_loss_sum = 0.0
        count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader_val):
                if batch_idx >= int(cfg.max_val_batches):
                    break

                inp = [t.to(device) for t in batch["input"]]
                out = batch["output"].to(device).unsqueeze(-1)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    _, loss_potential, decoded_map_loss = model.forward_loss(inp, out)
                    total_loss = loss_potential
                    if use_map_loss:
                        total_loss = total_loss + float(args.map_loss_weight) * decoded_map_loss

                val_total_loss_sum += float(total_loss.item())
                val_pot_loss_sum += float(loss_potential.item())
                val_map_loss_sum += float(decoded_map_loss.item())
                count += 1

                if batch_idx % 10 == 0:
                    print(
                        "val",
                        epoch,
                        batch_idx,
                        float(total_loss.item()),
                        float(loss_potential.item()),
                        float(decoded_map_loss.item()),
                    )

        if count > 0 and experiment is not None:
            experiment.log_metric("val_total_loss", val_total_loss_sum / count, step=step)
            experiment.log_metric("val_loss", val_pot_loss_sum / count, step=step)
            experiment.log_metric("val_map_loss", val_map_loss_sum / count, step=step)

    for epoch in range(cfg.epochs):
        model.train(True)
        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)

            inp = [t.to(device) for t in batch["input"]]
            out = batch["output"].to(device).unsqueeze(-1)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, loss_potential, decoded_map_loss = model.forward_loss(inp, out)
                total_loss = loss_potential
                if use_map_loss:
                    total_loss = total_loss + float(args.map_loss_weight) * decoded_map_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if experiment is not None:
                experiment.log_metric("total_loss", float(total_loss.item()), step=global_step)
                experiment.log_metric("loss", float(loss_potential.item()), step=global_step)
                experiment.log_metric("map_loss", float(decoded_map_loss.item()), step=global_step)

            if batch_idx % 10 == 0:
                print(
                    f"Epoch: {epoch} | Batch: {batch_idx} | "
                    f"Total Loss: {total_loss.item():.6f} | "
                    f"Potential Loss: {loss_potential.item():.6f} | "
                    f"Decoded Map Loss: {decoded_map_loss.item():.6f}"
                )

            if cfg.checkpoint_every_steps > 0 and global_step > 0:
                if global_step % cfg.checkpoint_every_steps == 0:
                    torch.save(model.state_dict(), ckpt_path)
                    print("checkpoint:", ckpt_path, "step:", global_step)

            global_step += 1

            if cfg.val_every_steps > 0 and global_step > 0 and global_step % cfg.val_every_steps == 0:
                run_validation(global_step, epoch)
                model.train(True)

        run_validation(global_step, epoch)

    torch.save(model.state_dict(), ckpt_path)
    print("saved:", ckpt_path)


if __name__ == "__main__":
    main()
