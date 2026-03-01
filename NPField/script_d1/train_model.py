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

from model_nn import Autoencoder_path


def _try_create_comet_experiment(
    project_name: str,
    workspace: str,
) -> Optional[object]:
    from comet_ml import start  # type: ignore

    api_key = os.getenv("COMET_API_KEY")
    return start(api_key=api_key, project_name=project_name, workspace=workspace)


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


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


def _require_key(blob: dict, key: str, *, source_name: str) -> np.ndarray:
    if key not in blob:
        raise KeyError(f"Missing key '{key}' in {source_name}")
    return np.asarray(blob[key])


def _load_partial_state_dict(
    model: nn.Module, payload: dict,
) -> tuple[list[str], list[str], list[str]]:
    """Load only checkpoint tensors that match existing model keys and shapes."""
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


def _count_params(module: nn.Module, *, trainable_only: bool) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def describe_model(model: nn.Module) -> str:
    total = _count_params(model, trainable_only=False)
    trainable = _count_params(model, trainable_only=True)
    return (
        f"Model class: {model.__class__.__name__}\n"
        f"Total params: {total:,}\n"
        f"Trainable params: {trainable:,}"
    )


def forward_train(model: Autoencoder_path, batch):
    """Training forward pass that returns (potential_pred, decoded_map).

    Mirrors model.step_ctrl but also returns the decoded map for the
    optional map-reconstruction loss term.
    """
    mapp, x_crd, y_crd, theta = batch
    map_encode = model.encoder(mapp[:, :1, :, :])

    map_encode_robot = (
        model.encoder_robot(mapp[:, -1:, :, :]).flatten().view(mapp.shape[0], -1)
    )
    x_cr_encode = model.x_cord(x_crd)
    y_cr_encode = model.y_cord(y_crd)
    tsin_encode = model.theta_sin(torch.sin(theta))
    tcos_encode = model.theta_cos(torch.cos(theta))

    encoded_input = map_encode
    encoded_input = model.encoder_after(encoded_input)
    encoded_input = model.decoder_after(encoded_input)

    decoded_map = model.decoder_MAP(encoded_input)

    encoded_input = model.pos(encoded_input)
    encoded_input = model.transformer(encoded_input)
    encoded_input = model.decoder_pos(encoded_input)
    encoded_input = model.decoder(encoded_input).view(encoded_input.shape[0], -1)

    encoded_input = torch.cat(
        (
            encoded_input,
            map_encode_robot,
            x_cr_encode,
            y_cr_encode,
            tsin_encode,
            tcos_encode,
        ),
        1,
    )

    pred = model.linear_after_mean(encoded_input)
    return pred, decoded_map


class PotentialDatasetD1(Dataset):
    """D1 training dataset.

    Each sample pairs a single sub-map frame (with baked-in obstacle)
    and robot footprint with a scalar potential value.

    Input to forward_train:
        mapp   : (2, 50, 50)  channel-0 = sub-map, channel-1 = footprint
        x_crd  : (1,)
        y_crd  : (1,)
        theta  : (1,)
    Target: scalar potential (clamped and normalised to [0, 1]).
    """

    def __init__(
        self,
        maps: np.ndarray,
        potential: np.ndarray,
        footprint: np.ndarray,
        *,
        time_offset: int = 1,
        num_steps: int = 10,
        clamp_max: float = 30.0,
    ) -> None:
        self.maps = maps
        self.potential = potential
        self.footprint = footprint
        self.clamp_max = float(clamp_max)

        max_t = min(time_offset + num_steps, maps.shape[1], potential.shape[1])
        self.time_indices = list(range(time_offset, max_t))
        if not self.time_indices:
            raise ValueError(
                f"No valid time indices: maps has {maps.shape[1]} frames, "
                f"potential has {potential.shape[1]} steps, time_offset={time_offset}"
            )

        self.n_maps = int(maps.shape[0])
        self.n_times = len(self.time_indices)
        self.n_positions = int(potential.shape[2])
        self.n_theta = int(potential.shape[3])

    def __len__(self) -> int:
        return self.n_maps * self.n_times * self.n_positions * self.n_theta

    def __getitem__(self, idx: int):
        per_map = self.n_times * self.n_positions * self.n_theta
        map_id = idx // per_map
        rem = idx % per_map

        per_time = self.n_positions * self.n_theta
        time_local = rem // per_time
        rem2 = rem % per_time

        pos_id = rem2 // self.n_theta
        theta_id = rem2 % self.n_theta

        t = self.time_indices[time_local]

        map_inp = torch.tensor(
            np.stack((self.maps[map_id, t], self.footprint)),
            dtype=torch.float32,
        ) / 100.0

        x = torch.tensor(self.potential[map_id, t, pos_id, theta_id, 0], dtype=torch.float32)
        y = torch.tensor(self.potential[map_id, t, pos_id, theta_id, 1], dtype=torch.float32)
        theta = torch.tensor(self.potential[map_id, t, pos_id, theta_id, 2], dtype=torch.float32)

        target = (
            torch.clamp(
                torch.tensor(self.potential[map_id, t, pos_id, theta_id, 3], dtype=torch.float32),
                min=0.0,
                max=self.clamp_max,
            )
            / self.clamp_max
        )

        return {
            "input": (map_inp, x.unsqueeze(0), y.unsqueeze(0), theta.unsqueeze(0)),
            "output": target,
        }


@dataclass(frozen=True)
class TrainConfig:
    train_split: int = 990
    epochs: int = 4
    lr: float = 1e-4
    grad_clip: float = 10.0
    batch_size: int = 256
    val_batch_size: int = 16
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    time_offset: int = 1
    num_steps: int = 10
    potential_clip_max: float = 30.0
    checkpoint_every_steps: int = 1000
    val_every_steps: int = 1000
    max_val_batches: int = 100
    checkpoint_name: str = "NPField_D1_finetune.pth"
    comet_project: str = "npfield-d1-training"
    comet_workspace: str = "aleksei2"


def _make_dataloader(
    ds: Dataset, *, batch_size: int, shuffle: bool, cfg: TrainConfig,
) -> DataLoader:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune NPField D1 (Static-MLP) model")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--val-batch-size", type=int, default=TrainConfig.val_batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-split", type=int, default=TrainConfig.train_split)
    parser.add_argument("--time-offset", type=int, default=TrainConfig.time_offset)
    parser.add_argument("--num-steps", type=int, default=TrainConfig.num_steps)
    parser.add_argument("--potential-clip-max", type=float, default=TrainConfig.potential_clip_max)
    parser.add_argument("--checkpoint-every-steps", type=int, default=TrainConfig.checkpoint_every_steps)
    parser.add_argument("--val-every-steps", type=int, default=TrainConfig.val_every_steps)
    parser.add_argument("--max-val-batches", type=int, default=TrainConfig.max_val_batches)
    parser.add_argument("--checkpoint-name", type=str, default=TrainConfig.checkpoint_name)
    parser.add_argument(
        "--pretrained", type=str, default="",
        help="Path to pretrained NPField_D1.pth (auto-resolved if empty)",
    )
    parser.add_argument(
        "--resume-from", type=str, default="",
        help="Resume training from a checkpoint (loads weights + ignores --pretrained)",
    )
    parser.add_argument("--no-comet", action="store_true", help="Disable Comet logging")
    parser.add_argument("--no-map-loss", action="store_true", help="Disable decoded map loss term")
    parser.add_argument("--use-map-loss", action="store_true", help="(deprecated) Enable map loss")
    parser.add_argument("--map-loss-weight", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument(
        "--dropout", type=float, default=0.15,
        help="Dropout rate for encoder/attention (default: 0.15, matching pretrained model)",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        train_split=int(args.train_split),
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        val_batch_size=int(args.val_batch_size),
        num_workers=int(args.num_workers),
        time_offset=int(args.time_offset),
        num_steps=int(args.num_steps),
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

    potential_path = _first_existing_path([
        dataset_root / "potentials" / "dataset_1000_potential_husky.pkl",
        dataset_root / "dataset_1000_potential_husky.pkl",
    ])
    sub_maps_blob = _load_pickle(dataset_root / "dataset_1000_maps_0_100_all.pkl")
    footprint_blob = _load_pickle(dataset_root / "data_footprint.pkl")
    potential_blob = _load_pickle(potential_path)

    maps = _require_key(sub_maps_blob, "submaps", source_name="dataset_1000_maps_0_100_all.pkl")
    footprint = _require_key(footprint_blob, "footprint_husky", source_name="data_footprint.pkl")
    position_potential = _require_key(
        potential_blob, "position_potential", source_name=potential_path.name,
    )

    print(f"maps shape: {maps.shape}")
    print(f"potential shape: {position_potential.shape}")
    print(f"footprint shape: {footprint.shape}")

    if maps.shape[0] < 2:
        raise ValueError(f"Need at least 2 maps for train/val split, got {maps.shape[0]}")
    split = min(max(1, int(cfg.train_split)), int(maps.shape[0]) - 1)
    if split != int(cfg.train_split):
        print(f"Adjusted train_split from {cfg.train_split} to {split}")

    dataset_train = PotentialDatasetD1(
        maps[:split], position_potential[:split], footprint,
        time_offset=cfg.time_offset,
        num_steps=cfg.num_steps,
        clamp_max=cfg.potential_clip_max,
    )
    dataset_val = PotentialDatasetD1(
        maps[split:], position_potential[split:], footprint,
        time_offset=cfg.time_offset,
        num_steps=cfg.num_steps,
        clamp_max=cfg.potential_clip_max,
    )
    print(f"dataset_train len: {len(dataset_train)} | dataset_val len: {len(dataset_val)}")

    loader = _make_dataloader(dataset_train, batch_size=cfg.batch_size, shuffle=True, cfg=cfg)
    loader_val = _make_dataloader(dataset_val, batch_size=cfg.val_batch_size, shuffle=False, cfg=cfg)

    model = Autoencoder_path(
        mode="k",
        cnn_dropout=float(args.dropout),
        attn_dropout=float(args.dropout),
    ).to(device)
    model.device = device

    if args.resume_from:
        ckpt_resume = Path(args.resume_from)
        payload = torch.load(ckpt_resume, map_location=device)
        if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]
        if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
            payload = payload["model"]
        loaded_keys, skipped_shape, missing_keys = _load_partial_state_dict(model, payload)
        print(f"resumed from: {ckpt_resume}")
        print(
            f"partial load summary: loaded={len(loaded_keys)} "
            f"skipped_shape={len(skipped_shape)} "
            f"missing_in_checkpoint={len(missing_keys)}"
        )
        if skipped_shape:
            for msg in skipped_shape[:20]:
                print(f"  - {msg}")
    else:
        if args.pretrained:
            pretrained_path = Path(args.pretrained)
        else:
            pretrained_path = _first_existing_path([
                trained_models_dir / "NPField_D1.pth",
                dataset_root / ".." / "trained-models" / "NPField_D1.pth",
            ])
        if pretrained_path.exists():
            state = torch.load(pretrained_path, map_location="cpu")
            model.load_state_dict(state)
            print(f"Loaded pretrained weights from {pretrained_path}")
        else:
            print(f"WARNING: pretrained weights not found at {pretrained_path}, training from scratch")

    print(describe_model(model))

    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    criterion = nn.MSELoss()
    use_amp = bool(args.amp and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    use_map_loss = (not bool(args.no_map_loss)) or bool(args.use_map_loss)
    map_loss_weight = float(args.map_loss_weight)
    print(f"map_loss: {'enabled (weight={map_loss_weight})' if use_map_loss else 'disabled'}")

    experiment = None
    if not args.no_comet:
        try:
            experiment = _try_create_comet_experiment(cfg.comet_project, cfg.comet_workspace)
        except Exception:
            experiment = None

    global_step = 0
    ckpt_path = trained_models_dir / cfg.checkpoint_name

    def run_validation(step: int, epoch: int) -> None:
        model.eval()
        val_total_sum = 0.0
        val_pot_sum = 0.0
        val_map_sum = 0.0
        count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader_val):
                if batch_idx >= int(cfg.max_val_batches):
                    break
                inp = [t.to(device) for t in batch["input"]]
                target = batch["output"].to(device)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    pred, decoded_map = forward_train(model, inp)
                    loss_potential = criterion(pred.squeeze(-1), target)
                    loss_map = criterion(decoded_map, inp[0]) if use_map_loss else torch.zeros(1, device=device)
                    total_loss = loss_potential
                    if use_map_loss:
                        total_loss = total_loss + map_loss_weight * loss_map

                val_total_sum += float(total_loss.item())
                val_pot_sum += float(loss_potential.item())
                val_map_sum += float(loss_map.item())
                count += 1

                if batch_idx % 10 == 0:
                    print(
                        f"  val epoch={epoch} batch={batch_idx} "
                        f"total={total_loss.item():.6f} pot={loss_potential.item():.6f} "
                        f"map={loss_map.item():.6f}"
                    )

        if count > 0:
            print(
                f"  val avg: total={val_total_sum / count:.6f} "
                f"pot={val_pot_sum / count:.6f} map={val_map_sum / count:.6f}"
            )
            if experiment is not None:
                experiment.log_metric("val_total_loss", val_total_sum / count, step=step)
                experiment.log_metric("val_loss", val_pot_sum / count, step=step)
                experiment.log_metric("val_map_loss", val_map_sum / count, step=step)

    for epoch in range(cfg.epochs):
        model.train(True)
        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)

            inp = [t.to(device) for t in batch["input"]]
            target = batch["output"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                pred, decoded_map = forward_train(model, inp)
                loss_potential = criterion(pred.squeeze(-1), target)
                loss_map = criterion(decoded_map, inp[0]) if use_map_loss else torch.zeros(1, device=device)
                total_loss = loss_potential
                if use_map_loss:
                    total_loss = total_loss + map_loss_weight * loss_map

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if experiment is not None:
                experiment.log_metric("total_loss", float(total_loss.item()), step=global_step)
                experiment.log_metric("loss", float(loss_potential.item()), step=global_step)
                experiment.log_metric("map_loss", float(loss_map.item()), step=global_step)

            if batch_idx % 10 == 0:
                print(
                    f"Epoch: {epoch} | Batch: {batch_idx} | "
                    f"Total Loss: {total_loss.item():.6f} | "
                    f"Potential Loss: {loss_potential.item():.6f} | "
                    f"Decoded Map Loss: {loss_map.item():.6f}"
                )

            if cfg.checkpoint_every_steps > 0 and global_step > 0:
                if global_step % cfg.checkpoint_every_steps == 0:
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"checkpoint: {ckpt_path} step: {global_step}")

            global_step += 1

            if cfg.val_every_steps > 0 and global_step > 0 and global_step % cfg.val_every_steps == 0:
                run_validation(global_step, epoch)
                model.train(True)

        run_validation(global_step, epoch)

    torch.save(model.state_dict(), ckpt_path)
    print(f"saved: {ckpt_path}")


if __name__ == "__main__":
    main()
