# Dynamic Neural Potential Field: Online Trajectory Optimization in the Presence of Moving Obstacles

Official implementation of **Dyn-NPField** (ICRA 2026).

<p align="center">
  <img src="https://github.com/user-attachments/assets/906cd8d4-b934-4fe5-8c50-4c6fef28f328" width="45%" alt="Dyn-NPField"/>
  <img src="https://github.com/user-attachments/assets/aa96f2e7-ada7-442d-ba35-c25c33d356a4" width="45%" alt="Dyn-NPField"/>
</p>

[Paper](https://arxiv.org/abs/2410.06819) | [Video](https://youtu.be/8NqUtvvCOi4?si=WsPIDKKH9Dgz2Uy9) | [Models](https://disk.yandex.ru/d/arqq97Yun_3f0w) | [Dataset](https://disk.yandex.ru/d/fbWIw6NJgjBBSw)

## Overview

We present **Dynamic Neural Potential Field (NPField-GPT)**, a learning-enhanced Model Predictive Control (MPC) framework that couples classical optimization with a Transformer-based predictor of footprint-aware repulsive potentials. Given an occupancy sub-map, robot footprint, and dynamic-obstacle cues, the neural model forecasts a horizon of differentiable potentials that are injected into a sequential quadratic MPC program via **L4CasADi**, yielding real-time, constraint-aware trajectory optimization.

The framework comprises three neural architectures that balance accuracy and latency:

| Variant | Description | Key property |
|---|---|---|
| **NPField-StaticMLP** (`script_d1`) | Treats dynamic scenes as a sequence of static frames | Lowest latency |
| **NPField-DynamicMLP** (`script_d2`) | Predicts future potentials in parallel via separate MLP heads conditioned on dynamic obstacle state | Mid-range latency |
| **NPField-GPT** (`script_d3`) | Non-autoregressive Transformer predicts the full potential horizon in one forward pass | Best safety & path quality |

<img width="622" height="293" alt="npfield-gpt" src="https://github.com/user-attachments/assets/cf9e7698-d297-40b5-b273-6e3f10be902e" />

## Project Structure

```
Dynamic-Neural-Potential-Field/
├── NPField/
│   ├── config/                     # Benchmark scenario generator
│   │   └── generate_MPC_config.py
│   ├── dataset/                    # Dataset & trained model weights
│   ├── output/                     # GIFs, metrics, benchmark results
│   ├── script_d1/                  # NPField-StaticMLP
│   │   ├── train_model.py
│   │   ├── test_solver.py
│   │   └── mpc_params.py
│   ├── script_d2/                  # NPField-DynamicMLP
│   │   ├── train_model.py          (uses pretrained D1 encoder)
│   │   ├── test_solver.py
│   │   └── mpc_params.py
│   ├── script_d3/                  # NPField-GPT (single dynamic obstacle)
│   │   ├── train_model.py
│   │   ├── test_solver_GPT.py
│   │   ├── NPField_model_GPT.py   # NN visualization script
│   │   └── mpc_params.py
│   ├── script_d3_multi/            # NPField-GPT (multiple dynamic obstacles)
│   │   ├── train_model.py
│   │   ├── test_solver_multi.py    # Multi-obstacle evaluation
│   │   ├── test_solver_GPT.py
│   │   ├── NPField_model_GPT.py
│   │   ├── scenario_multi_obstacle.json
│   │   └── mpc_params.py
│   └── TransPath/                  # Global planner utilities
├── Planners/
│   └── Ciao-star/                  # CIAO* baseline
└── Dockerfile
```

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA-compatible drivers
- Docker with NVIDIA Container Toolkit (`nvidia-docker`)

### 1. Clone and Build

```bash
git clone https://github.com/CognitiveAISystems/Dynamic-Neural-Potential-Field
cd Dynamic-Neural-Potential-Field
docker build -t dyn_npfield .
```

### 2. Download Data

Download the [Dataset](https://disk.yandex.ru/d/fbWIw6NJgjBBSw) and [Trained Models](https://disk.yandex.ru/d/arqq97Yun_3f0w) and place them so the container can access them (see the `-v` mounts below).

### 3. Run the Container

```bash
docker rm -f dyn_npfield

docker run -it --gpus all --name dyn_npfield -p 80:80 \
  -v "$(pwd)/NPField:/app/NPField" \
  -v /path/to/your/dataset:/app/NPField/dataset \
  dyn_npfield
```

- `-v "$(pwd)/NPField:/app/NPField"` — bind-mounts the local `NPField` folder so host edits are reflected immediately.
- `-v /path/to/your/dataset:/app/NPField/dataset` — mounts the dataset directory (should contain `dataset1000/` and `trained-models/`).

To attach a shell to the running container:

```bash
docker exec -it dyn_npfield /bin/bash
```

All commands below assume you are inside the container with the dataset mounted.

### 4. Set Environment

```bash
export NPFIELD_DATASET_DIR=/app/NPField/dataset/dataset1000
```

---

## Training

### NPField-GPT (D3)

```bash
cd /app
export NPFIELD_DATASET_DIR=/app/NPField/dataset/dataset1000

python NPField/script_d3/train_model.py \
  --epochs 10 \
  --lr 5e-5 \
  --batch-size 64 \
  --val-batch-size 16 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 256 \
  --dropout 0.1 \
  --amp \
  --no-map-loss \
  --checkpoint-name NPField_D3_finetune.pth
```



### NPField-StaticMLP (D1)

```bash
cd /app
export NPFIELD_DATASET_DIR=/app/NPField/dataset/dataset1000

python NPField/script_d1/train_model.py \
  --epochs 10 \
  --lr 5e-5 \
  --batch-size 8 \
  --val-batch-size 4 \
  --dropout 0.1 \
  --amp \
  --no-map-loss \
  --checkpoint-name NPField_D1_finetune.pth
```

---

## Evaluation

### Single Dynamic Obstacle Benchmarks

#### Step 1 — Generate benchmark scenarios

```bash
cd NPField/config
python generate_MPC_config.py --save-json --num-scenarios 100
```

This writes `NPField/output/benchmark_scenarios.json` with 100 randomized start/goal/obstacle configurations drawn from the BenchMR-style map pool.

#### Step 2 — Run each planner

**NPField-GPT (D3):**

```bash
cd NPField/script_d3
python test_solver_GPT.py \
  --benchmark-json ../output/benchmark_scenarios.json \
  --finetune-checkpoint /app/NPField/dataset/trained-models/NPField_D3_finetune.pth \
  --save-potential-gif \
  --allow-backward
```

**NPField-DynamicMLP (D2):**

```bash
cd NPField/script_d2
python test_solver.py \
  --benchmark-json ../output/benchmark_scenarios.json \
  --save-potential-gif \
  --allow-backward
```

**NPField-StaticMLP (D1):**

```bash
cd NPField/script_d1
python test_solver.py \
  --benchmark-json ../output/benchmark_scenarios.json \
  --finetune-checkpoint /app/NPField/dataset/trained-models/NPField_D1_finetune.pth \
  --save-potential-gif \
  --allow-backward
```

Metrics (planning time, path length, smoothness, AOL, safety distance) and trajectory GIFs are saved to `NPField/output/`.

#### Quick single-episode test

```bash
cd NPField/script_d3
python test_solver_GPT.py \
  --map-id 993 \
  --episodes 10 \
  --finetune-checkpoint /app/NPField/dataset/trained-models/NPField_D3_finetune.pth \
  --save-potential-gif
```

### Multi Dynamic Obstacle Experiments

The `script_d3_multi` module evaluates NPField-GPT on scenes with **2–5 simultaneously moving obstacles** on an empty map. Each obstacle is independently encoded; the resulting potentials are summed (additive superposition) and injected into MPC.

Scenarios are defined in `NPField/script_d3_multi/scenario_multi_obstacle.json` (10 hand-crafted scenarios including converging obstacles, gate crossings, slaloms, diamond formations, and dense gauntlets).

#### Run all multi-obstacle scenarios

```bash
cd NPField/script_d3_multi
python test_solver_multi.py --test-episode --save-potential-gif
```

#### Run a single scenario by ID

```bash
cd NPField/script_d3_multi
python test_solver_multi.py --test-episode --scenario-id 4 --save-potential-gif
```

#### Run from a custom scenario JSON

```bash
cd NPField/script_d3_multi
python test_solver_multi.py --scenario-json my_scenarios.json --save-potential-gif
```

#### Regenerate the default scenario file

```bash
cd NPField/script_d3_multi
python test_solver_multi.py --dump-scenario
```

Output GIFs and per-scenario metrics are saved to `NPField/output/benchmark_D3_multi/`. A summary `all_metrics.json` aggregates results across all scenarios.

**CLI flags for `test_solver_multi.py`:**

| Flag | Description |
|---|---|
| `--test-episode` | Run all scenarios from `scenario_multi_obstacle.json` |
| `--scenario-json PATH` | Path to a custom scenario JSON |
| `--scenario-id N` | Run only scenario N (0-based) |
| `--save-potential-gif` | Overlay the summed potential field in the GIF |
| `--finetune-checkpoint PATH` | Path to model checkpoint |
| `--allow-backward` | Allow backward motion |
| `--no-gif` | Skip GIF generation (metrics only) |

---

## Visualize Neural Potential Field

To visualize the predicted potential field for a given episode and dynamic obstacle:

```bash
cd /app
export NPFIELD_DATASET_DIR=/app/NPField/dataset/dataset1000

python NPField/script_d3/NPField_model_GPT.py \
  --finetune-checkpoint /app/NPField/dataset/trained-models/NPField_D3_finetune.pth
```

**Parameters:**
- `episode` (int) — dataset episode index
- `id_dyn` (int) — dynamic obstacle ID within the episode
- `angle` (float, degrees) — query heading angle
- `--device` (`cpu` | `cuda`) — inference device
- `--chunk-size` (int) — batch size for grid inference

Output GIFs are written to `NPField/output/` as `NPField_D3_ep{episode}_dyn{id_dyn}_angle_{angle}deg.gif`.

---

## Citation

If you use this work, please cite:

### Dyn-NPField
```bibtex
@misc{staroverov2024dynamicneuralpotentialfield,
      title={Dynamic Neural Potential Field: Online Trajectory Optimization in Presence of Moving Obstacles}, 
      author={Aleksey Staroverov and Muhammad Alhaddad and Aditya Narendra and Konstantin Mironov and Aleksandr Panov},
      year={2024},
      eprint={2410.06819},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.06819}, 
}
```

### NPField
```bibtex
@inproceedings{alhaddad2024neural,
  title={Neural potential field for obstacle-aware local motion planning},
  author={Alhaddad, Muhammad and Mironov, Konstantin and Staroverov, Aleksey and Panov, Aleksandr},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={9313--9320},
  year={2024},
  organization={IEEE}
}
```

