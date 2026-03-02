# Dynamic Neural Potential Field: Online Trajectory Optimization in Presence of Moving Obstacles

Official Implementation of [Dyn-NPField](https://arxiv.org/abs/2410.06819)

<p align="center">
  <img src="https://github.com/user-attachments/assets/906cd8d4-b934-4fe5-8c50-4c6fef28f328" width="45%" alt="Dyn-NPField"/>
  <img src="https://github.com/user-attachments/assets/aa96f2e7-ada7-442d-ba35-c25c33d356a4" width="45%" alt="Dyn-NPField"/>
</p>


[Paper](https://arxiv.org/abs/2410.06819) | [Video]( https://youtu.be/8NqUtvvCOi4?si=WsPIDKKH9Dgz2Uy9) | [Models](https://disk.yandex.ru/d/arqq97Yun_3f0w) | [Dataset](https://disk.yandex.ru/d/fbWIw6NJgjBBSw)

## OVERVIEW
We address local trajectory planning for a mobile robot in the presence of static and dynamic obstacles. The trajectory is computed as a numerical solution to a Model Predictive Control (MPC) problem, with collision avoidance incorporated by adding obstacle repulsive potential to the MPC cost function. Our approach estimates this repulsive potential using a neural model. We explore three strategies for handling dynamic obstacles: treating them as a sequence of static environments, predicting a full sequence of repulsive potentials at once, and predicting future potentials step by step in an autoregressive mode.

<img width="622" height="293" alt="npfield-gpt" src="https://github.com/user-attachments/assets/c691632e-fd19-49b2-a48b-8df7f621fb4f" />


## GETTING STARTED

### **Using Docker**

#### Steps:

1. **Clone the Repository:**
   Begin by cloning the repository containing the Dockerfile to your local machine.
```bash
   git clone https://github.com/CognitiveAISystems/Dynamic-Neural-Potential-Field
   cd Dynamic-Neural-Potential-Field
   ```

2. **Build the Docker Image:**
   Use the following command to build the Docker image from the Dockerfile in your repository.
```bash
    docker build -t dyn_npfield .
   ```

3. **Run the Docker Container:**
   After building the image, run a container from it. The following commands will first remove any existing container named `dyn_npfield`, then start a new one with the specified options. The `NPField` directory is mounted from your host into the container (instead of being copied):

```bash
   docker rm -f dyn_npfield

   docker run -it --gpus all --name dyn_npfield -p 80:80 \
     -v "$(pwd)/NPField:/app/NPField" \
     -v /data/Docker_data/npfiled_dataset:/app/NPField/dataset \
     dyn_npfield
   ```
   - The `-v "$(pwd)/NPField:/app/NPField"` bind-mounts your local `NPField` folder into the container so changes on the host are reflected immediately.
   - The `-v /data/Docker_data/npfiled_dataset:/app/NPField/dataset` mounts a local directory containing your datasets into the container at the given path.

4. **Accessing the Container:**
   You can access the running container via:
```bash
    docker exec -it dyn_npfield /bin/bash
   ```
   This will open a bash shell inside the container where you can interact with the software and run commands.

5. **Example Run:**
   Inside the container, run the following command to generate the default example GIF:
```bash
   export NPFIELD_DATASET_DIR=/app/NPField/dataset/dataset1000
   python NPField/script_d3/NPField_model_GPT.py --finetune-checkpoint /app/NPField/dataset/trained-models/NPField_D3_finetune.pth
   ```
   This script is intended for testing the neural net and visualizing the resulting neural potential field in the presence of a dynamic obstacle.
   Parameters for `NPField/script_d3/NPField_model_GPT.py`:
   - `episode` (int): index into the dataset episode list.
   - `id_dyn` (int): dynamic obstacle id within the episode.
   - `angle` (float, degrees): query heading angle for inference (passed internally in radians).
   Optional flags:
   - `--device` (`cpu` or `cuda`): inference device.
   - `--chunk-size` (int): batch size for grid inference.
   Output GIFs are written to `NPField/output` with names like `NPField_D3_ep{episode}_dyn{id_dyn}_angle_{angle}deg.gif`.

6. **Train D3 (new model/checkpoint):**
```bash
   export NPFIELD_DATASET_DIR=/app/NPField/dataset/dataset1000 && python NPField/script_d3/train_model.py \
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

   # D1 finetune:
   ```bash
   export NPFIELD_DATASET_DIR=/app/NPField/dataset/dataset1000 && python NPField/script_d1/train_model.py \
        --epochs 10 \
        --lr 5e-5 \
        --batch-size 8 \
        --val-batch-size 4 \
        --dropout 0.1 \
        --amp \
        --no-map-loss \
        --checkpoint-name NPField_D1_finetune.pth
   ```


7. **Test D3 trajectory with the finetuned checkpoint:**
```bash
   python NPField/script_d3/test_solver_GPT.py \
     --map-id 993 \
     --episodes 10 \
     --finetune-checkpoint /app/NPField/dataset/trained-models/NPField_D3_finetune.pth \
     --save-potential-gif

   python NPField/script_d2/test_solver.py \
     --map-id 993 \
     --episodes 10 \
     --save-potential-gif
   ```

8. **Generate 100 benchmark configs and Evaluate all scenarios**
```bash
   # Step 1: 
   cd NPField/config
   python generate_MPC_config.py --save-json --num-scenarios 100

   # Step 2: 
   cd NPField/script_d3
   python test_solver_GPT.py \
      --benchmark-json ../output/benchmark_scenarios.json \
      --finetune-checkpoint /app/NPField/dataset/trained-models/NPField_D3_finetune.pth \
      --save-potential-gif \
      --allow-backward

   cd NPField/script_d2
   python test_solver.py \
      --benchmark-json ../output/benchmark_scenarios.json \
      --save-potential-gif \
      --allow-backward

   # D1 (finetuned weights):
   cd NPField/script_d1
   python test_solver.py \
      --benchmark-json ../output/benchmark_scenarios.json \
      --finetune-checkpoint /app/NPField/dataset/trained-models/NPField_D1_finetune.pth \
      --save-potential-gif \
      --allow-backward
   ```

#### Troubleshooting:
- Ensure CUDA-compatible drivers are installed on your host machine.
- Check for any Docker-related errors during image building or container running by examining the Docker logs.

## CITATION
If you use this framework please cite the following two papers:

### Dyn-NPField: 
```
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

### NPField:
```
@misc{alhaddad2023neuralpotentialfieldobstacleaware,
      title={Neural Potential Field for Obstacle-Aware Local Motion Planning}, 
      author={Muhammad Alhaddad and Konstantin Mironov and Aleksey Staroverov and Aleksandr Panov},
      year={2023},
      eprint={2310.16362},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2310.16362}, 
}
```





