# Dynamic Neural Potential Field: Online Trajectory Optimization in Presence of Moving Obstacles

Official Implementation of [Dyn-NPField](https://arxiv.org/abs/2410.06819)

![Dyn-NPField](https://github.com/user-attachments/assets/53c3f20a-3e9c-4c38-a5bf-453c30371902)

[Paper](https://arxiv.org/abs/2410.06819) | [Video]( https://youtu.be/8NqUtvvCOi4?si=WsPIDKKH9Dgz2Uy9) | [Models](https://disk.yandex.ru/d/arqq97Yun_3f0w) | [Dataset](https://disk.yandex.ru/d/fbWIw6NJgjBBSw)

## OVERVIEW
We address local trajectory planning for a mobile robot in the presence of static and dynamic obstacles. The trajectory is computed as a numerical solution to a Model Predictive Control (MPC) problem, with collision avoidance incorporated by adding obstacle repulsive potential to the MPC cost function. Our approach estimates this repulsive potential using a neural model. We explore three strategies for handling dynamic obstacles: treating them as a sequence of static environments, predicting a full sequence of repulsive potentials at once, and predicting future potentials step by step in an autoregressive mode.

![image](https://github.com/user-attachments/assets/0907bb43-3868-4119-ab8e-52f5edcbd979)

## GETTING STARTED

### **Using Docker**

#### Prerequisites:
1. **Docker:** Ensure Docker is installed on your machine. You can download it from [Docker's official site](https://docs.docker.com/get-docker/).
2. **NVIDIA GPU:** Since this Dockerfile is based on NVIDIA CUDA, having an NVIDIA GPU and ensuring [NVIDIA Docker Toolkit](https://github.com/NVIDIA/nvidia-docker) is installed are necessary for GPU acceleration.

#### Steps:

1. **Clone the Repository:**
   Begin by cloning the repository containing the Dockerfile to your local machine.
   ```bash
   git clone https://github.com/CognitiveAISystems/Dynamic-Neural-Potential-Field
   cd Dynamic-Neural-Potential-Field
   ```

2. **Build the Docker Image:**
   Use the following command to build the Docker image from the Dockerfile in your repository. Replace `<your-image-name>` with a name of your choice for the Docker image.
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
     dyn_npfield \
     # Add any additional parameters you require here, for example:
     # --some-option value --another-option
   ```

   - The `--gpus all` flag enables GPU support in the container.
   - The `-p 80:80` option forwards port 80 from the container to the host, making any web service inside the container accessible via the host's port 80.
   - The `-v "$(pwd)/NPField:/app/NPField"` bind-mounts your local `NPField` folder into the container so changes on the host are reflected immediately.
   - The `-v /data/Docker_data/npfiled_dataset:/app/NPField/dataset` mounts a local directory containing your datasets into the container at the given path.
   - You can optionally add more command-line arguments or environment variables at the end of the `docker run` command to control the application's behavior.

4. **Accessing the Container:**
   You can access the running container via:
   ```bash
    docker exec -it dyn_npfield /bin/bash
   ```
   This will open a bash shell inside the container where you can interact with the software and run commands.

5. **Example Run (Default Parameters 0 0 0):**
   Inside the container, run the following command to generate the default example GIF:
   ```bash
   export NPFIELD_DATASET_DIR=/app/NPField/dataset/dataset1000
   python NPField/script_d3/NPField_model_GPT.py 0 0 0
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

#### Troubleshooting:
- Ensure CUDA-compatible drivers are installed on your host machine.
- Check for any Docker-related errors during image building or container running by examining the Docker logs.

### **Without Docker**

#### Prerequisites:
- **Ubuntu 22.04:** Ensure you are running on an Ubuntu 22.04 environment.
- **NVIDIA GPU:** A compatible NVIDIA GPU is necessary since the software relies on CUDA libraries.
- **Python, Git, and Build Tools:** Install Python, Git, CMake, GCC, and other necessary tools.

#### Steps:

1. **Install CUDA and NVIDIA Drivers:**
   Install CUDA 12.4.1 and corresponding NVIDIA drivers if they are not already installed.
   ```bash
   sudo apt-get install nvidia-cuda-toolkit
   ```

2. **Clone the Repository:**
   Clone the repository that contains your project.
   ```bash
   git clone https://github.com/CognitiveAISystems/Dynamic-Neural-Potential-Field
   cd Dynamic-Neural-Potential-Field
   ```

3. **Set Up Python Environment:**
   It's recommended to use a virtual environment.
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install --upgrade pip
   ```

4. **Third Party Libraries:**
   Compile and install the following libraries and their dependencies (PyTorch, CMake etc):
   - [acados](https://github.com/acados/acados)
   - [l4casadi](https://github.com/Tim-Salzmann/l4casadi)  

6. **Run the Application:**
   Navigate to NPField directory, choose the required version (d1, d2 or d3) and run: 
   ```
   python test_solver.py
   ```

#### Troubleshooting:
- Ensure all dependencies and their versions are correctly installed.
- Verify CUDA and GPU drivers are properly installed and recognized by your system.


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
