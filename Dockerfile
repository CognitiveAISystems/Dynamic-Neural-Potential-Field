# Base image with NVIDIA CUDA 12.4.1 on Ubuntu 22.04
ARG IMAGE_NAME=nvidia/cuda
FROM ${IMAGE_NAME}:12.4.1-devel-ubuntu22.04 as base

# Set environment variables for various CUDA libraries
ENV NV_CUDA_LIB_VERSION 12.4.1-1
ENV NV_NVTX_VERSION 12.4.127-1
ENV NV_LIBNPP_VERSION 12.2.5.30-1
ENV NV_LIBNPP_PACKAGE libnpp-12-4=${NV_LIBNPP_VERSION}
ENV NV_LIBCUSPARSE_VERSION 12.3.1.170-1
ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas-12-4
ENV NV_LIBCUBLAS_VERSION 12.4.5.8-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}
ENV NV_LIBNCCL_PACKAGE_NAME libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION 2.21.5-1
ENV NCCL_VERSION 2.21.5-1
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.4

# Update and install necessary Linux packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    make \
    gcc \
    g++ \
    vim \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    wget \
    cuda-libraries-12-4=${NV_CUDA_LIB_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    cuda-nvtx-12-4=${NV_NVTX_VERSION} \
    libcusparse-12-4=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*docke

# Upgrade pip
RUN pip3 install --upgrade pip

# Create a Python virtual environment and activate it
RUN python3 -m venv /env
ENV PATH="/env/bin:$PATH"

RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Copy your files into the container
COPY Planners /app/Planners
COPY NPField /app/NPField

RUN mkdir -p /app/third-party

# Install acados from third-party directory
RUN cd /app/third-party && \
    git clone https://github.com/acados/acados.git && \
    cd acados && \
    git submodule update --recursive --init --depth 1  && \
    mkdir -p build && cd build && \
    cmake .. -DACADOS_WITH_QPOASES=ON && \
    make install -j4

# Install acados python Interface    
RUN pip install -e /app/third-party/acados/interfaces/acados_template

# Set environment variables needed by acados
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app/third-party/acados/lib
ENV ACADOS_SOURCE_DIR="/app/third-party/acados"


# Install L4casadi
RUN cd /app/third-party && \
    git clone https://github.com/Tim-Salzmann/l4casadi.git && \
    cd l4casadi && \
    pip install -r requirements_build.txt && \
    pip install . --no-build-isolation


# Set the working directory to /app
WORKDIR /app

# Expose port 80 for service access
EXPOSE 80

# Default command to run a bash shell
CMD ["/bin/bash"]
