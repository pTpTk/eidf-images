FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    clang-format \
    clangd-19 \
    vim \
    python3 \
    python3-pip \
    mpich \
    python3-mpi4py \
    python3.12-venv

# add NV cuda repo
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
RUN mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/

# NV driver and toolkit
RUN apt-get update && apt-get install -y \
    cuda-toolkit-13-0 \
    nvidia-driver-580 \
    && rm -rf /var/lib/apt/lists/*

# Remove default 'ubuntu' user (UID 1000) to prevent devcontainer permission conflicts
# Ref: https://github.com/rapidsai/devcontainers/pull/373
RUN if grep ubuntu:x:1000:1000 /etc/passwd >/dev/null; then userdel -f -r ubuntu; fi

# Install torch and other python packages
COPY ../requirements.txt /install/requirements.txt
COPY ../install/install_python_packages.sh /install/install_python_packages.sh
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN bash /install/install_python_packages.sh cu130 && pip3 install pre-commit
RUN git clone https://github.com/pTpTk/flashinfer.git && cd flashinfer && python -m pip install -v .

# clangd
ENV PATH="/usr/lib/llvm-19/bin:$PATH"

# Triton
ENV TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

ENV SHELL=/bin/bash
