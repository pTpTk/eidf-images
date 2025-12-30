FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    clang-format \
    clangd-19 \
    vim \
    && rm -rf /var/lib/apt/lists/*
    
# Remove default 'ubuntu' user (UID 1000) to prevent devcontainer permission conflicts
# Ref: https://github.com/rapidsai/devcontainers/pull/373
RUN if grep ubuntu:x:1000:1000 /etc/passwd >/dev/null; then userdel -f -r ubuntu; fi

# Create a non-root user
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*


# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Install python
COPY install/install_python.sh /install/install_python.sh
RUN bash /install/install_python.sh /home/$USERNAME/conda py312

RUN echo "source activate py312" >> ~/.bashrc
ENV PATH="/home/$USERNAME/conda/bin:$PATH"
ENV PATH="/home/$USERNAME/conda/envs/py312/bin:$PATH"

# Install torch and other python packages
COPY requirements.txt /install/requirements.txt
COPY install/install_python_packages.sh /install/install_python_packages.sh
RUN bash /install/install_python_packages.sh cu130 && pip3 install pre-commit

# Install mpi4py in the conda environment
RUN conda install -n py312 -y mpi4py mpich

# clangd
ENV PATH="/usr/lib/llvm-19/bin:$PATH"

# Triton
ENV TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

ENV SHELL=/bin/bash
CMD [ "bash" ]
