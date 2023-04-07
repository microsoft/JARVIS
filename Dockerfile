# Base image
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu16.04

# Install system dependencies and NVIDIA drivers
RUN apt-get update && \
    apt-get install -y curl wget git && \
    rm -rf /var/lib/apt/lists/* && \
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update && \
    apt-get install -y nvidia-driver-470 && \
    rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update && \
    apt-get install -y nvidia-container-toolkit && \
    rm -rf /var/lib/apt/lists/*

# Install Python and other dependencies
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3-pip python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY server/requirements.txt .
RUN pip3 install -r requirements.txt

# Download and install PyTorch
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Copy the entire project into the container
COPY . .

# Expose the default API port
EXPOSE 8004

# Start the API server
CMD ["python3", "models_server.py", "--config", "config.yaml", "--mode", "server"]
