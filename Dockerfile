# Base image
FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl wget git python3.8 python3-pip python3-dev build-essential && \
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
CMD ["python3", "models_server.py", "--config", "config.yaml"]
