# Use NVIDIA PyTorch base image with CUDA support
FROM mcr.microsoft.com/devcontainers/python:dev-3.12-bookworm

# Set working directory
WORKDIR /workspace

RUN pip install pipx
RUN pipx ensurepath
RUN pipx install uv

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    tmux \
    htop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# Set environment to use the virtual environment
ENV PATH="/workspace/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/workspace/.venv"

# Expose typical ports for Jupyter, TensorBoard, etc.
EXPOSE 8888 6006

# Set default command
CMD ["/bin/zsh"]
