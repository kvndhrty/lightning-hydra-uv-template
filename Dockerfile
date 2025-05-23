# Use NVIDIA PyTorch base image with CUDA support
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    tmux \
    htop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy project files
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY scripts ./scripts
COPY tests ./tests

# Create virtual environment and install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e ".[all]"

# Set environment to use the virtual environment
ENV PATH="/workspace/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/workspace/.venv"

# Expose typical ports for Jupyter, TensorBoard, etc.
EXPOSE 8888 6006

# Set default command
CMD ["/bin/bash"]