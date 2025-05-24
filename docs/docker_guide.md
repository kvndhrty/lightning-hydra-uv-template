# Docker & Dev Container Guide

This project includes comprehensive Docker support for both local development and VS Code Dev Containers.

## Prerequisites

- Docker Desktop installed and running
- NVIDIA Container Toolkit (for GPU support)
- VS Code with Dev Containers extension (for .devcontainer support)

## Quick Start

### Using Docker

```bash
# Build the GPU-enabled image
make docker-build-gpu

# Run container with GPU support
make docker-run

# Or run CPU-only development container
make docker-run-cpu
```

### Using Docker Compose

```bash
# Start all services (main dev container, TensorBoard, Jupyter)
make docker-compose-up

# View logs
make docker-compose-logs

# Stop all services
make docker-compose-down
```

### Using VS Code Dev Containers

1. Open the project in VS Code
2. Press `F1` and select "Dev Containers: Reopen in Container"
3. VS Code will build and start the container with all extensions pre-installed

## Docker Images

### Production Image (GPU-enabled)
- **Base**: `nvcr.io/nvidia/pytorch:24.01-py3`
- **File**: `Dockerfile`
- **Features**: Full CUDA support, all dependencies, optimized for training

### Development Image (CPU-only)
- **Base**: `python:3.11-slim`
- **File**: `Dockerfile.dev`
- **Features**: Lightweight, fast builds, suitable for code development

## Container Features

### GPU Support
Containers automatically detect and use available GPUs:
```bash
# Check GPU availability inside container
nvidia-smi
```

### Pre-installed Tools
- UV package manager
- Git, vim, tmux, htop
- Python development tools
- All project dependencies

### Volume Mounts
- Project directory → `/workspace`
- Cache directory → `/home/user/.cache`
- SSH keys → `/home/user/.ssh` (read-only)

### Exposed Ports
- `8888`: Jupyter Lab
- `6006`: TensorBoard
- `4040`: Spark UI (if needed)

## Development Workflow

### Training in Container
```bash
# Inside container
uv run python src/train.py experiment=example

# Or use make commands
make train
```

### Using Jupyter
```bash
# Start Jupyter service
docker-compose up jupyter

# Access at http://localhost:8889
```

### Using TensorBoard
```bash
# Start TensorBoard service
docker-compose up tensorboard

# Access at http://localhost:6007
```

## VS Code Dev Container

### Features
- Automatic Python environment setup
- Pre-configured extensions:
  - Python, Pylance, Jupyter
  - Ruff (linting & formatting)
  - GitLens, GitHub Copilot
  - Error Lens, TODO Tree
  - Markdown support
- GPU passthrough enabled
- Runs as non-root user

### Customization
Edit `.devcontainer/devcontainer.json` to:
- Add more VS Code extensions
- Change Python interpreter settings
- Modify container features
- Add environment variables

## Tips & Troubleshooting

### Building Images
```bash
# Force rebuild without cache
./scripts/docker_build.sh gpu true

# Build all image variants
make docker-build-all
```

### Custom Container Names
```bash
# Run with custom name
./scripts/docker_run.sh --gpu --name my-experiment
```

### Data Mounting
```bash
# Mount additional data directory
docker run -v /path/to/data:/data:ro ...
```

### Environment Variables
Create a `.env` file for sensitive variables:
```env
WANDB_API_KEY=your_api_key
NEPTUNE_API_TOKEN=your_token
```

### Common Issues

**GPU not detected**:
- Ensure NVIDIA Container Toolkit is installed
- Check Docker Desktop GPU support settings
- Verify with `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

**Permission issues**:
- The dev container runs as non-root user `vscode`
- Use `sudo` for system-level operations
- Check file ownership in mounted volumes

**Out of memory**:
- Increase Docker Desktop memory allocation
- Use `--shm-size=8g` flag for larger shared memory
- Adjust `ulimit` settings in docker-compose.yml
