<div align="center">

# Lightning-Hydra-Template

[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10_%7C_3.11_%7C_3.12-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![uv](https://img.shields.io/badge/Package_Manager-UV-green?logo=python&logoColor=white)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/badge/Code%20Style-Ruff-red.svg?labelColor=gray)](https://github.com/astral-sh/ruff) <br>
[![tests](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

A clean PyTorch Lightning + Hydra template for rapid deep learning experimentation ⚡

</div>

## Features

- **Modular Configuration** - Compose experiments with Hydra
- **Fast Package Management** - UV for 10-100x faster installs
- **Enhanced Logging** - Structured JSON/CSV logs with W&B integration
- **CLI Tools** - Checkpoint management and workflow automation
- **Docker Support** - GPU-enabled containers and VS Code integration
- **Best Practices** - Pre-configured linting, testing, and CI/CD

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ashleve/lightning-hydra-template
cd lightning-hydra-template

# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv --python 3.9
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[all]"
```

### Basic Usage

Train model with default configuration:
```bash
python src/train.py

# Train on GPU
python src/train.py trainer=gpu

# Use specific experiment
python src/train.py experiment=example
```

## Project Structure

```
├── configs/                # Hydra configuration files
│   ├── callbacks/         # Callback configurations
│   ├── data/             # Data configurations
│   ├── debug/            # Debugging configurations
│   ├── experiment/       # Experiment configurations
│   ├── logger/           # Logger configurations
│   ├── model/            # Model configurations
│   ├── trainer/          # Trainer configurations
│   ├── eval.yaml         # Evaluation config
│   └── train.yaml        # Training config
├── src/                  # Source code
│   ├── cli/             # CLI tools
│   ├── data/            # Data modules
│   ├── models/          # Model implementations
│   ├── utils/           # Utilities and helpers
│   ├── eval.py          # Evaluation script
│   └── train.py         # Training script
├── tests/               # Test suite
├── .devcontainer/       # VS Code dev container
├── docker/              # Docker configurations
├── docs/                # Documentation
└── pyproject.toml       # Project configuration
```

## Key Features

### 🎯 Hydra Configuration

Override any parameter from command line:
```bash
python src/train.py trainer.max_epochs=20 model.optimizer.lr=1e-4

# Multi-run for hyperparameter search
python src/train.py -m model.optimizer.lr=1e-4,1e-3 data.batch_size=32,64,128
```

### 📊 Enhanced Logging

Structured logging with automatic metric tracking:
```bash
# Use enhanced W&B logging
python src/train.py logger=wandb_enhanced

# Local JSON/CSV logging
python src/train.py logger=json
```

Logs are saved in:
```
logs/train/runs/YYYY-MM-DD_HH-MM-SS/
├── metrics/
│   ├── metrics.json      # All metrics by epoch
│   ├── metrics.csv       # Same in CSV format
│   └── test_results.json # Final test metrics
└── checkpoints/
    └── best.ckpt
```

### 🛠️ CLI Tools

Manage checkpoints and workflows:
```bash
# List available checkpoints
lht ckpt list

# Resume training with original config
lht ckpt resume

# Resume with modifications
lht ckpt resume trainer.max_epochs=100
```

### 🐳 Docker Support

```bash
# Build GPU-enabled image
make docker-build-gpu

# Run container
make docker-run

# Use VS Code Dev Container
# 1. Open in VS Code
# 2. Press F1 → "Reopen in Container"
```

### 🧪 Testing

```bash
# Run all tests
make test

# Run specific tests
pytest tests/test_train.py -v
```

## Configuration Examples

### Training Options

```bash
# Debug mode - 1 epoch with minimal data
python src/train.py debug=default

# Distributed training
python src/train.py trainer=ddp trainer.devices=4

# Mixed precision
python src/train.py trainer=gpu +trainer.precision=16

# Profile performance
python src/train.py debug=profiler
```

### Experiment Management

```bash
# Run predefined experiment
python src/train.py experiment=example

# Hyperparameter search with Optuna
python src/train.py -m hparams_search=mnist_optuna

# Tag experiments
python src/train.py tags=["baseline","v1"]
```

## Development

### Code Quality

```bash
# Format and lint code
make format

# Update dependencies
uv pip install package-name
```

### Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests and linting
4. Submit a pull request

## Documentation

- [UV Package Manager Guide](README_UV.md)
- [Docker Guide](docs/docker_guide.md)
- [CLI Tools Guide](docs/cli_guide.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

<div align="center">
Maintained by the community with ❤️
</div>