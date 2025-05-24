# UV Package Management Guide

This project uses [UV](https://github.com/astral-sh/uv) for fast, reliable Python package management.

## Installation

### Install UV

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install pipx
pipx install uv
```

### Setup Project

```bash
# Clone the repository
git clone https://github.com/ashleve/lightning-hydra-template
cd lightning-hydra-template

# Create a virtual environment with Python 3.9+
uv sync

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install the project and dependencies
uv pip install -e ".[all]"
```

## Common UV Commands

```bash
# Install production dependencies only
uv pip install -e .

# Install with specific extras (e.g., loggers)
uv pip install -e ".[loggers]"

# Install development dependencies
uv pip install -e ".[dev]"

# Add a new dependency
uv pip install package-name
# Then add it to pyproject.toml manually

# Upgrade all dependencies
uv pip install -U -e ".[all]"

# Show installed packages
uv pip list

# Show dependency tree
uv pip tree
```

## Development Workflow

```bash
# Run training with UV
uv run python src/train.py

# Run tests
uv run pytest

# Run with specific Python version
uv run --python 3.10 python src/train.py

# Use make commands (they now use UV internally)
make train
make test
make format
```

## Why UV?

- **10-100x faster** than pip and pip-tools
- **Reliable** dependency resolution
- **Minimal** - single static binary
- **Compatible** with existing Python packaging standards
- **Cross-platform** support

## Migrating from pip/conda

If you have an existing environment:

```bash
# Export current environment
pip freeze > old_requirements.txt

# Create new UV environment
uv venv
source .venv/bin/activate

# Install from pyproject.toml
uv pip install -e ".[all]"
```
