"""Test Docker and devcontainer setup."""
import json
import subprocess
from pathlib import Path

import pytest
import yaml


def test_dockerfile_exists():
    """Test that Dockerfile exists."""
    dockerfile = Path(__file__).parent.parent / "Dockerfile"
    dockerfile_dev = Path(__file__).parent.parent / "Dockerfile.dev"

    assert dockerfile.exists(), "Dockerfile not found"
    assert dockerfile_dev.exists(), "Dockerfile.dev not found"


def test_dockerfile_content():
    """Test Dockerfile has required components."""
    dockerfile = Path(__file__).parent.parent / "Dockerfile"
    content = dockerfile.read_text()

    # Check for NVIDIA base image
    assert "FROM nvcr.io/nvidia/pytorch" in content, "Should use NVIDIA PyTorch base image"

    # Check for UV installation
    assert "astral.sh/uv/install.sh" in content, "Should install UV"

    # Check for virtual environment setup
    assert "uv venv" in content, "Should create virtual environment with UV"
    assert "uv pip install" in content, "Should install dependencies with UV"

    # Check for proper working directory
    assert "WORKDIR /workspace" in content, "Should set working directory"


def test_dockerignore_exists():
    """Test that .dockerignore exists and has proper entries."""
    dockerignore = Path(__file__).parent.parent / ".dockerignore"
    assert dockerignore.exists(), ".dockerignore not found"

    content = dockerignore.read_text()
    important_ignores = [".git", "__pycache__", "logs/", ".venv/", "*.egg-info"]

    for ignore in important_ignores:
        assert ignore in content, f"{ignore} should be in .dockerignore"


def test_devcontainer_json():
    """Test .devcontainer/devcontainer.json configuration."""
    devcontainer_path = Path(__file__).parent.parent / ".devcontainer" / "devcontainer.json"
    assert devcontainer_path.exists(), "devcontainer.json not found"

    # Parse JSON (with comments)
    content = devcontainer_path.read_text()
    # Remove comments for JSON parsing
    lines = []
    for line in content.split("\n"):
        if "//" not in line:
            lines.append(line)

    config = json.loads("\n".join(lines))

    # Check essential configuration
    assert "name" in config
    assert "build" in config
    assert "customizations" in config
    assert "vscode" in config["customizations"]

    # Check VS Code extensions
    extensions = config["customizations"]["vscode"]["extensions"]
    required_extensions = ["ms-python.python", "ms-toolsai.jupyter", "charliermarsh.ruff"]
    for ext in required_extensions:
        assert ext in extensions, f"{ext} should be in VS Code extensions"

    # Check GPU support
    assert "runArgs" in config
    assert "--gpus=all" in config["runArgs"], "Should have GPU support"


def test_docker_compose():
    """Test docker-compose.yml configuration."""
    compose_path = Path(__file__).parent.parent / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml not found"

    with open(compose_path) as f:
        config = yaml.safe_load(f)

    # Check services
    assert "services" in config
    assert "dev" in config["services"]

    # Check GPU configuration in dev service
    dev_service = config["services"]["dev"]
    assert "deploy" in dev_service
    assert "resources" in dev_service["deploy"]
    assert "reservations" in dev_service["deploy"]["resources"]
    assert "devices" in dev_service["deploy"]["resources"]["reservations"]

    # Check volumes
    assert "volumes" in dev_service
    assert any("/workspace" in v for v in dev_service["volumes"])

    # Check optional services
    assert "tensorboard" in config["services"]
    assert "jupyter" in config["services"]


def test_docker_scripts():
    """Test Docker helper scripts exist and are executable."""
    scripts_dir = Path(__file__).parent.parent / "scripts"

    docker_build = scripts_dir / "docker_build.sh"
    docker_run = scripts_dir / "docker_run.sh"

    assert docker_build.exists(), "docker_build.sh not found"
    assert docker_run.exists(), "docker_run.sh not found"

    # Check if executable (Unix only)
    import os
    import stat

    if os.name != "nt":  # Not Windows
        assert os.access(docker_build, os.X_OK), "docker_build.sh should be executable"
        assert os.access(docker_run, os.X_OK), "docker_run.sh should be executable"


def test_makefile_docker_commands():
    """Test that Makefile includes Docker commands."""
    makefile = Path(__file__).parent.parent / "Makefile"
    content = makefile.read_text()

    docker_commands = [
        "docker-build",
        "docker-build-gpu",
        "docker-build-dev",
        "docker-run",
        "docker-run-cpu",
        "docker-compose-up",
        "docker-compose-down",
    ]

    for cmd in docker_commands:
        assert f"{cmd}:" in content, f"Makefile should have {cmd} target"


@pytest.mark.skipif(
    subprocess.run(["docker", "--version"], capture_output=True).returncode != 0,
    reason="Docker not installed",
)
def test_dockerfile_syntax():
    """Test Dockerfile syntax is valid (requires Docker)."""
    dockerfile = Path(__file__).parent.parent / "Dockerfile"

    # Use docker build with --check flag (dry run)
    result = subprocess.run(
        ["docker", "build", "--check", "-f", str(dockerfile), "."],
        capture_output=True,
        text=True,
        cwd=dockerfile.parent,
    )

    if result.returncode != 0 and "--check" in result.stderr:
        # Fallback for older Docker versions without --check
        pytest.skip("Docker version doesn't support --check flag")

    assert result.returncode == 0, f"Dockerfile syntax error: {result.stderr}"
