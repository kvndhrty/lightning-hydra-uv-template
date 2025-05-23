"""Test UV package management integration."""
import subprocess
import sys
from pathlib import Path

import pytest


def test_pyproject_toml_exists():
    """Test that pyproject.toml exists and is valid."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml not found"
    
    # Check it's parseable
    import tomllib
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    # Check required sections
    assert "project" in data
    assert "dependencies" in data["project"]
    assert "build-system" in data
    assert data["project"]["requires-python"] == ">=3.9"


def test_no_setup_py():
    """Test that setup.py has been removed."""
    setup_path = Path(__file__).parent.parent / "setup.py"
    assert not setup_path.exists(), "setup.py should be removed when using pyproject.toml"


def test_no_requirements_txt():
    """Test that requirements.txt has been removed."""
    requirements_path = Path(__file__).parent.parent / "requirements.txt"
    assert not requirements_path.exists(), "requirements.txt should be removed when using pyproject.toml"


def test_makefile_uses_uv():
    """Test that Makefile commands use UV."""
    makefile_path = Path(__file__).parent.parent / "Makefile"
    assert makefile_path.exists(), "Makefile not found"
    
    content = makefile_path.read_text()
    assert "uv run pytest" in content, "Makefile should use 'uv run' for tests"
    assert "uv run python" in content, "Makefile should use 'uv run' for python commands"
    assert "uv pip" in content, "Makefile should include UV pip commands"


def test_console_scripts_defined():
    """Test that console scripts are properly defined in pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    
    import tomllib
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    assert "scripts" in data["project"]
    assert "lht-train" in data["project"]["scripts"]
    assert "lht-eval" in data["project"]["scripts"]


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Project requires Python 3.9+")
def test_python_version_requirement():
    """Test Python version requirement."""
    assert sys.version_info >= (3, 9), "Python 3.9+ is required"


def test_uv_config_in_pyproject():
    """Test UV-specific configuration in pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    
    import tomllib
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    assert "uv" in data["tool"], "UV configuration section missing"
    assert "dev-dependencies" in data["tool"]["uv"]
    
    # Also check ruff is configured (modern linter to replace flake8/black/isort)
    assert "ruff" in data["tool"], "Ruff configuration missing"