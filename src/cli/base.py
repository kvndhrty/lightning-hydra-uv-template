"""Base CLI class for common functionality across CLI modules."""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rich.console import Console


class BaseCLI(ABC):
    """Base class for CLI commands with common functionality."""

    def __init__(self) -> None:
        """Initialize base CLI with console."""
        self.console = Console()

    @abstractmethod
    def add_subparser(self, subparsers: Any) -> None:
        """Add subcommands to the parser.

        Args:
            subparsers: The argparse subparsers object to add commands to.
        """
        pass

    @abstractmethod
    def execute(self, args: Any) -> None:
        """Execute the CLI command.

        Args:
            args: Parsed command line arguments.
        """
        pass

    def error_exit(self, message: str, exit_code: int = 1) -> None:
        """Print error message and exit.

        Args:
            message: Error message to display.
            exit_code: Exit code to use (default: 1).
        """
        self.console.print(f"[red]{message}[/red]")
        sys.exit(exit_code)

    def success_message(self, message: str) -> None:
        """Print success message.

        Args:
            message: Success message to display.
        """
        self.console.print(f"[green]{message}[/green]")

    def warning_message(self, message: str) -> None:
        """Print warning message.

        Args:
            message: Warning message to display.
        """
        self.console.print(f"[yellow]{message}[/yellow]")

    def info_message(self, message: str) -> None:
        """Print info message.

        Args:
            message: Info message to display.
        """
        self.console.print(f"[blue]{message}[/blue]")

    def validate_directory(self, path: Path, name: str = "Directory") -> Path:
        """Validate that a directory exists.

        Args:
            path: Path to validate.
            name: Name of the directory for error messages.

        Returns:
            The validated path.

        Raises:
            SystemExit: If directory doesn't exist.
        """
        if not path.exists():
            self.error_exit(f"{name} not found: {path}")
        if not path.is_dir():
            self.error_exit(f"{name} is not a directory: {path}")
        return path

    def validate_file(self, path: Path, name: str = "File") -> Path:
        """Validate that a file exists.

        Args:
            path: Path to validate.
            name: Name of the file for error messages.

        Returns:
            The validated path.

        Raises:
            SystemExit: If file doesn't exist.
        """
        if not path.exists():
            self.error_exit(f"{name} not found: {path}")
        if not path.is_file():
            self.error_exit(f"{name} is not a file: {path}")
        return path

    def get_project_root(self) -> Path:
        """Get the project root directory.

        Returns:
            Path to the project root.
        """
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        return Path.cwd()

    def ensure_directory(self, path: Path) -> Path:
        """Ensure directory exists, create if needed.

        Args:
            path: Directory path to ensure exists.

        Returns:
            The directory path.
        """
        path.mkdir(parents=True, exist_ok=True)
        return path
