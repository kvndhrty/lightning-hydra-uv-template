"""Debug CLI commands (placeholder)."""
import sys

from rich.console import Console


class DebugCLI:
    """Debug command CLI."""

    def __init__(self):
        self.console = Console()

    def add_subparser(self, subparsers):
        """Add debug subcommands to the parser."""
        debug_parser = subparsers.add_parser(
            "debug",
            help="Debug commands (coming soon)",
        )
        debug_parser.add_argument(
            "--cpu",
            action="store_true",
            help="Debug on CPU with single batch",
        )

    def execute(self, args):
        """Execute debug command."""
        self.console.print("[yellow]Debug commands coming soon![/yellow]")
        self.console.print("This will include:")
        self.console.print("  - Single batch CPU debugging")
        self.console.print("  - VS Code launch.json generation")
        self.console.print("  - Profiling support")


# Create singleton instance
debug_cli = DebugCLI()
