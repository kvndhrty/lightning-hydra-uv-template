"""Experiment management CLI commands (placeholder)."""

from src.cli.base import BaseCLI


class ExperimentCLI(BaseCLI):
    """Experiment management CLI."""

    def __init__(self):
        super().__init__()

    def add_subparser(self, subparsers):
        """Add experiment subcommands to the parser."""
        exp_parser = subparsers.add_parser(
            "exp",
            help="Experiment management commands (coming soon)",
        )
        exp_parser.add_argument(
            "--list",
            action="store_true",
            help="List recent experiments",
        )

    def execute(self, args):
        """Execute experiment command."""
        self.warning_message("Experiment commands coming soon!")
        self.console.print("This will include:")
        self.console.print("  - List recent experiments with metrics")
        self.console.print("  - Compare configs between runs")
        self.console.print("  - Open TensorBoard for specific runs")


# Create singleton instance
experiment_cli = ExperimentCLI()
