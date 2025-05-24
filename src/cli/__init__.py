"""Lightning-Hydra-Template CLI tools."""
import sys
from typing import Optional

from src.cli.checkpoint import checkpoint_cli
from src.cli.debug import debug_cli
from src.cli.experiment import experiment_cli


def main(argv: Optional[list[str]] = None):
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="lht",
        description="Lightning-Hydra-Template CLI tools",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add subcommands
    checkpoint_cli.add_subparser(subparsers)
    debug_cli.add_subparser(subparsers)
    experiment_cli.add_subparser(subparsers)

    # Parse arguments
    args = parser.parse_args(argv)

    # Execute command
    if args.command == "ckpt":
        checkpoint_cli.execute(args)
    elif args.command == "debug":
        debug_cli.execute(args)
    elif args.command == "exp":
        experiment_cli.execute(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
