"""Checkpoint management CLI commands."""
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from src.utils.checkpoint_utils import (
    CheckpointInfo,
    get_best_checkpoint,
    get_latest_checkpoint,
    get_resume_command,
    scan_checkpoints,
)


class CheckpointCLI:
    """Checkpoint management CLI."""
    
    def __init__(self):
        self.console = Console()
    
    def add_subparser(self, subparsers):
        """Add checkpoint subcommands to the parser."""
        ckpt_parser = subparsers.add_parser(
            "ckpt",
            help="Checkpoint management commands",
        )
        
        ckpt_subparsers = ckpt_parser.add_subparsers(
            dest="ckpt_command",
            help="Checkpoint commands",
        )
        
        # List checkpoints
        list_parser = ckpt_subparsers.add_parser(
            "list",
            help="List available checkpoints",
        )
        list_parser.add_argument(
            "--logs-dir",
            type=Path,
            default=Path("logs"),
            help="Logs directory (default: logs)",
        )
        list_parser.add_argument(
            "--task",
            default="train",
            help="Task name (default: train)",
        )
        list_parser.add_argument(
            "--limit",
            type=int,
            default=10,
            help="Maximum number of checkpoints to show (default: 10)",
        )
        
        # Resume from checkpoint
        resume_parser = ckpt_subparsers.add_parser(
            "resume",
            help="Resume training from a checkpoint",
        )
        resume_parser.add_argument(
            "--logs-dir",
            type=Path,
            default=Path("logs"),
            help="Logs directory (default: logs)",
        )
        resume_parser.add_argument(
            "--task",
            default="train",
            help="Task name (default: train)",
        )
        resume_parser.add_argument(
            "--best",
            action="store_true",
            help="Use best checkpoint instead of latest",
        )
        resume_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show command without executing",
        )
        resume_parser.add_argument(
            "overrides",
            nargs="*",
            help="Additional Hydra overrides",
        )
    
    def execute(self, args):
        """Execute checkpoint command."""
        if args.ckpt_command == "list":
            self.list_checkpoints(args)
        elif args.ckpt_command == "resume":
            self.resume_checkpoint(args)
        else:
            self.console.print("[red]Please specify a checkpoint command: list, resume[/red]")
            sys.exit(1)
    
    def list_checkpoints(self, args):
        """List available checkpoints."""
        if not args.logs_dir.exists():
            self.console.print(f"[red]Logs directory not found: {args.logs_dir}[/red]")
            sys.exit(1)
        
        checkpoints = scan_checkpoints(args.logs_dir, args.task)
        
        if not checkpoints:
            self.console.print(f"[yellow]No checkpoints found in {args.logs_dir}/{args.task}[/yellow]")
            return
        
        # Create table
        table = Table(title=f"Available Checkpoints (showing up to {args.limit})")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Checkpoint", style="green")
        table.add_column("Epoch", style="yellow")
        table.add_column("Metric", style="red")
        table.add_column("Created", style="blue")
        table.add_column("Config", style="magenta")
        
        for i, ckpt in enumerate(checkpoints[:args.limit], 1):
            epoch_str = str(ckpt.epoch) if ckpt.epoch is not None else "-"
            time_str = ckpt.timestamp.strftime("%Y-%m-%d %H:%M") if ckpt.timestamp else "-"
            config_str = "✓" if ckpt.has_config else "✗"
            
            # Format metric
            metric_str = "-"
            if ckpt.metric_value is not None and ckpt.metric_name:
                metric_str = f"{ckpt.metric_name}: {ckpt.metric_value:.4f}"
            elif ckpt.metrics:
                # Try to find a key metric
                for key in ["val/acc", "val/accuracy", "test/acc"]:
                    if key in ckpt.metrics:
                        metric_str = f"{key}: {ckpt.metrics[key]:.4f}"
                        break
            
            table.add_row(
                str(i),
                ckpt.name,
                epoch_str,
                metric_str,
                time_str,
                config_str,
            )
        
        self.console.print(table)
        
        if len(checkpoints) > args.limit:
            self.console.print(
                f"\n[dim]Showing {args.limit} of {len(checkpoints)} checkpoints. "
                f"Use --limit to see more.[/dim]"
            )
    
    def resume_checkpoint(self, args):
        """Resume training from a checkpoint."""
        if not args.logs_dir.exists():
            self.console.print(f"[red]Logs directory not found: {args.logs_dir}[/red]")
            sys.exit(1)
        
        # Get checkpoint
        if args.best:
            checkpoint = get_best_checkpoint(args.logs_dir, args.task)
            if not checkpoint:
                self.console.print("[red]No best checkpoint found[/red]")
                sys.exit(1)
        else:
            # Show selection menu
            checkpoint = self._select_checkpoint(args.logs_dir, args.task)
            if not checkpoint:
                return
        
        # Check if config exists
        if not checkpoint.has_config:
            self.console.print(
                f"[red]No config found for checkpoint: {checkpoint.name}[/red]\n"
                f"[yellow]Config should be at: {checkpoint.config_path}[/yellow]"
            )
            sys.exit(1)
        
        # Generate resume command
        executable, cmd_args = get_resume_command(
            checkpoint,
            additional_overrides=args.overrides
        )
        
        # Show command
        full_command = f"{executable} {' '.join(cmd_args)}"
        self.console.print(f"\n[bold]Resume command:[/bold]")
        self.console.print(f"[cyan]{full_command}[/cyan]")
        
        if args.dry_run:
            self.console.print("\n[yellow]Dry run mode - command not executed[/yellow]")
            return
        
        # Confirm execution
        if not Prompt.ask("\nExecute this command?", choices=["y", "n"], default="y") == "y":
            self.console.print("[yellow]Cancelled[/yellow]")
            return
        
        # Execute command
        self.console.print("\n[green]Starting training...[/green]\n")
        try:
            subprocess.run([executable] + cmd_args, check=True)
        except subprocess.CalledProcessError as e:
            self.console.print(f"[red]Training failed with exit code {e.returncode}[/red]")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Training interrupted[/yellow]")
            sys.exit(130)
    
    def _select_checkpoint(self, logs_dir: Path, task: str) -> Optional[CheckpointInfo]:
        """Interactive checkpoint selection."""
        checkpoints = scan_checkpoints(logs_dir, task)
        
        if not checkpoints:
            self.console.print(f"[yellow]No checkpoints found in {logs_dir}/{task}[/yellow]")
            return None
        
        # Show recent checkpoints
        self.console.print("\n[bold]Recent Checkpoints:[/bold]")
        
        table = Table()
        table.add_column("#", style="cyan", width=3)
        table.add_column("Checkpoint", style="green")
        table.add_column("Epoch", style="yellow")
        table.add_column("Created", style="blue")
        
        # Show up to 10 recent checkpoints
        display_checkpoints = checkpoints[:10]
        for i, ckpt in enumerate(display_checkpoints, 1):
            epoch_str = str(ckpt.epoch) if ckpt.epoch is not None else "-"
            time_str = ckpt.timestamp.strftime("%Y-%m-%d %H:%M") if ckpt.timestamp else "-"
            
            # Highlight best/last checkpoints
            name = ckpt.name
            if "best" in ckpt.path.name:
                name = f"[bold]{name} (best)[/bold]"
            elif "last" in ckpt.path.name:
                name = f"{name} (last)"
            
            table.add_row(str(i), name, epoch_str, time_str)
        
        self.console.print(table)
        
        # Get selection
        if len(display_checkpoints) == 1:
            return display_checkpoints[0]
        
        choice = IntPrompt.ask(
            f"\nSelect checkpoint [1-{len(display_checkpoints)}]",
            default=1,
        )
        
        if 1 <= choice <= len(display_checkpoints):
            return display_checkpoints[choice - 1]
        else:
            self.console.print("[red]Invalid selection[/red]")
            return None


# Create singleton instance
checkpoint_cli = CheckpointCLI()