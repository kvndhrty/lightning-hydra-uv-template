# CLI Tools Guide

The Lightning-Hydra-Template includes powerful CLI tools to streamline common ML workflows.

## Installation

After installing the project, the `lht` command will be available:

```bash
# Install with UV
uv pip install -e .

# Verify installation
lht --help
```

## Checkpoint Management

### List Checkpoints

View all available checkpoints from your training runs:

```bash
# List recent checkpoints
lht ckpt list

# List from specific directory
lht ckpt list --logs-dir custom_logs/

# Show more checkpoints
lht ckpt list --limit 20
```

Output example:
```
                    Available Checkpoints (showing up to 10)                    
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ #   ┃ Checkpoint                         ┃ Epoch ┃ Created          ┃ Config ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ 1   │ 2024-01-20_15-30-45/best.ckpt      │ 42    │ 2024-01-20 15:30 │ ✓      │
│ 2   │ 2024-01-20_15-30-45/last.ckpt      │ 50    │ 2024-01-20 15:45 │ ✓      │
│ 3   │ 2024-01-19_10-15-00/epoch_030.ckpt │ 30    │ 2024-01-19 10:45 │ ✓      │
└─────┴────────────────────────────────────┴───────┴──────────────────┴────────┘
```

### Resume Training

Resume training from a checkpoint with the original configuration:

```bash
# Interactive selection
lht ckpt resume

# Preview the command without running
lht ckpt resume --dry-run

# Resume with additional overrides
lht ckpt resume trainer.max_epochs=100 model.lr=0.0001

# Resume from best checkpoint automatically
lht ckpt resume --best
```

The resume command will:
1. Show available checkpoints
2. Let you select one interactively
3. Load the original training configuration
4. Add the checkpoint path
5. Execute the training command

Example interaction:
```
Recent Checkpoints:
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ # ┃ Checkpoint                             ┃ Epoch ┃ Created          ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━╇
│ 1 │ 2024-01-20_15-30-45/best.ckpt (best)   │ 42    │ 2024-01-20 15:30 │
│ 2 │ 2024-01-20_15-30-45/last.ckpt (last)   │ 50    │ 2024-01-20 15:45 │
│ 3 │ 2024-01-19_10-15-00/epoch_030.ckpt     │ 30    │ 2024-01-19 10:45 │
└───┴────────────────────────────────────────┴───────┴──────────────────┘

Select checkpoint [1-3] (1): 1

Resume command:
python src/train.py experiment=example model.lr=0.001 trainer.max_epochs=50 ckpt_path=logs/train/runs/2024-01-20_15-30-45/checkpoints/best.ckpt

Execute this command? [y/n] (y): y

Starting training...
```

## Debug Tools (Coming Soon)

```bash
# Single-batch CPU debugging
lht debug cpu

# Generate VS Code launch.json
lht debug vscode

# Profile training
lht debug profile
```

## Experiment Tools (Coming Soon)

```bash
# List recent experiments
lht exp list

# Compare two experiments
lht exp diff run1 run2

# Open TensorBoard for specific run
lht exp tb 2024-01-20_15-30-45
```

## Integration with VS Code

For debugging with VS Code, you can generate a launch configuration:

```bash
# Coming soon
lht debug vscode --checkpoint path/to/checkpoint.ckpt
```

This will create a `.vscode/launch.json` with the correct Hydra configuration for debugging.

## Tips

1. **Checkpoint Selection**: The CLI shows checkpoints sorted by creation time, with the newest first.

2. **Config Preservation**: The original Hydra configuration is automatically loaded from the checkpoint's run directory.

3. **Override Support**: You can add new overrides when resuming, which will be applied on top of the original configuration.

4. **Dry Run**: Always use `--dry-run` first to preview the command that will be executed.

5. **Best Checkpoint**: Use `--best` to automatically select checkpoints named "best.ckpt" or similar.