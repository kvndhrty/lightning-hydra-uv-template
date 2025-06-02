"""Utilities for file and path operations."""

import json
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd


def safe_load_json(path: Union[str, Path]) -> Optional[dict[str, Any]]:
    """Safely load JSON file with error handling.

    Args:
        path: Path to JSON file.

    Returns:
        Dictionary contents of JSON file, or None if loading fails.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


def safe_save_json(data: dict[str, Any], path: Union[str, Path]) -> bool:
    """Safely save data to JSON file with error handling.

    Args:
        data: Dictionary to save as JSON.
        path: Path to save JSON file.

    Returns:
        True if save was successful, False otherwise.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except (PermissionError, OSError):
        return False


def safe_load_csv(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Safely load CSV file with error handling.

    Args:
        path: Path to CSV file.

    Returns:
        DataFrame contents of CSV file, or None if loading fails.
    """
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, PermissionError):
        return None


def safe_save_csv(df: pd.DataFrame, path: Union[str, Path]) -> bool:
    """Safely save DataFrame to CSV file with error handling.

    Args:
        df: DataFrame to save as CSV.
        path: Path to save CSV file.

    Returns:
        True if save was successful, False otherwise.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return True
    except (PermissionError, OSError):
        return False


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The directory path as a Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_files_by_pattern(
    root_dir: Union[str, Path], pattern: str, recursive: bool = True
) -> list[Path]:
    """Find files matching a pattern in a directory.

    Args:
        root_dir: Root directory to search in.
        pattern: Glob pattern to match files against.
        recursive: Whether to search recursively in subdirectories.

    Returns:
        List of paths matching the pattern.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return []

    if recursive:
        return list(root_dir.rglob(pattern))
    else:
        return list(root_dir.glob(pattern))


def get_file_size_mb(path: Union[str, Path]) -> float:
    """Get file size in megabytes.

    Args:
        path: Path to file.

    Returns:
        File size in MB, or 0.0 if file doesn't exist.
    """
    try:
        return Path(path).stat().st_size / (1024 * 1024)
    except (FileNotFoundError, OSError):
        return 0.0


def copy_file_safe(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """Safely copy a file with error handling.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        True if copy was successful, False otherwise.
    """
    try:
        import shutil

        src, dst = Path(src), Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except (FileNotFoundError, PermissionError, OSError):
        return False


def get_latest_file(directory: Union[str, Path], pattern: str = "*") -> Optional[Path]:
    """Get the most recently modified file in a directory.

    Args:
        directory: Directory to search in.
        pattern: File pattern to match (default: all files).

    Returns:
        Path to the most recent file, or None if no files found.
    """
    directory = Path(directory)
    if not directory.exists():
        return None

    files = list(directory.glob(pattern))
    if not files:
        return None

    return max(files, key=lambda p: p.stat().st_mtime)
