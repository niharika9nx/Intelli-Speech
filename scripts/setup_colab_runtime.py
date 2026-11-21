#!/usr/bin/env python3
"""Read-only Colab runtime diagnostics for the Intelli-Speech Python environment."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import Iterable


PROJECT_DRIVE_DIR = Path("/content/drive/MyDrive/Intelli-Speech-data")
PAIRED_DATASET_DIR = PROJECT_DRIVE_DIR / "paired_dataset_v3"


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def print_package_compatibility(package_name: str) -> None:
    print(f"{package_name}:")
    print(f"  installed={package_version(package_name)}")
    if importlib.util.find_spec(package_name) is None:
        print("  importable=False")
        return
    print("  importable=True")

    try:
        requirements: Iterable[str] = importlib.metadata.requires(package_name) or []
    except importlib.metadata.PackageNotFoundError:
        requirements = []
    relevant = [
        requirement
        for requirement in requirements
        if requirement.lower().startswith(
            ("numpy", "torch", "torchaudio", "librosa", "soundfile", "opencv")
        )
    ]
    if relevant:
        print("  relevant declared requirements:")
        for requirement in relevant:
            print(f"    {requirement}")


def configure_drive_caches(project_dir: Path) -> dict[str, Path]:
    """Configure persistent cache paths without downloading or installing packages."""
    cache_root = project_dir / "persistent_cache"
    pip_cache = cache_root / "pip"
    hf_cache = cache_root / "huggingface"
    hf_hub_cache = hf_cache / "hub"
    clearvoice_cache = cache_root / "clearvoice_runtime"

    for path in (
        cache_root,
        pip_cache,
        hf_cache,
        hf_hub_cache,
        clearvoice_cache,
    ):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["PIP_CACHE_DIR"] = str(pip_cache)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub_cache)

    return {
        "cache_root": cache_root,
        "pip": pip_cache,
        "huggingface": hf_cache,
        "huggingface_hub": hf_hub_cache,
        "clearvoice": clearvoice_cache,
    }


def main() -> int:
    print("=" * 72)
    print("INTELLI-SPEECH COLAB RUNTIME DIAGNOSTICS")
    print("=" * 72)
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"NumPy: {package_version('numpy')}")
    print(f"PyTorch: {package_version('torch')}")
    print(f"torchaudio: {package_version('torchaudio')}")
    print(f"ClearerVoice: {package_version('clearvoice')}")

    numpy_version = package_version("numpy")
    if numpy_version != "not installed":
        major = numpy_version.split(".", 1)[0]
        if major.isdigit() and int(major) >= 2:
            print(
                "Compatibility warning: ClearerVoice 0.1.2 declares NumPy "
                "<2.0; package compatibility is not yet established."
            )

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("CUDA available: unknown (torch not installed)")

    print_package_compatibility("clearvoice")
    print_package_compatibility("numpy")

    in_colab = importlib.util.find_spec("google.colab") is not None
    print(f"Google Colab environment: {in_colab}")

    if in_colab:
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception as exc:
            print(f"Drive mount failed: {exc}")
            return 1

    if PROJECT_DRIVE_DIR.is_dir():
        print(f"Project Drive directory: {PROJECT_DRIVE_DIR}")
        caches = configure_drive_caches(PROJECT_DRIVE_DIR)
        for name, path in caches.items():
            print(f"  {name} cache: {path}")
    else:
        print(
            "Project Drive directory not found. "
            "Mount Drive and verify the dataset path before continuing."
        )

    if PAIRED_DATASET_DIR.is_dir():
        metadata = PAIRED_DATASET_DIR / "metadata.csv"
        print(f"Paired dataset: {PAIRED_DATASET_DIR}")
        print(f"Metadata: {'found' if metadata.is_file() else 'missing'}")
    else:
        print(f"Paired dataset not found: {PAIRED_DATASET_DIR}")

    print("No package installation or downgrade was performed.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
