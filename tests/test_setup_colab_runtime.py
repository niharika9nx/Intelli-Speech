from pathlib import Path

from scripts.setup_colab_runtime import configure_drive_caches


def test_configure_drive_caches_creates_only_under_project(tmp_path: Path) -> None:
    caches = configure_drive_caches(tmp_path)

    assert caches["cache_root"].is_dir()
    assert caches["pip"].is_dir()
    assert caches["huggingface"].is_dir()
    assert caches["huggingface_hub"].is_dir()
    assert caches["clearvoice"].is_dir()
