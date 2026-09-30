"""Защитные тесты доменной структуры Python-пакета."""

from pathlib import Path


def test_package_root_contains_only_entrypoint_and_package_marker() -> None:
    package_root = Path(__file__).parents[2] / "src" / "mini_llm"
    root_python_files = {path.name for path in package_root.glob("*.py")}

    assert root_python_files == {"__init__.py", "main.py"}


def test_ui_placeholder_contains_no_implementation() -> None:
    package_root = Path(__file__).parents[2] / "src" / "mini_llm"
    ui_files = {path.name for path in (package_root / "ui").iterdir()}

    assert ui_files <= {".gitkeep"}
