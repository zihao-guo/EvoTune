import importlib
import pathlib


def _module_from_path(path: pathlib.Path, base_path: pathlib.Path) -> str:
    rel_path = path.relative_to(base_path.parent).with_suffix("")
    return ".".join(["packing"] + list(rel_path.parts))


def import_all_tasks():
    base_path = pathlib.Path(__file__).parent
    for path in base_path.rglob("task_*.py"):
        importlib.import_module(_module_from_path(path, base_path))


def import_task(task_name: str) -> bool:
    """Import only the module that matches the requested task name."""
    base_path = pathlib.Path(__file__).parent
    normalized_name = task_name.replace("_", "").lower()

    for path in base_path.rglob("task_*.py"):
        module_task_name = path.stem.removeprefix("task_").replace("_", "").lower()
        if module_task_name == normalized_name:
            importlib.import_module(_module_from_path(path, base_path))
            return True

    return False
