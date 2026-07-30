"""Registration factory for ReEvo-style HGS/Pop VRP-variant tasks.

Owns :func:`register_variant`, which a thin ``task_<variant>_hgs_pop.py``
module (e.g. ``task_cvrp_hgs_pop.py``) calls to register a
``<variant>_hgs_pop`` EvoTune task that evolves PyVRP's pure-Python HGS
population parent-selection module (``pyvrp/Population.py``, class
``Population``) for one PyVRP problem variant.

Deliberately kept out of the ``task_*.py`` glob that
``packing.evaluate.import_all_tasks()`` uses to discover task modules -- see
``hgs_task_factory.py``'s docstring for the identical rationale.
``task_cvrp_hgs_pop.py`` is the sole trigger for CVRP_POP registration.

Wiring per variant mirrors ``hgs_task_factory.register_variant`` (see that
module for the shared conventions) with these differences:
- ``get_initial_func``: pristine seed Python (``seed/Population.original.py``).
- ``evaluate_func``: the SAME ``hgs_common.evaluate_candidate`` -- routing to
  the no-compile bridge path is entirely task-cfg-driven (D1 of
  ``docs/superpowers/plans/2026-07-30-hgs-pop-evolution.md``), not a
  separate evaluate function.
- prompts: ``prompts/pop_common/*.txt`` + ``prompts/<prompt_dir>/*.txt``
  (ported from ReEvo's ``prompts/HGS_Pop/<prompt_dir>_hgs/``).
- ``parse_llm_output``: ``extraction_pop.parse_llm_output`` (AST-based,
  not the C++ fence/marker scanner in ``extraction.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from packing.evaluate.hgs import extraction_pop
from packing.evaluate.hgs.hgs_common import evaluate_candidate
from packing.evaluate.registry import TASK_REGISTRY

_HGS_DIR = Path(__file__).resolve().parent
_SEED_PATH = _HGS_DIR / "seed" / "Population.original.py"
_PROMPTS_DIR = _HGS_DIR / "prompts"
_COMMON_PROMPTS_DIR = _PROMPTS_DIR / "pop_common"

# Exported Python symbol shared by every variant (frozen signature, see
# prompts/pop_<variant>/func_signature.txt); this is also each variant
# yaml's `function_str_to_extract`.
FUNC_NAME = "Population"

# set -> dataset size. Frozen: main.py calls generate_input(cfg.task, "train")
# once at startup; continuous_execution.py calls generate_input(cfg.task,
# evalset/dataset) with cfg.evalset ("trainperturbedset" by default) /
# cfg.testset ("testset" by default) -- see configs/config.yaml. Identical to
# hgs_task_factory._DATASET_SET_SIZES.
_DATASET_SET_SIZES: dict[str, int] = {
    "train": 51,
    "trainperturbedset": 101,
    "testset": 201,
}


def _read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _seed_source() -> str:
    return _SEED_PATH.read_text(encoding="utf-8")


def _make_get_initial_func(func_name: str) -> Callable[[Any], tuple[Callable, str]]:
    def get_initial_func(cfg):
        seed_source = _seed_source()

        def _hgs_pop_seed_stub(*args, **kwargs):
            raise RuntimeError(
                "hgs pop seed stub is never invoked directly; its Python source is "
                "carried via _raw_source and evaluated out-of-process by "
                "hgs_common.evaluate_candidate."
            )

        _hgs_pop_seed_stub._raw_source = seed_source
        return _hgs_pop_seed_stub, func_name

    return get_initial_func


def _make_generate_input(default_variant: str, default_dataset_root: str, default_baseline_root: str):
    def generate_input(cfg, set: str) -> dict:
        if set not in _DATASET_SET_SIZES:
            raise ValueError(f"Unknown set: {set}")
        size = _DATASET_SET_SIZES[set]

        variant = getattr(cfg, "variant", default_variant) if cfg is not None else default_variant
        dataset_root = getattr(cfg, "dataset_root", default_dataset_root) if cfg is not None else default_dataset_root
        baseline_root = getattr(cfg, "baseline_root", default_baseline_root) if cfg is not None else default_baseline_root

        return {
            "variant": variant,
            "instances_dir": f"{dataset_root}/{variant}/{size}",
            "baselines_dir": f"{baseline_root}/{variant}/{size}/pyvrp",
            "size": size,
        }

    return generate_input


def register_variant(
    task_name: str,
    variant: str,
    prompt_dir: str,
    *,
    dataset_root: str = "utils/data/dataset/generated",
    baseline_root: str = "utils/data/dataset/opt",
) -> None:
    """Registers ``task_name`` (e.g. ``"cvrp_hgs_pop"``) as an EvoTune task
    that evolves PyVRP's ``Population.py`` parent-selection module for
    ``variant`` (e.g. ``"CVRP"``). ``prompt_dir`` names a subdirectory of
    ``prompts/`` (e.g. ``"pop_cvrp"``) holding that variant's
    ``func_desc.txt``, ``func_signature.txt``, and ``external_knowledge.txt``
    (ported from ReEvo's ``prompts/HGS_Pop/<prompt_dir>_hgs/``).

    ``dataset_root``/``baseline_root`` default to the standard repo-relative
    locations but can be overridden per call; the corresponding task yaml
    keys (``cfg.task.variant``/``dataset_root``/``baseline_root``) take
    precedence at prompt/dataset-resolution time via ``generate_input`` --
    identical contract to ``hgs_task_factory.register_variant``.
    """
    variant_prompts_dir = _PROMPTS_DIR / prompt_dir

    system_generator = _read_prompt(_COMMON_PROMPTS_DIR / "system_generator.txt")
    func_signature = _read_prompt(variant_prompts_dir / "func_signature.txt")
    func_desc = _read_prompt(variant_prompts_dir / "func_desc.txt")
    external_knowledge = _read_prompt(variant_prompts_dir / "external_knowledge.txt")
    output_contract = _read_prompt(_COMMON_PROMPTS_DIR / "output_contract.txt")

    system_prompt = "\n\n".join(
        [system_generator, func_signature, func_desc, external_knowledge, output_contract]
    )

    append_prompt = _read_prompt(_COMMON_PROMPTS_DIR / "append_instruction.txt").format(func_name=FUNC_NAME)

    TASK_REGISTRY.register(
        task_name,
        generate_input=_make_generate_input(variant, dataset_root, baseline_root),
        evaluate_func=evaluate_candidate,
        get_initial_func=_make_get_initial_func(FUNC_NAME),
        system_prompt=system_prompt,
        append_prompt=append_prompt,
        parse_llm_output=extraction_pop.parse_llm_output,
    )
