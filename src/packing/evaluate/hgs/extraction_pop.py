"""
Standalone Python extraction helpers for the HGS/Pop "raw source" tasks
(D2 of ``docs/superpowers/plans/2026-07-30-hgs-pop-evolution.md``).

This module is intentionally dependency-free (stdlib only: `ast`, `re`,
`logging`) so it can be imported from the py3.10 `evotune` worker without
pulling in anything else from `packing.evaluate.hgs`. Mirrors
``extraction.py`` (the Crex/C++ counterpart) but validates candidates via
`ast.parse` instead of a marker-substring check, since the candidate here is
a complete Python module, not an isolated function body.

Frozen interface (relied on by ``hgs_pop_factory.py``):
    extract_population_code(text: str) -> str
    parse_llm_output(text: str) -> tuple[str, str]

Ported (last-fence-wins scanning + last-resort raw fallback) from ReEvo's
generic Python code extraction, adapted to require a top-level
``class Population`` with the three preserved method names.
"""

import ast
import logging
import re

logger = logging.getLogger(__name__)

# Every fenced ```python / ```py block in the text (checked in reverse order
# -- the LAST candidate block a well-behaved LLM reply contains is the one
# meant to be read; see prompts/pop_common/output_contract.txt).
_PY_BLOCK_PATTERN = re.compile(
    r"```(?:python|py)\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)

# A genuine Population.py candidate must define a top-level `class Population`
# whose methods include (at least) these three preserved public signatures
# (see prompts/pop_cvrp/func_signature.txt).
_REQUIRED_METHODS = frozenset({"select", "tournament", "_tournament"})


def _validate_population_code(code: str) -> bool:
    """True iff `code` parses as Python and contains a top-level
    `class Population` whose method names are a superset of
    `_REQUIRED_METHODS`."""
    if not code.strip():
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == "Population"):
            continue
        method_names = {
            item.name for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if _REQUIRED_METHODS <= method_names:
            return True

    return False


def extract_population_code(text: str) -> str:
    """
    Extract a `Population.py` candidate from `text`.

    Scans every fenced ```python/```py block in REVERSE order and returns
    the first one that parses as valid Python and contains a top-level
    `class Population` with `select`/`tournament`/`_tournament` methods. If
    no fenced block validates (including when there are no fenced blocks at
    all), falls back to the raw stripped `text` itself. Returns "" and logs
    a warning if nothing validates. This is the only visibility into
    extraction failures, since callers (model.py::generate_from_server)
    silently drop empty results.
    """
    matches = _PY_BLOCK_PATTERN.findall(text or "")
    for match in reversed(matches):
        code = match.strip()
        if _validate_population_code(code):
            return code

    raw = (text or "").strip()
    if _validate_population_code(raw):
        return raw

    logger.warning(
        "extract_population_code: no fenced ```python/```py block (or raw "
        "fallback) contained a valid top-level `class Population` with methods %s",
        sorted(_REQUIRED_METHODS),
    )
    return ""


def parse_llm_output(text: str) -> tuple[str, str]:
    """
    Registry-facing `parse_llm_output` hook.

    Returns `(population_source_or_empty_string, "")`. The empty second
    element keeps the `(function_str, imports_str)` tuple shape used
    elsewhere in the pipeline; Python candidates are stored as a single raw
    source file with no separate "imports" section.
    """
    return extract_population_code(text), ""
