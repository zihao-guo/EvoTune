"""
Standalone C++ extraction helpers for the HGS/Crex "raw source" tasks.

This module is intentionally dependency-free (stdlib only: `re`, `logging`)
so it can be imported from the py3.10 `evotune` worker without pulling in
anything else from `packing.evaluate.hgs`.

Frozen interface (relied on by later tasks / the task registry):
    extract_cpp_source(text: str) -> str
    parse_llm_output(text: str) -> tuple[str, str]

Ported from ReEvo's `problems/hgs_share/code_extraction.py`
(`extract_cpp_code` / `_looks_like_cpp_source`), trimmed to the last-fence
rule described in the implementation plan (no raw-text fallback).
"""

import logging
import re

logger = logging.getLogger(__name__)

# Last fenced ```cpp / ```c++ / ```cxx / ```cc block in the text.
_CPP_BLOCK_PATTERN = re.compile(
    r"```(?:cpp|c\+\+|cxx|cc)\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)

# All three markers must be present for a candidate block to be accepted as a
# genuine selectiveRouteExchange C++ source file (not a decoy/partial snippet).
_REQUIRED_MARKERS = (
    '#include "selective_route_exchange.h"',
    "selectiveRouteExchange",
    "ProblemData const &data",
)


def _looks_like_cpp_source(code: str) -> bool:
    return all(marker in code for marker in _REQUIRED_MARKERS)


def extract_cpp_source(text: str) -> str:
    """
    Extract the last fenced ```cpp/c++/cxx/cc code block from `text`.

    Returns the extracted source (stripped) if it contains all of the
    required `selectiveRouteExchange` markers, otherwise returns "" and logs
    a warning. This is the only visibility into extraction failures, since
    callers (model.py::generate_from_server) silently drop empty results.
    """
    matches = _CPP_BLOCK_PATTERN.findall(text or "")
    if not matches:
        logger.warning(
            "extract_cpp_source: no fenced ```cpp/c++/cxx/cc block found in LLM output"
        )
        return ""

    code = matches[-1].strip()
    if not _looks_like_cpp_source(code):
        logger.warning(
            "extract_cpp_source: last fenced cpp block failed marker validation "
            "(missing one or more of %s)",
            _REQUIRED_MARKERS,
        )
        return ""

    return code


def parse_llm_output(text: str) -> tuple[str, str]:
    """
    Registry-facing `parse_llm_output` hook.

    Returns `(cpp_source_or_empty_string, "")`. The empty second element
    keeps the `(function_str, imports_str)` tuple shape used elsewhere in the
    pipeline; C++ candidates are stored as a single raw source file with no
    separate "imports" section.
    """
    return extract_cpp_source(text), ""
