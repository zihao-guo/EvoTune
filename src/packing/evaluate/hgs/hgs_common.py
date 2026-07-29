"""py3.10-side glue for the HGS/Crex evaluation pipeline.

Owns two frozen interfaces (see
``docs/superpowers/plans/2026-07-28-hgs-crex-evolution.md``, Task 4):

- :func:`ensure_slot` -- idempotent, flock-guarded priming of a persistent
  per-worker sandbox (a full copy of ``pyvrp_rep`` with an incremental meson
  build) under ``<repo>/scratch/hgs/<VARIANT>/w<slot>/pyvrp_rep``.
- :func:`evaluate_candidate` -- the ``evaluate_func`` a ``cvrp_hgs``-style
  task registers. Routes a candidate ``.cpp`` to the right sandbox, shells
  out to the py3.12 bridge (``hgs_eval_py312.py``, Task 3) to compile+solve,
  and maps the result onto a ``FunctionClass`` using the exact fields/flags
  ``task_tsp.py::evaluate_func`` uses (score, true_score, fail_flag,
  fail_exception, correct_flag).

This module is imported by the py3.10 ``evotune`` worker only; it never
imports the py3.12-only ``hgs_eval_py312.py`` module, it *shells out* to it.
It must not import ``extraction.py`` (Task 2, not part of this task).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import signal
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bridge interpreter's bin dir; must lead PATH so meson.build's bare `python`
# shell-outs resolve to the env with meson/ninja/pybind11 (mirrors
# hgs_eval_py312.py::RLFT_BIN exactly -- keep in sync if that ever changes).
RLFT_BIN = "/home/zguo/miniconda/envs/rlft/bin"
DEFAULT_PY312 = f"{RLFT_BIN}/python3.12"

HGS_EVAL_SCRIPT = Path(__file__).resolve().with_name("hgs_eval_py312.py")

# The pristine seed (D5: same seed for all six variants) and the vendored
# copy it must stay byte-identical to (checked lazily, once per repo root).
# Committed under this module's own seed/ dir (NOT utils/data/Operator/,
# which is gitignored and absent on a fresh checkout -- see
# hgs_task_factory._SEED_PATH, the other consumer of this same file).
PRISTINE_SEED_PATH = Path(__file__).resolve().parent / "seed" / "selective_route_exchange.original.cpp"
VENDORED_SEED_RELATIVE = Path("pyvrp_rep/pyvrp/cpp/crossover/selective_route_exchange.cpp")
CANDIDATE_RELATIVE_PATH = Path("pyvrp") / "cpp" / "crossover" / "selective_route_exchange.cpp"

DIAGNOSTIC_TAIL_CHARS = 4000
_COMPILER_DIAGNOSTIC_MARKERS = ("error:", "fatal error", "undefined reference")

_pristine_seed_cache: dict[Path, str] = {}


# ---------------------------------------------------------------------------
# Small config/helper utilities
# ---------------------------------------------------------------------------


def _cfg_get(cfg_node: Any, key: str, default: Any = None) -> Any:
    """Defensive read that works for OmegaConf DictConfig, plain dicts, and
    plain objects alike (the codebase convention is direct attribute access
    on ``cfg.task``, but new optional hgs knobs should not hard-fail when a
    task yaml omits them)."""
    if cfg_node is None:
        return default
    getter = getattr(cfg_node, "get", None)
    if callable(getter):
        try:
            value = getter(key, default)
            return default if value is None else value
        except Exception:
            pass
    value = getattr(cfg_node, key, default)
    return default if value is None else value


def _repo_root() -> Path:
    # src/packing/evaluate/hgs/hgs_common.py -> parents[4] is the repo root.
    root = Path(__file__).resolve().parents[4]
    if not (root / "pyvrp_rep").is_dir():
        raise RuntimeError(
            f"hgs: resolved repo root {root} does not contain pyvrp_rep; "
            "the parents[] depth in hgs_common._repo_root no longer matches "
            "the file's location."
        )
    return root


def _pristine_seed_text(repo_root: Path) -> str:
    """Returns the pristine Crex seed, verifying it is byte-identical to the
    vendored copy under pyvrp_rep the first time it's read for a given repo
    root (cheap: ~7.6KB file)."""
    cached = _pristine_seed_cache.get(repo_root)
    if cached is not None:
        return cached

    pristine_bytes = PRISTINE_SEED_PATH.read_bytes()

    vendored_path = repo_root / VENDORED_SEED_RELATIVE
    if vendored_path.is_file():
        vendored_bytes = vendored_path.read_bytes()
        if vendored_bytes != pristine_bytes:
            raise RuntimeError(
                f"hgs: pristine seed mismatch -- {PRISTINE_SEED_PATH} differs from "
                f"{vendored_path}; refusing to use it as ground truth."
            )

    text = pristine_bytes.decode("utf-8")
    _pristine_seed_cache[repo_root] = text
    return text


# ---------------------------------------------------------------------------
# ensure_slot: persistent per-worker sandbox priming
# ---------------------------------------------------------------------------


def _sandbox_is_valid(sandbox: Path) -> bool:
    if not sandbox.is_dir():
        return False
    if not (sandbox / "build" / "build.ninja").is_file():
        return False
    if not any((sandbox / "pyvrp" / "crossover").glob("_crossover.cpython-*.so")):
        return False
    if not (sandbox / CANDIDATE_RELATIVE_PATH).is_file():
        return False
    return True


def _run_step(step_args: list[str], cwd: Path, env: dict) -> None:
    proc = subprocess.run(
        step_args,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"hgs: `{' '.join(step_args)}` (cwd={cwd}) failed with code {proc.returncode}:\n"
            f"{proc.stdout[-DIAGNOSTIC_TAIL_CHARS:]}"
        )


def _prime_sandbox(repo_root: Path, sandbox: Path) -> None:
    """Fresh copytree of pyvrp_rep + full meson setup/compile/install."""
    source = repo_root / "pyvrp_rep"
    sandbox.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: `/build-*/` (anchored, hyphen required) -- NOT the glob `build*`,
    # which would also match and exclude the required `buildtools/`.
    rsync_cmd = [
        "rsync", "-a", "--delete",
        "--exclude=/build/",
        "--exclude=/build-*/",
        "--exclude=.git/",
        "--exclude=/docs/",
        "--exclude=/tests/",
        "--exclude=__pycache__/",
        f"{source}/", f"{sandbox}/",
    ]
    proc = subprocess.run(rsync_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"hgs: rsync priming of {sandbox} failed:\n{proc.stdout[-DIAGNOSTIC_TAIL_CHARS:]}")

    env = os.environ.copy()
    env["PATH"] = f"{RLFT_BIN}:{env.get('PATH', '')}"

    build_dir = sandbox / "build"
    _run_step(
        ["meson", "setup", "build", "--buildtype", "release", "-Dwerror=false",
         f"-Dpython.platlibdir={sandbox}"],
        cwd=sandbox, env=env,
    )
    _run_step(["meson", "compile", "-C", str(build_dir)], cwd=sandbox, env=env)
    _run_step(["meson", "install", "-C", str(build_dir)], cwd=sandbox, env=env)


def _restore_pristine_cpp(repo_root: Path, sandbox: Path) -> None:
    pristine_text = _pristine_seed_text(repo_root)
    target = sandbox / CANDIDATE_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    current = target.read_text(encoding="utf-8") if target.is_file() else None
    if current != pristine_text:
        target.write_text(pristine_text, encoding="utf-8")


def _slot_paths(task_cfg: Any, repo_root: Path, variant: str, slot: int) -> tuple[Path, Path]:
    sandbox_root = repo_root / str(_cfg_get(task_cfg, "sandbox_root", "scratch/hgs"))
    slot_dir = sandbox_root / variant / f"w{slot}"
    return slot_dir, slot_dir / "pyvrp_rep"


def _ensure_slot_locked(repo_root: Path, variant: str, slot: int, sandbox: Path) -> None:
    """Primes/repairs ``sandbox`` and restores the pristine candidate .cpp.
    Assumes the caller already holds the per-slot flock (LOCK_EX)."""
    if not _sandbox_is_valid(sandbox):
        logging.info(f"[hgs] priming sandbox variant={variant} slot={slot} at {sandbox}")
        if sandbox.exists():
            shutil.rmtree(sandbox)
        _prime_sandbox(repo_root, sandbox)
        logging.info(f"[hgs] sandbox variant={variant} slot={slot} primed successfully")
    _restore_pristine_cpp(repo_root, sandbox)


def ensure_slot(task_cfg: Any, variant: str, slot: int) -> Path:
    """Ensures ``<repo>/scratch/hgs/<VARIANT>/w<slot>/pyvrp_rep`` exists and
    is built; primes it (copytree + meson setup/compile/install) if missing
    or corrupt. Idempotent and flock-guarded across concurrent callers for
    the same slot. Always leaves the candidate .cpp at the pristine seed
    before returning. Raises on unrecoverable priming failures (loud infra
    failure, never silently returns a broken sandbox).

    Standalone helper (e.g. for warmup scripts/tests); ``evaluate_candidate``
    uses :func:`_held_slot` instead so the same flock spans priming *and* the
    subsequent compile+solve bridge calls."""
    repo_root = _repo_root()
    slot_dir, sandbox = _slot_paths(task_cfg, repo_root, variant, slot)
    slot_dir.mkdir(parents=True, exist_ok=True)

    lock_path = slot_dir / ".ensure_slot.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            _ensure_slot_locked(repo_root, variant, slot, sandbox)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return sandbox


@contextmanager
def _held_slot(task_cfg: Any, repo_root: Path, variant: str, slot: int):
    """Acquires the per-slot flock for the ENTIRE evaluation lifecycle (prime
    check + compile + solve), not just priming, and holds it until the
    caller is done with the sandbox. This makes a reused slot block until
    any previous bridge invocation against it has exited (FIX 4a), instead
    of releasing the lock right after priming and letting two evaluations
    race on the same sandbox's candidate .cpp / build dir."""
    slot_dir, sandbox = _slot_paths(task_cfg, repo_root, variant, slot)
    slot_dir.mkdir(parents=True, exist_ok=True)

    lock_path = slot_dir / ".ensure_slot.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            _ensure_slot_locked(repo_root, variant, slot, sandbox)
            yield sandbox
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Score cache: sha256(exact candidate bytes + eval-context) -> jsonl
# ---------------------------------------------------------------------------

# Bumped whenever the cache key derivation or record shape changes in a way
# that makes old entries unsafe to reuse (FIX 2: v1 keyed only on
# whitespace/comment-normalized source, scoped only by variant -- collided
# across dataset size/seed/iteration knobs and could normalize away
# semantically-significant whitespace, e.g. inside a `#define`). Records
# without a matching "v" are treated as misses, never read as hits.
CACHE_SCHEMA_VERSION = 2


def _cache_key(
    func_str: str,
    *,
    repo_root: Path,
    instances_dir: Path,
    baselines_dir: Path,
    num_instances: int,
    max_iterations: int,
    hgs_seed: int,
    smoke_instances: int,
) -> str:
    """Hashes the EXACT candidate source bytes (no comment/whitespace
    normalization -- correctness over hit rate) together with every knob
    that affects the resulting score, so identical source evaluated under a
    different dataset size/seed/iteration budget never collides."""

    def _rel(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(repo_root).as_posix()
        except ValueError:
            return resolved.as_posix()

    context = "\x1f".join([
        _rel(instances_dir),
        _rel(baselines_dir),
        str(num_instances),
        str(max_iterations),
        str(hgs_seed),
        str(smoke_instances),
    ])
    hasher = hashlib.sha256()
    hasher.update(func_str.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(context.encode("utf-8"))
    return hasher.hexdigest()


def _cache_path(repo_root: Path, task_cfg: Any, variant: str) -> Path:
    # Frozen location: scratch/hgs/cache/<VARIANT>.jsonl (sibling of the
    # per-variant slot directories under sandbox_root).
    sandbox_root = repo_root / str(_cfg_get(task_cfg, "sandbox_root", "scratch/hgs"))
    return sandbox_root / "cache" / f"{variant}.jsonl"


def _cache_lookup(cache_path: Path, key: str) -> dict | None:
    if not cache_path.is_file():
        return None
    best = None
    try:
        with open(cache_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("v") != CACHE_SCHEMA_VERSION:
                    continue  # stale/foreign schema -- never a hit
                if record.get("hash") == key:
                    best = record  # last one wins (most recent re-evaluation)
    except OSError:
        return None
    return best


def _cache_append(cache_path: Path, record: dict) -> None:
    record = {"v": CACHE_SCHEMA_VERSION, **record}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(record) + "\n")
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# evaluate_candidate: py3.10-side orchestration of the py3.12 bridge
# ---------------------------------------------------------------------------


@dataclass
class _EvalContext:
    repo_root: Path
    task_cfg: Any
    variant: str
    slot: int
    sandbox: Path
    py312: str
    instances_dir: Path
    baselines_dir: Path
    max_iterations: int
    hgs_seed: int
    eval_timeout: float
    smoke_timeout: float
    tmp_dir: Path


def _resolve_dataset_config(dataset_config: Any, repo_root: Path) -> tuple[Path, Path, Any]:
    def get(key: str, default: Any = None) -> Any:
        if isinstance(dataset_config, dict):
            return dataset_config.get(key, default)
        return getattr(dataset_config, key, default)

    instances_dir = get("instances_dir")
    baselines_dir = get("baselines_dir")
    if instances_dir is None or baselines_dir is None:
        raise ValueError(
            "hgs evaluate_candidate: dataset_config must provide 'instances_dir' and "
            f"'baselines_dir' (got {dataset_config!r})."
        )

    instances_dir = Path(instances_dir)
    if not instances_dir.is_absolute():
        instances_dir = repo_root / instances_dir
    baselines_dir = Path(baselines_dir)
    if not baselines_dir.is_absolute():
        baselines_dir = repo_root / baselines_dir

    return instances_dir, baselines_dir, get("n")


def _bridge_call(
    ctx: _EvalContext, candidate_path: Path, n_instances: int, compile_flag: bool, timeout: float,
) -> dict:
    json_out = ctx.tmp_dir / f"result_w{ctx.slot}_{os.getpid()}_{uuid.uuid4().hex}.json"
    argv = [
        ctx.py312, str(HGS_EVAL_SCRIPT),
        "--sandbox", str(ctx.sandbox),
        "--candidate", str(candidate_path),
        "--variant", ctx.variant,
        "--instances", str(ctx.instances_dir),
        "--baselines", str(ctx.baselines_dir),
        "--num-instances", str(n_instances),
        "--max-iterations", str(ctx.max_iterations),
        "--seed", str(ctx.hgs_seed),
        "--compile", "1" if compile_flag else "0",
        "--json-out", str(json_out),
    ]
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    try:
        stdout_text, _ = proc.communicate(timeout=timeout)
    except BaseException:
        # ANY exception here (not just TimeoutExpired: e.g. the worker being
        # torn down mid-call) must still reap the bridge's process group --
        # otherwise a reused slot's next evaluation races an orphaned bridge
        # still holding the sandbox's build dir (FIX 4b). SIGKILL of the
        # worker itself remains a documented residual risk: no Python code
        # runs to get here in that case.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass
        proc.wait()
        json_out.unlink(missing_ok=True)
        raise

    try:
        if not json_out.is_file():
            raise RuntimeError(
                f"hgs bridge exited {proc.returncode} without producing JSON output at {json_out}; "
                f"output tail: {(stdout_text or '')[-2000:]}"
            )
        with open(json_out, "r", encoding="utf-8") as fh:
            return json.load(fh)
    finally:
        json_out.unlink(missing_ok=True)


def _looks_like_compiler_diagnostic(stderr_tail: str) -> bool:
    lowered = (stderr_tail or "").lower()
    return any(marker in lowered for marker in _COMPILER_DIAGNOSTIC_MARKERS)


def _probe_pristine_compiles(ctx: _EvalContext, timeout: float) -> bool:
    """One-off compile-only probe used to discriminate "candidate is broken"
    from "sandbox is broken" during the recovery ladder. Returns True if the
    pristine seed compiles cleanly in ctx.sandbox."""
    pristine_text = _pristine_seed_text(ctx.repo_root)
    probe_path = ctx.tmp_dir / f"pristine_probe_w{ctx.slot}_{uuid.uuid4().hex}.cpp"
    probe_path.write_text(pristine_text, encoding="utf-8")
    try:
        result = _bridge_call(ctx, probe_path, 1, compile_flag=True, timeout=timeout)
    finally:
        probe_path.unlink(missing_ok=True)
    compile_failed = result.get("stage") == "compile" and not result.get("ok")
    return not compile_failed


def _compile_and_solve_with_recovery(
    ctx: _EvalContext, candidate_path: Path, n_instances: int, timeout: float
) -> tuple[str, dict]:
    """Runs the bridge with the recovery ladder from the plan on compile
    failure. Returns ``(status, payload)`` where status is one of:
      - "ok": payload is the full bridge result JSON.
      - "candidate_invalid": payload is {"reason": <diagnostic text>}.
      - "solve_failed": payload is the full bridge result JSON (non-compile
        stage failure, e.g. an exception while solving).
    Raises RuntimeError only if re-priming itself fails or the pristine seed
    still doesn't compile afterwards (unrecoverable infra failure, loud,
    never silently scored). A candidate that still fails to compile against
    a freshly re-primed (and therefore proven-healthy) sandbox is the
    candidate's fault, not infra's -- that maps to "candidate_invalid"
    (FIX 5), not a raise.

    Must be called with the sandbox's per-slot flock already held by the
    caller (see ``_held_slot``): the re-prime path below calls
    ``_ensure_slot_locked`` directly, not ``ensure_slot``, to avoid
    re-acquiring (and self-deadlocking on) that same lock."""
    result = _bridge_call(ctx, candidate_path, n_instances, compile_flag=True, timeout=timeout)
    if result.get("ok"):
        return "ok", result
    if result.get("stage") != "compile":
        return "solve_failed", result

    stderr_tail = result.get("compile_stderr_tail", "")
    if _looks_like_compiler_diagnostic(stderr_tail):
        return "candidate_invalid", {"reason": f"hgs candidate failed to compile:\n{stderr_tail}"}

    logging.warning(
        f"[hgs] ambiguous compile failure (no compiler diagnostic markers) for "
        f"variant={ctx.variant} slot={ctx.slot}; probing sandbox health with the pristine seed."
    )
    if _probe_pristine_compiles(ctx, timeout):
        return "candidate_invalid", {
            "reason": f"hgs candidate failed to compile (sandbox verified healthy):\n{stderr_tail}"
        }

    logging.warning(
        f"[hgs] sandbox variant={ctx.variant} slot={ctx.slot} appears corrupt "
        "(pristine seed also fails to compile); re-priming once."
    )
    shutil.rmtree(ctx.sandbox, ignore_errors=True)
    _ensure_slot_locked(ctx.repo_root, ctx.variant, ctx.slot, ctx.sandbox)

    retry = _bridge_call(ctx, candidate_path, n_instances, compile_flag=True, timeout=timeout)
    if retry.get("ok"):
        return "ok", retry
    if retry.get("stage") == "compile":
        # Re-priming above just rebuilt+installed the sandbox from the
        # pristine seed without raising, so the sandbox is proven healthy;
        # this compile failure is the candidate's fault (FIX 5).
        return "candidate_invalid", {
            "reason": (
                "hgs candidate failed to compile (sandbox re-primed and verified "
                f"healthy):\n{retry.get('compile_stderr_tail', '')}"
            )
        }
    return "solve_failed", retry


def _finish_failed(
    function_class, failed_score: float, reason: str, cache_path: Path | None = None, cache_key: str | None = None,
):
    function_class.fail_flag = 1
    function_class.score = failed_score
    function_class.true_score = failed_score
    function_class.fail_exception = str(reason)[-DIAGNOSTIC_TAIL_CHARS:]
    if cache_path is not None and cache_key is not None:
        _cache_append(cache_path, {
            "hash": cache_key,
            "score": failed_score,
            "true_score": failed_score,
            "fail_flag": 1,
            "correct_flag": 0,
            "fail_exception": function_class.fail_exception,
            "timestamp": time.time(),
        })
    return function_class


def _finish_success(function_class, result: dict, cache_path: Path | None = None, cache_key: str | None = None):
    mean_gap_percent = result.get("mean_gap_percent", 0.0)
    score = round(-mean_gap_percent, 3)
    function_class.score = score
    function_class.true_score = score
    function_class.fail_flag = 0
    function_class.correct_flag = 1
    if cache_path is not None and cache_key is not None:
        _cache_append(cache_path, {
            "hash": cache_key,
            "score": score,
            "true_score": score,
            "fail_flag": 0,
            "correct_flag": 1,
            "fail_exception": "",
            "n_instances": result.get("n_instances"),
            "n_feasible": result.get("n_feasible"),
            "mean_gap_percent": mean_gap_percent,
            "timestamp": time.time(),
        })
    return function_class


def evaluate_candidate(cfg: Any, dataset_config: Any, function_class):
    """``evaluate_func`` for HGS/Crex tasks. Mirrors the fields/flags set by
    ``src/packing/evaluate/tsp/task_tsp.py::evaluate_func`` (score,
    true_score, fail_flag, fail_exception on failure; +correct_flag on
    success) but with a py3.12 subprocess bridge in place of an in-process
    exec()."""
    task_cfg = cfg.task
    repo_root = _repo_root()

    variant = _cfg_get(task_cfg, "variant", "CVRP")
    py312 = _cfg_get(task_cfg, "py312", DEFAULT_PY312)
    max_iterations = int(_cfg_get(task_cfg, "max_iterations", 1000))
    hgs_seed = int(_cfg_get(task_cfg, "hgs_seed", 0))
    eval_timeout = float(_cfg_get(task_cfg, "eval_timeout", 480))
    smoke_timeout = float(_cfg_get(task_cfg, "smoke_timeout", eval_timeout))
    smoke_instances = int(_cfg_get(task_cfg, "smoke_instances", 0) or 0)
    failed_score = _cfg_get(task_cfg, "failed_score", -20000)
    num_instances = int(_cfg_get(task_cfg, "num_eval_instances", 50))

    func_str = function_class.function_str or ""
    if not func_str.strip():
        return _finish_failed(
            function_class, failed_score,
            "evaluate_candidate: function_str is empty (extraction failed upstream).",
        )

    slot = max(int(getattr(function_class.eval, "idx_process", 0) or 0), 0)

    instances_dir, baselines_dir, _n = _resolve_dataset_config(dataset_config, repo_root)

    cache_path = _cache_path(repo_root, task_cfg, variant)
    cache_key = _cache_key(
        func_str,
        repo_root=repo_root,
        instances_dir=instances_dir,
        baselines_dir=baselines_dir,
        num_instances=num_instances,
        max_iterations=max_iterations,
        hgs_seed=hgs_seed,
        smoke_instances=smoke_instances,
    )
    cached = _cache_lookup(cache_path, cache_key)
    if cached is not None:
        logging.info(
            f"[hgs] score cache HIT variant={variant} slot={slot} key={cache_key[:12]} "
            f"score={cached.get('score')}"
        )
        function_class.score = cached["score"]
        function_class.true_score = cached["true_score"]
        function_class.fail_flag = cached["fail_flag"]
        function_class.fail_exception = cached.get("fail_exception", "")
        if cached.get("fail_flag") == 0:
            function_class.correct_flag = cached.get("correct_flag", 1)
        return function_class

    tmp_dir = repo_root / "scratch" / "hgs" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = tmp_dir / f"candidate_w{slot}_{os.getpid()}_{uuid.uuid4().hex}.cpp"
    candidate_path.write_text(func_str, encoding="utf-8")

    try:
        # Infra failures while priming are raised loudly, never silently
        # scored. The per-slot flock is held for the whole prime+compile+
        # solve lifecycle (FIX 4a) so a reused slot blocks until any
        # previous bridge invocation against it has fully exited.
        with _held_slot(task_cfg, repo_root, variant, slot) as sandbox:
            ctx = _EvalContext(
                repo_root=repo_root, task_cfg=task_cfg, variant=variant, slot=slot, sandbox=sandbox,
                py312=py312, instances_dir=instances_dir, baselines_dir=baselines_dir,
                max_iterations=max_iterations, hgs_seed=hgs_seed, eval_timeout=eval_timeout,
                smoke_timeout=smoke_timeout, tmp_dir=tmp_dir,
            )

            if smoke_instances > 0:
                try:
                    status, payload = _compile_and_solve_with_recovery(
                        ctx, candidate_path, smoke_instances, smoke_timeout,
                    )
                except subprocess.TimeoutExpired:
                    return _finish_failed(
                        function_class, failed_score,
                        f"hgs smoke stage timed out after {smoke_timeout}s (slot={slot})",
                    )
                if status == "candidate_invalid":
                    return _finish_failed(function_class, failed_score, payload["reason"], cache_path, cache_key)
                if status == "solve_failed":
                    return _finish_failed(
                        function_class, failed_score, f"hgs smoke solve stage failed: {payload}", cache_path, cache_key,
                    )
                # status == "ok": the smoke instance(s) compiled and solved
                # without a bridge/compile failure. Per FIX 6, an infeasible
                # smoke result is NOT a rejection reason -- the bridge caps
                # infeasible gap at +100% already, so proceed to the full
                # run and let that score speak for itself.

            try:
                status, payload = _compile_and_solve_with_recovery(ctx, candidate_path, num_instances, eval_timeout)
            except subprocess.TimeoutExpired:
                return _finish_failed(
                    function_class, failed_score, f"hgs evaluation timed out after {eval_timeout}s (slot={slot})",
                )

            if status == "candidate_invalid":
                return _finish_failed(function_class, failed_score, payload["reason"], cache_path, cache_key)
            if status == "solve_failed":
                return _finish_failed(
                    function_class, failed_score, f"hgs solve stage failed: {payload}", cache_path, cache_key,
                )

            return _finish_success(function_class, payload, cache_path, cache_key)
    finally:
        candidate_path.unlink(missing_ok=True)
