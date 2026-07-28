# HGS/Crex Problems in EvoTune — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** EvoTune evolves the C++ SREX ("Crex") crossover `pyvrp_rep/pyvrp/cpp/crossover/selective_route_exchange.cpp`, ReEvo-style, for six VRP variants (CVRP first), with incremental meson compilation and evaluation on `utils/data/dataset/generated/<VARIANT>/51`, LLM = local Qwen3 via vLLM.

**Architecture:** Raw C++ full-file is the stored program (`function_str`); an opt-in `parse_llm_output` task hook extracts the last ```cpp fence (bypassing the python line-scanner which truncates at `return `). Evaluation bridges from the py3.10 `evotune` worker to `/home/zguo/miniconda/envs/rlft/bin/python3.12` (only interpreter that can import the vendored cp312 pyvrp), which compiles incrementally in a persistent per-worker sandbox and solves instances. Score = −mean(gap% vs pyvrp baseline `.sol`), rounded to 3 decimals.

**Tech stack:** Hydra 1.3, vLLM 0.8.5.post1 (evotune env), meson/ninja (rlft env), pybind11, PyVRP 0.12.2 (vendored).

## Global Constraints

- Main loop interpreter: `/home/zguo/miniconda/envs/evotune/bin/python` (py3.10). Bridge interpreter: `/home/zguo/miniconda/envs/rlft/bin/python3.12`.
- Modify existing EvoTune files as little as possible. Whole-plan budget for edits to existing files: `src/packing/evaluate/registry.py` (~6 lines), `src/packing/model/model.py` (3 small spots), `src/packing/utils/functions.py` (2 guards). Everything else is additive.
- Everything generated stays inside `/home/zguo/Coding/baseline/EvoTune`.
- NEVER commit: `utils/model/` (28G weights + **leaked wandb API key** in `utils/model/*/wandb_run.json`), `utils/data/Operator/` (584M), any `build*/`, `*.so`, `scratch/`.
- Git identity `zihao-guo <zeio99guo@gmail.com>`; remote `https://github.com/zihao-guo/EvoTune.git` (already set). First push: force (user-authorized).
- Naming mirrors ReEvo: hydra `task=cvrp_hgs` … `vrptw_hgs`.

## Design provenance (Opus ⊕ Sonnet synthesis, Fable-adjudicated)

- D1: raw C++ stored (Opus — prompt fidelity, ReEvo-proven format) via registry-level opt-in hook (Sonnet — no content-sniffing in shared utils). Both verified the stock scanner truncates C++ at `return `.
- D2: rlft py3.12 bridge subprocess, one script does compile+solve, JSON out (Opus), `start_new_session=True` + killpg on timeout.
- D3: per-worker persistent sandboxes under `scratch/hgs/<VARIANT>/w<slot>/pyvrp_rep`; **never reuse `build-evohgs-rlft-py312`** (both independently confirmed its ninja/meson state points at `/home/zguo/Coding/evohgs/rlft/pyvrp_rep`); `-Dwerror=false`; keep LTO; measured hot path ≈ 17.5 s (3 ninja edges).
- D4: local-dir passthrough in `get_full_model_name`; `vllm_extra_args` passthrough; thinking disabled per-request via `chat_template_kwargs: {"enable_thinking": false}` (works with either bundled template variant).
- D5: `src/packing/evaluate/hgs/` module; seed = pristine original (NOT ReEvo's pre-evolved cvrp seed) for all six variants.
- D6: `.gitignore` is commit #1; pyvrp_rep source committed (no build dirs); dataset (99M) committed.

---

### Task 1: `.gitignore` + safety commit

**Files:** Create `/home/zguo/Coding/baseline/EvoTune/.gitignore`

- [ ] Write `.gitignore`:

```gitignore
# heavy / secret-bearing artifacts — NEVER track
/utils/model/
/utils/data/Operator/
**/wandb_run.json
wandb/

# build artifacts
/pyvrp_rep/build*/
*.so
*.o
*.a
__pycache__/
*.py[cod]

# runtime sandboxes & outputs
/scratch/
/out/
/outputs/
/multirun/
flag_resume.txt
```

- [ ] Verify: `cd /home/zguo/Coding/baseline/EvoTune && git add -n . 2>/dev/null | grep -E 'utils/model|wandb_run|\.so$|build-evohgs' ; echo "exit=$?"` → must print nothing / exit 1.
- [ ] Verify staged size sanity: `git add -n . | wc -l` and `du -sh utils/data/dataset` (~99M expected; utils/model must not appear).
- [ ] Commit `.gitignore` ALONE: `git add .gitignore && git commit -m "chore: add .gitignore (exclude model weights, secrets, build artifacts)"`.
- [ ] Commit vendored source: `git add pyvrp_rep CLAUDE.md .claude docs && git commit -m "chore: vendor pyvrp_rep 0.12.2 source; add workflow docs"`. Verify with `git show --stat HEAD | tail -5` that no `.so`/build files got in.

### Task 2: C++ extraction module + minimal framework hooks

**Files:**
- Create: `src/packing/evaluate/hgs/__init__.py` (empty), `src/packing/evaluate/hgs/extraction.py`
- Modify: `src/packing/evaluate/registry.py`, `src/packing/model/model.py` (generate_from_server only), `src/packing/utils/functions.py`

**Interfaces (frozen — later tasks rely on these exact names):**
- `extraction.extract_cpp_source(text: str) -> str` — last fenced ```cpp/c++/cxx/cc block; must contain all three markers `#include "selective_route_exchange.h"`, `selectiveRouteExchange`, `ProblemData const &data`; returns `""` AND logs a warning on failure (model.py silently drops empties — this log is the only visibility).
- `extraction.parse_llm_output(text: str) -> tuple[str, str]` — returns `(cpp_or_empty, "")`.

- [ ] Implement `extraction.py` (port regex/validators from `/home/zguo/Coding/baseline/ReEvo/problems/hgs_share/code_extraction.py::extract_cpp_code/_looks_like_cpp_source`).
- [ ] `registry.py`: `TASK_REGISTRY.register(...)` gains optional kwarg `parse_llm_output=None` stored alongside the other entries; accessor unchanged pattern. Purely additive; existing 3 tasks unaffected.
- [ ] `model.py::generate_from_server` (~line 370 where `extract_functions`/`extract_imports` run): if the registered task dict has non-None `parse_llm_output`, use it for `(function_str, imports_str)`; else existing path. Read the surrounding code first; keep raw LLM text flowing into chat/DPO data unmodified.
- [ ] `functions.py::function_to_string`: first lines `raw = getattr(func, "_raw_source", None); if raw is not None: return raw` (lets a stub seed callable carry C++; avoids editing main.py:197).
- [ ] `functions.py::separate_imports_from_func`: if no `def ` line found, `return "", func_str` (safety for offline eval.py path).
- [ ] Unit check (no LLM): craft a fake reply with prose + a decoy ```python fence + valid ```cpp block containing 4 `return` lines; assert extract returns the full block unmangled; assert a reply without markers returns `""`. Run with `PYTHONPATH=src /home/zguo/miniconda/envs/evotune/bin/python -c ...`.
- [ ] Regression: `PYTHONPATH=src /home/zguo/miniconda/envs/evotune/bin/python -c "from packing.evaluate import import_all_tasks; import_all_tasks()"` still imports tsp/bin/flatpack cleanly.
- [ ] Commit: `feat: opt-in raw-source LLM output parsing hook + C++ extractor`.

### Task 3: py3.12 bridge — compile + solve + score

**Files:** Create `src/packing/evaluate/hgs/hgs_eval_py312.py` (standalone; stdlib + pyvrp only; must NOT import `packing`; must NOT be named `task_*.py`).

**Interface (frozen):**
```
/home/zguo/miniconda/envs/rlft/bin/python3.12 src/packing/evaluate/hgs/hgs_eval_py312.py \
  --sandbox <abs path to sandbox pyvrp_rep root> --candidate <cpp path> --variant CVRP \
  --instances <dir with *.vrp> --baselines <dir with *.sol> \
  --num-instances 50 --max-iterations 1000 --seed 0 --compile 1 --json-out <path>
```
JSON out: `{"ok": bool, "stage": "compile"|"solve"|"done", "compile_stderr_tail": str, "n_instances": int, "n_feasible": int, "mean_gap_percent": float, "per_instance": [{"name","obj","baseline","gap_percent","feasible"}], "wall_seconds": float}`.

- [ ] Port from `/home/zguo/Coding/baseline/ReEvo/problems/hgs_share/eval.py`: `parse_vrp_instance`, `build_manual_model` (generic across variants via TYPE header sets), `calculate_route_cost`, solve via `Model.solve(stop=MaxIterations(max_iterations), seed=..., display=False)`. Drop: ReEvo's temp-sandbox cloning, argv protocol, stdout-line protocol, plagiarism (handled py3.10-side).
- [ ] Compile step (when `--compile 1`): write candidate into `<sandbox>/pyvrp/cpp/crossover/selective_route_exchange.cpp`, then `meson compile -C <sandbox>/build` + `meson install -C <sandbox>/build` with `PATH=/home/zguo/miniconda/envs/rlft/bin:$PATH` (meson.build calls bare `python`). Nonzero → JSON `{"ok": false, "stage": "compile", "compile_stderr_tail": <last 4000 chars>}`.
- [ ] Solve step: `sys.path.insert(0, sandbox)`; per instance gap% = `(obj/baseline_obj - 1)*100`; **infeasible → gap capped at +100.0** (deviation from ReEvo's huge penalty — keeps scores inside softmax-friendly range); mean over instances.
- [ ] Verify with pristine seed on 3 instances (expect |mean_gap| < 2 since baselines came from pyvrp): run the exact CLI above with `--num-instances 3` against a hand-primed sandbox (Task 4 provides priming; for this task's test, prime one sandbox manually per Task 4 step 1 or coordinate).
- [ ] Commit: `feat: py3.12 pyvrp bridge (incremental compile + solve + JSON scoring)`.

### Task 4: worker sandboxes + priming + py3.10 evaluation glue

**Files:** Create `src/packing/evaluate/hgs/hgs_common.py`, `scripts/hgs_prime_sandboxes.py`.

**Interfaces (frozen):**
- `hgs_common.ensure_slot(task_cfg, variant: str, slot: int) -> Path` — sandbox at `<repo>/scratch/hgs/<VARIANT>/w<slot>/pyvrp_rep`; `fcntl.flock` guarded; if missing/corrupt: copytree `pyvrp_rep` (exclude `build*`, `.git`, `docs`, `tests`, `__pycache__`) then meson setup+build via `pyvrp_rep`-style commands (`meson setup build --buildtype release -Dwerror=false -Dpython.platlibdir=<sandbox>` with rlft PATH, then compile+install). Restore pristine `.cpp` before returning.
- `hgs_common.evaluate_candidate(cfg, dataset_config, function_class) -> FunctionClass` — slot = `max(getattr(function_class.eval, "idx_process", 0), 0)`; writes `function_str` to a temp `.cpp`; calls the Task-3 bridge with `subprocess.run(..., start_new_session=True, timeout=task_cfg.eval_timeout)`; on `TimeoutExpired` → `os.killpg`; parses JSON; sets scores exactly per the contract in `src/packing/evaluate/tsp/task_tsp.py:345`'s evaluate_func (read it; same fields/flags). Score = `round(-mean_gap_percent, 3)`; compile fail/extraction-empty/timeout → `cfg.task.failed_score`, fail flag set.
- Recovery ladder on compile failure: (1) compiler diagnostics in stderr → candidate invalid → failed_score; (2) else recompile pristine seed — if THAT fails, `rm -rf` sandbox, re-prime once, retry candidate; still broken → raise (loud infra failure, not silent zero-scores).
- Two-stage screen: if `task_cfg.smoke_instances > 0`, first bridge call with `--num-instances <smoke_instances>`; only on feasible result run the full set.
- Score cache: sha256 of whitespace/comment-normalized C++ → jsonl at `scratch/hgs/cache/<VARIANT>.jsonl`; hit → return cached score (LLMs regenerate near-identical files; biggest throughput win).

- [ ] Implement both files.
- [ ] `scripts/hgs_prime_sandboxes.py --variant CVRP --slots 0 1 2` primes in parallel (calls `ensure_slot`). Run it; expect ~40–60 s per cold slot.
- [ ] Incremental timing proof: `touch` the `.cpp` in slot 0, `time meson compile -C scratch/hgs/CVRP/w0/pyvrp_rep/build` → seconds (≈17 s), ninja log shows ~3 edges.
- [ ] Bridge round-trip: pristine seed through `evaluate_candidate`-equivalent call on 3 instances → sane JSON, |gap| small.
- [ ] Commit: `feat: persistent per-worker HGS sandboxes with incremental rebuild + eval glue`.

### Task 5: task registration + prompts + task configs (CVRP)

**Files:** Create `src/packing/evaluate/hgs/task_hgs_shared.py` (registration factory; NOT `task_*.py`-globbed logic duplication — it holds `register_variant(task_name, variant, prompt_dir)`), `src/packing/evaluate/hgs/task_cvrp_hgs.py` (3 lines calling the factory), `src/packing/evaluate/hgs/seed/selective_route_exchange.original.cpp` (copied; verify byte-identical to `pyvrp_rep/pyvrp/cpp/crossover/selective_route_exchange.cpp` AND to `utils/data/Operator/selective_route_exchange.original.cpp`), `src/packing/evaluate/hgs/prompts/{common/*.txt, cvrp/*.txt}` (ported from ReEvo `prompts/hgs_common/` + `prompts/HGS_Crex/cvrp_hgs/{func_desc,func_signature,external_knowledge}.txt`; full-file C++17 output contract, last-```cpp-fence rule), `configs/task/cvrp_hgs.yaml`.

- [ ] Registry pieces per `src/packing/evaluate/README.md` contract: `generate_input(cfg, set)` returns picklable dict {instances_dir, baselines_dir, n, …} for `train`(size 51)/`trainperturbedset`(101)/`testset`(201); `get_initial_func(cfg)` returns seed stub with `_raw_source` = pristine cpp (+ the same string); `evaluate_func` = `hgs_common.evaluate_candidate`; `system_prompt`/`append_prompt` assembled from prompt files; `parse_llm_output=extraction.parse_llm_output`.
- [ ] `configs/task/cvrp_hgs.yaml`: `task_name: "cvrp_hgs"`, `function_str_to_extract: "selectiveRouteExchange"`, `variant: "CVRP"`, `dataset_root: "utils/data/dataset/generated"`, `baseline_root: "utils/data/dataset/opt"` (+`/pyvrp` leaf), sizes `51/101/201`, `num_eval_instances: 50`, `smoke_instances: 1`, `max_iterations: 1000`, `hgs_seed: 0`, `compile_timeout: 300`, `eval_timeout: 900`, `timeout_period: 1200`, `failed_score: -20000`, `mem_limit_gb: 16`, `py312: "/home/zguo/miniconda/envs/rlft/bin/python3.12"`, `sandbox_root: "scratch/hgs"`, `programdatabaseConfig: {temp: 0.5}` (scores live in ~[−5, 0]; tsp's 10.0 flattens softmax).
- [ ] All paths resolved repo-root-relative (hydra may chdir — resolve from `Path(__file__).parents[k]`, mirror how task_tsp handles data paths).
- [ ] Verify registration: `PYTHONPATH=src …python -c "from packing.evaluate import import_all_tasks, registry; import_all_tasks(); print('cvrp_hgs' in registry.TASK_REGISTRY.tasks or registry.TASK_REGISTRY)"` (adapt to actual registry API).
- [ ] Verify hydra composes: `PYTHONPATH=src …python -c` hydra compose or a `--cfg job` dry call with `task=cvrp_hgs`.
- [ ] Commit: `feat: cvrp_hgs task (ReEvo-style HGS/Crex problem) with prompts and config`.

### Task 6: model wiring (Qwen3 local, vLLM)

**Files:** Modify `src/packing/model/model.py` (2 spots); Create `configs/model/qwen3_14b.yaml`, `configs/model/qwen3_06b.yaml`.

- [ ] `get_full_model_name::get_name`: before the final `else/raise`, add local-dir passthrough: resolve `<repo_root>/utils/model/<name>`; if `os.path.isdir` → return that absolute path (repo root from `Path(__file__).resolve().parents[3]` — verify depth).
- [ ] `start_vllm_server`: append `cfg.model.get("vllm_extra_args", [])`; `make_vllm_request`: include `"chat_template_kwargs": dict(cfg.model.chat_template_kwargs)` in payload when the key exists in the model config (works for both bundled Qwen3 template variants; granite/llama32/phi payloads unchanged).
- [ ] `configs/model/qwen3_14b.yaml`: `model_name: "Qwen3-14B-Instruct"` (matches dir name), `temperature: 0.8`, `topk: 20`, `topp: 0.9`, `max_tokens: 6144`, `chat_template_kwargs: {enable_thinking: false}`, `vllm_extra_args: ["--max-model-len","16384","--gpu-memory-utilization","0.90","--max-num-seqs","8","--enable-prefix-caching"]`.
- [ ] `configs/model/qwen3_06b.yaml`: `model_name: "Qwen3-0.6B"`, `max_tokens: 4096`, same kwargs, `vllm_extra_args: ["--max-model-len","16384","--gpu-memory-utilization","0.35"]`.
- [ ] Verify: `vllm serve` manually with 0.6B + one `curl` chat completion with `chat_template_kwargs` → reply contains no `<think>` content; then `pkill -f "vllm serve"`.
- [ ] Commit: `feat: local Qwen3 model wiring (dir passthrough, vllm extra args, no-think)`.

### Task 7: end-to-end smoke (0.6B), then short 14B run

- [ ] Preflight: slots 0–2 primed; `nvidia-smi` free; `ls utils/model/Qwen3-14B-Instruct/model-0000*.safetensors | wc -l` = 8.
- [ ] Smoke: `cd /home/zguo/Coding/baseline/EvoTune && PYTHONPATH=src /home/zguo/miniconda/envs/evotune/bin/python src/experiments/main.py task=cvrp_hgs model=qwen3_06b train=none +wandb=0 num_rounds=1 num_cont_rounds=1 num_outputs_per_prompt=2 num_workers=2 task.num_eval_instances=3 task.max_iterations=200 prefix=smoke1 seed=0` (`+wandb=0` with leading `+` is mandatory — key absent from yaml; fresh `prefix` every rerun — resume flag advances rounds silently).
- [ ] Success criteria: vLLM serves; ≥1 candidate extracted (check extraction warnings); seed scored ≈ 0; incremental compiles ≈ seconds; run exits cleanly. `pkill -f "vllm serve"` after (main.py never stops it).
- [ ] Debug loop until green (use systematic-debugging; check `out/`/hydra run dir logs).
- [ ] 14B short run: same but `model=qwen3_14b num_cont_rounds=2 num_outputs_per_prompt=4 num_workers=4 task.num_eval_instances=10 prefix=q14b_smoke` → confirm ≥1 valid candidate compiles and scores.
- [ ] Commit any fixes: `fix: smoke-run corrections for cvrp_hgs pipeline`.

### Task 8: five remaining variants

**Files:** Create `src/packing/evaluate/hgs/task_{ovrp,ovrptw,vrpb,vrpl,vrptw}_hgs.py` (3 lines each), `configs/task/{ovrp,ovrptw,vrpb,vrpl,vrptw}_hgs.yaml` (clone cvrp yaml, change `task_name`/`variant`), `prompts/{ovrp,ovrptw,vrpb,vrpl,vrptw}/*.txt` (port each variant's `external_knowledge.txt`/`func_desc.txt` from ReEvo; the five non-CVRP ReEvo seeds are identical — we use the pristine seed everywhere anyway).
- [ ] Registration + hydra compose check for all six.
- [ ] Bridge sanity per variant: pristine seed, 2 instances each from `generated/<V>/51` vs `opt/<V>/51/pyvrp` (validates `build_manual_model` TYPE handling: open routes, backhaul, route limit, time windows).
- [ ] Commit: `feat: add ovrp/ovrptw/vrpb/vrpl/vrptw HGS problems`.

### Task 9: QA + push

- [ ] Codex ultra QA over the full diff (challenge: over-modification of existing code, correctness of scoring, secret leakage in staged files).
- [ ] Address findings; re-verify smoke.
- [ ] `git log --stat` review: confirm no weights/secrets/build artifacts in ANY commit.
- [ ] Push: `git push --force origin main` (first push overrides remote per user instruction).

## Top risks (carry into execution)

1. Throughput ~170 s/candidate at full settings → mitigations already in plan: score cache, `smoke_instances=1` screen, tunable `num_eval_instances`/`max_iterations`.
2. Silent run death: `make_vllm_request` has no HTTP timeout and 400s kill the run via termination_event → keep prompt ≤ ~13k tokens (max-model-len 16384, functions_per_prompt=2); log every empty extraction.
3. Sandbox corruption from SIGKILL'd workers → killpg + flock + pristine-recompile discriminator + re-prime ladder.
