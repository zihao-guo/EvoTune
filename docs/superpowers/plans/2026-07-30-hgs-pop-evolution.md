# HGS/Pop Operator in EvoTune — Implementation Plan

> Follows the proven Crex architecture (see 2026-07-28-hgs-crex-evolution.md). CVRP first, per user instruction. Executed by fast-worker under Fable orchestration; shared-file edits gated on dpo800 completion.

**Goal:** EvoTune evolves PyVRP's HGS population parent-selection, ReEvo-style (`prompt_family=HGS_Pop`): the candidate is a complete Python `pyvrp/Population.py` (class `Population`, methods `select` / `tournament` / `_tournament` preserved), evaluated on `utils/data/dataset/generated/CVRP/51` against pyvrp baselines, LLM = local Qwen3-14B.

**Key difference vs Crex:** pure-Python target — NO meson compile. Sandbox slots are primed once (compiled `.so` already installed); each eval just writes `pyvrp/Population.py`, imports pyvrp from the sandbox, solves. Eval per candidate ≈ solve time only.

## Design decisions (Fable, from Explore port-map)

- D1: Parametrize, don't duplicate. `hgs_common.py` + `hgs_eval_py312.py` gain optional task-cfg-driven knobs with defaults == current Crex behavior: `candidate_relative_path` (default `pyvrp/cpp/crossover/selective_route_exchange.cpp`; pop: `pyvrp/Population.py`), `candidate_suffix` (`.cpp`/`.py`), `needs_compile` (1/0), `seed_filename` (default `selective_route_exchange.original.cpp`; pop: `Population.original.py`), `sandbox_variant` (default = `variant`; pop: `CVRP_POP` — separate slots so restore-pristine logic stays single-file per sandbox family).
- D2: New additive module `extraction_pop.py`: last-resort raw fallback like ReEvo; scan ```python fences in REVERSE; validate via AST (parses + top-level `class Population` + method names ⊇ {select, tournament, _tournament}); returns "" + warning log on failure. Registered per-task via existing `parse_llm_output` hook.
- D3: New additive factory `hgs_pop_factory.py` + 3-line `task_cvrp_hgs_pop.py`; task name `cvrp_hgs_pop`; hydra config `configs/task/cvrp_hgs_pop.yaml` (clone of cvrp_hgs.yaml + the D1 knobs + `programdatabaseConfig.temp: 0.5`; smaller `eval_timeout: 360` since no compile; `smoke_timeout: 120`).
- D4: Prompts: port `ReEvo/prompts/HGS_Pop/cvrp_hgs/{func_desc,func_signature,external_knowledge,seed_func}.txt` into `src/packing/evaluate/hgs/prompts/pop_common/` + `prompts/pop_cvrp/`; system/output-contract wording adapted per ReEvo's `_adapt_common_prompts_for_hgs_pop` ("complete Population.py file", single ```python fence, class name exactly `Population`, no third-party imports).
- D5: Seed = `src/packing/evaluate/hgs/seed/Population.original.py`, byte-copied from `utils/data/Operator/Population.original.py` (verify md5 4416e214e3b6d85d0c9224e9744417cc, byte-identical to `pyvrp_rep/pyvrp/Population.py`).
- D6: Bridge: `hgs_eval_py312.py` gains `--candidate-kind py|cpp` (default cpp) — py: write file, skip meson entirely, purge cached `pyvrp` modules before import (fresh `Population` per eval in the persistent-interpreter case is N/A — bridge is a fresh process per call, so a plain import suffices).
- D7: Scoring identical to Crex (mean gap% vs pyvrp .sol, infeasible capped +100, score = −gap, cache keyed by normalized source — extend cache key with operator/candidate_relative_path to avoid cross-operator collisions).

## Gating

- Phase A (safe during dpo800): plan doc, prompt assets, seed copy — pure new data files, no imports.
- Phase B (ONLY after dpo800 exits): extraction_pop.py, hgs_pop_factory.py, task file, config, hgs_common/hgs_eval_py312 parametrization, sandbox prime for CVRP_POP slots 0-3, registration + hydra compose checks, commit.
- Phase C: pristine-seed bridge sanity (3 instances, |gap| small), 0-compile timing check, then CVRP pop smoke: `task=cvrp_hgs_pop model=qwen3_14b train=none +wandb=0 +vllm_base_port=8181 num_rounds=1 num_cont_rounds=3 num_outputs_per_prompt=4 num_workers=2 task.num_eval_instances=3 prefix=popsmoke1` (12 candidates). Then real run `prefix=pop1`: `num_rounds=1 num_cont_rounds=12 num_outputs_per_prompt=4 num_workers=4` full 50-instance eval (48 candidates, evo4-scale).
- Phase D: codex ultra QA over the diff; push; report.

## Risks

1. Python candidate can hang the solver (infinite loop in select) → bridge already bounded by eval_timeout + killpg; no compile-stage discriminator needed.
2. Candidate imports third-party modules → ImportError inside bridge → returncode≠0 no JSON → loud raise. Mitigate: extraction rejects `import` of non-allowlisted modules? Keep ReEvo behavior (prompt-level ban only); bridge wraps candidate import in try/except → JSON `{"ok": false, "stage": "solve"}` (candidate's fault, failed_score).
3. Cross-operator cache collision → D7 cache-key extension.
4. Crex regression risk from D1 parametrization → defaults preserve behavior; regression check: cvrp_hgs registration + a cached-seed bridge call before any pop smoke.
