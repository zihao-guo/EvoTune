#!/usr/bin/env bash
set -euo pipefail

# run_hgs_evolution.sh
#
# Reproducible launcher for HGS/Crex evolution runs (task=cvrp_hgs).
#
# This script encodes three launch-environment requirements discovered on
# 2026-07-29 while getting evolution runs working reliably on this shared
# machine:
#
#   (a) /home/zguo/miniconda/envs/evotune/bin must be on PATH, because
#       start_vllm_server invokes the bare `vllm` command (not a full path),
#       and that command only exists inside the evotune conda env's bin dir.
#
#   (b) VLLM_USE_V1=0 is mandatory. The vLLM V1 engine unconditionally
#       imports flashinfer.sampling, which crashes with an OSError in this
#       environment (torch 2.6 pre-cxx11 ABI vs. the cxx11-ABI
#       torch_c_dlpack_ext .so). vLLM's import-guard only catches
#       ImportError, not OSError, so the crash is not swallowed and the
#       server fails to start unless the legacy (V0) engine is forced.
#
#   (c) The default vLLM port 8080 may already be occupied by other users
#       on this shared machine, so we default to a different base port
#       (+vllm_base_port=8181, overridable via VLLM_BASE_PORT) and
#       pre-check that it is free before launching.
#
# Also note: main.py never stops the vLLM server it starts. After the run
# finishes (or is interrupted), find its PID and `kill` that exact PID by
# hand. Do NOT `pkill -f main.py` — that pattern can also match wrapper
# shells (e.g. this script, job schedulers) and kill unrelated processes.
#
# Usage:
#   scripts/run_hgs_evolution.sh <prefix> [extra hydra overrides...]
#
# <prefix> is required and is passed through as prefix=<prefix>. Any
# additional arguments are appended after the built-in hydra overrides, so
# they win (hydra applies later overrides on top of earlier ones) and can
# be used to change task=, model=, sizes, etc.

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_hgs_evolution.sh <prefix> [extra hydra overrides...]" >&2
  exit 1
fi

PREFIX="$1"

# Resolve repo root as the script's parent-of-parent directory and cd there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [[ -d "out/logs/${PREFIX}" ]]; then
  echo "Refusing to run: out/logs/${PREFIX} already exists." >&2
  echo "A stale flag_resume.txt in that directory silently advances rounds instead of starting fresh." >&2
  echo "Pick a new, unused prefix for this run." >&2
  exit 1
fi

PORT="${VLLM_BASE_PORT:-8181}"

if ss -tln | awk '{print $4}' | grep -qE "[.:]${PORT}$"; then
  echo "Refusing to run: port ${PORT} is already listening (likely another user's vLLM server)." >&2
  echo "Pick another port via VLLM_BASE_PORT=<port> scripts/run_hgs_evolution.sh ${PREFIX} ..." >&2
  exit 1
fi

export PATH=/home/zguo/miniconda/envs/evotune/bin:$PATH
export VLLM_USE_V1=0

PYTHONPATH=src exec /home/zguo/miniconda/envs/evotune/bin/python src/experiments/main.py \
  task=cvrp_hgs \
  model=qwen3_14b \
  train=none \
  +wandb=0 \
  +vllm_base_port="${PORT}" \
  num_rounds=1 \
  num_cont_rounds=12 \
  num_outputs_per_prompt=4 \
  num_workers=4 \
  seed=0 \
  prefix="${PREFIX}" \
  "${@:2}"
