#!/usr/bin/env bash
#SBATCH --job-name=evotune_fp_default
#SBATCH --output=/mnt/beegfs/home/guo/these_guo/2026/EvoTune/outputs/%x_%j.out
#SBATCH --error=/mnt/beegfs/home/guo/these_guo/2026/EvoTune/outputs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

set -euo pipefail

REPO_DIR="/mnt/beegfs/home/guo/these_guo/2026/EvoTune"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src

PREFIX="${PREFIX:-fp_default_granite_main}"
MODEL="${MODEL:-granite}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
PROJECT="${PROJECT:-EvoTune}"
ENTITY="${ENTITY:-zeio99guo-institut-polytechnique-de-paris}"

cd "${REPO_DIR}"
mkdir -p outputs

source .venv/bin/activate

srun .venv/bin/python src/experiments/main.py \
  task=flatpack \
  model="${MODEL}" \
  train=dpo \
  cluster=example \
  gpu_nums="${GPU_ID}" \
  prefix="${PREFIX}" \
  seed="${SEED}" \
  wandb=1 \
  project="${PROJECT}" \
  entity="${ENTITY}" \
  use_vllm=0 \
  use_tgi=0
