# EvoTune TSP Train / Eval Notes

## 1. Search training command

The main search loop uses `src/experiments/main.py`. Using repository defaults for the core training parameters, a standard launch command is:

```bash
PYTHONPATH=src .venv/bin/python src/experiments/main.py \
  task=tsp \
  model=granite \
  train=dpo \
  cluster=example \
  gpu_nums=0 \
  prefix=wandb_smoketest \
  seed=0 \
  wandb=1 \
  project=EvoTune \
  entity=zeio99guo-institut-polytechnique-de-paris \
  use_vllm=0 \
  use_tgi=0
```

This command drives the full EvoTune loop:

- sample candidate `heuristics()` programs with the current LLM
- evaluate them on the TSP train set
- update the program database and DPO buffer
- periodically finetune the LLM

## 2. Smoke test command

For a quick startup check, use the following one-round smoke test command:

```bash
PYTHONPATH=src .venv/bin/python src/experiments/main.py \
  task=tsp \
  model=granite \
  train=dpo \
  cluster=example \
  gpu_nums=0 \
  prefix=tsp_granite_smoketest \
  seed=0 \
  wandb=1 \
  project=EvoTune \
  entity=zeio99guo-institut-polytechnique-de-paris \
  use_vllm=0 \
  use_tgi=0 \
  num_rounds=1 \
  num_cont_rounds=1
```

This command is only for checking that the pipeline can start correctly. It typically reaches model loading, prompt generation, worker startup, one round of sampling/evaluation, and log saving, but it does not reach DPO finetuning.

## 3. Direct finetune command

If DPO finetuning is resumed directly from an existing logs directory, use:

```bash
PYTHONPATH=src .venv/bin/accelerate launch \
  --config_file ./configs/accelerate_config/1gpu_0.yaml \
  ./src/packing/train/train.py \
  --logs_dir out/logs/wandb_smoketest/tsp_tsp_granite_dpo_0 \
  --model_name granite \
  --full_model_name ibm-granite/granite-3.1-2b-instruct \
  --model_adapter_dir out/logs/wandb_smoketest/tsp_tsp_granite_dpo_0/model_adapter_granite \
  --round_num 100 \
  --dpo_threshold -336.406
```

This command does not continue search. It only runs the DPO finetune stage and saves the merged checkpoint to `model_adapter_granite/`.

## 4. Offline eval command

Offline evaluation uses `src/experiments/eval.py` and re-evaluates saved programs from the logs directory:

```bash
PYTHONPATH=src .venv/bin/python src/experiments/eval.py \
  task=tsp \
  model=granite \
  train=dpo \
  cluster=example \
  gpu_nums=0 \
  prefix=wandb_smoketest \
  seed=0 \
  wandb=1 \
  project=EvoTune \
  entity=zeio99guo-institut-polytechnique-de-paris \
  logs_dir=out/logs/wandb_smoketest/tsp_tsp_granite_dpo_0 \
  evalset=testset \
  eval_frequency=100 \
  use_vllm=0 \
  use_tgi=0
```

Notes:

- `eval.py` evaluates saved programs in the program bank, not the finetuned LLM directly.
- For TSP, `evalset=testset` maps to `("val", "0.0")`.
- `evalset=trainperturbedset` maps to `("train", "0.2")`.

## 5. Current repository defaults

The table below lists the current checked-in defaults used by the training command above.

### Core loop parameters

| Parameter | Current repo default | Note |
| --- | --- | --- |
| `num_rounds` | `2701` | Total number of outer rounds. |
| `num_cont_rounds` | `100` | Number of continuous sampling/evaluation rounds per outer loop. |
| `finetuning_frequency` | `400` | Inherited from `configs/train/dpo.yaml`. |

Defaults are taken from [configs/config.yaml](/mnt/e/currentWORK/AAA_LCM/EvoTune/configs/config.yaml) and [configs/train/dpo.yaml](/mnt/e/currentWORK/AAA_LCM/EvoTune/configs/train/dpo.yaml).

### DPO parameters

| Parameter | Current repo default | Note |
| --- | --- | --- |
| `train.dpo_config.max_seq_length` | `6500` | Default sequence length in `configs/train/dpo.yaml`. |
| `train.dpo_config.per_device_train_batch_size` | `2` | Default per-device batch size. |
| `train.dpo_config.gradient_accumulation_steps` | `16` | Default gradient accumulation. |
| `train.dpo_config.beta` | `0.4` | Default DPO beta. |

## 6. Important clarification

Some earlier experiments used overrides such as `num_rounds=500`, `finetuning_frequency=10`, `train.dpo_config.max_seq_length=512`, or `train.dpo_config.per_device_train_batch_size=1`. Those are experiment-specific overrides, not repository defaults. The commands and tables above follow the current checked-in default configuration.
