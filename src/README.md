# EvoTune TSP Train / Eval Notes

## 1. Search training command

The main search loop uses `src/experiments/main.py`. A representative command for the run stored in `out/logs/wandb_smoketest/tsp_tsp_granite_dpo_0` is:

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
  num_rounds=500 \
  num_cont_rounds=100 \
  use_vllm=0 \
  use_tgi=0 \
  finetuning_frequency=10 \
  train.dpo_config.per_device_train_batch_size=1 \
  train.dpo_config.max_seq_length=512
```

This command drives the full EvoTune loop:

- sample candidate `heuristics()` programs with the current LLM
- evaluate them on the TSP train set
- update the program database and DPO buffer
- periodically finetune the LLM

## 2. Direct finetune command

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

## 3. Offline eval command

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

## 4. Parameter differences from current repo defaults

The table below compares the current checked-in defaults with the values used in the training command above.

### Core loop parameters

| Parameter | Current repo default | Run value | Note |
| --- | --- | --- | --- |
| `num_rounds` | `2701` | `500` | Reduced total outer rounds. |
| `num_cont_rounds` | `100` | `100` | Same as current default. |
| `finetuning_frequency` | `400` | `10` | Much more frequent finetuning than the default. |

Defaults are taken from [configs/config.yaml](/mnt/e/currentWORK/AAA_LCM/EvoTune/configs/config.yaml) and [configs/train/dpo.yaml](/mnt/e/currentWORK/AAA_LCM/EvoTune/configs/train/dpo.yaml).

### DPO parameters

| Parameter | Current repo default | Run value | Note |
| --- | --- | --- | --- |
| `train.dpo_config.max_seq_length` | `6500` | `512` | Reduced heavily to fit memory limits. |
| `train.dpo_config.per_device_train_batch_size` | `2` | `1` | Lowered to reduce memory use. |
| `train.dpo_config.gradient_accumulation_steps` | `16` | `16` | Same as default. |
| `train.dpo_config.beta` | `0.4` | `0.4` | Same as current default. |

## 5. Important clarification

Some older notes may mention different defaults such as `num_rounds=1000`, `num_cont_rounds=20`, or `max_seq_length=1024`. Those are not the current checked-in defaults of this repository at the time this README was written. The table above follows the current repository state.
