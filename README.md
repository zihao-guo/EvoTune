# EvoTune TSP 运行说明

本文记录在 `evotune` miniconda 环境中，使用本地模型
`/home/zguo/Coding/evohgs/utils/model/Qwen2.5-7B-Instruct` 跑 TSP 默认配置的方法。

## 进入环境

```bash
cd /home/zguo/Coding/baseline/EvoTune
conda activate evotune
```

## 从头开始运行 TSP

使用默认配置、TSP 任务、Qwen2.5-7B-Instruct 模型：

```bash
python src/experiments/main.py task=tsp model=qwen25
```

当前默认日志目录由配置自动生成：

```text
out/logs/example/tsp_tsp_qwen25_dpo_0
```

W&B 默认已开启：

```yaml
wandb: 1
project: evotune_TSP
entity: zeio99guo-institut-polytechnique-de-paris
```

## 后台运行

```bash
mkdir -p out
setsid env PATH=/home/zguo/miniconda/envs/evotune/bin:$PATH \
  PYTHONUNBUFFERED=1 VLLM_USE_V1=0 \
  /home/zguo/miniconda/envs/evotune/bin/python src/experiments/main.py task=tsp model=qwen25 \
  > out/run_tsp_qwen25_default.log 2>&1 < /dev/null &
echo $! > out/run_tsp_qwen25_default.pid
```

查看后台日志：

```bash
tail -f out/run_tsp_qwen25_default.log
```

停止后台运行：

```bash
kill "$(cat out/run_tsp_qwen25_default.pid)"
```

## Resume 指定日志目录

要 resume 这个目录：

```text
/home/zguo/Coding/baseline/EvoTune/out/logs/example/tsp_tsp_qwen25_dpo_0
```

直接使用同一组 `task/model/prefix/seed` 参数重新运行即可：

```bash
python src/experiments/main.py task=tsp model=qwen25 prefix=example seed=0
```

这条命令会重新解析到同一个日志目录：

```text
out/logs/example/tsp_tsp_qwen25_dpo_0
```

代码只有在该目录下存在 `flag_resume.txt` 时才会 resume。如果日志里看到：

```text
Resuming from logs directory
```

说明确实在 resume。如果看到：

```text
Starting from scratch
```

说明没有进入 resume。

可以先检查 resume 所需文件：

```bash
ls out/logs/example/tsp_tsp_qwen25_dpo_0/flag_resume.txt
ls out/logs/example/tsp_tsp_qwen25_dpo_0/running_dict.pkl
ls out/logs/example/tsp_tsp_qwen25_dpo_0/programbank.pkl
ls out/logs/example/tsp_tsp_qwen25_dpo_0/dpo_chats.pkl
ls out/logs/example/tsp_tsp_qwen25_dpo_0/round_num.txt
ls out/logs/example/tsp_tsp_qwen25_dpo_0/input_struct.pkl
```

如果这些文件还没有生成，说明上一次运行还没有到达保存点，不能从该目录 resume。
