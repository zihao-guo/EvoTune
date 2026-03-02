# Evaluation Sandboxes

This folder contains three standalone evaluation sandboxes:

- `tsp/`: Traveling Salesman Problem
- `bp/`: Bin Packing
- `fp/`: FlatPack

Each sandbox contains:

- an extracted best program from the paper,
- self-contained evaluation helpers,
- a `run_eval.py` entrypoint,
- a local `results/` folder for generated outputs.

The top-level `run_all.py` script collects the per-task outputs into a single
`src/eval/results/summary.json`, `src/eval/results/comparison_table.csv`, and
`src/eval/results/report.md`.

Run:

```bash
.venv/bin/python src/eval/tsp/run_eval.py
.venv/bin/python src/eval/bp/run_eval.py
.venv/bin/python src/eval/fp/run_eval.py
.venv/bin/python src/eval/run_all.py
```
