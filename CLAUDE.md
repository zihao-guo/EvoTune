# Model Workflow

## Role Assignment

- **Fable** — Orchestrator. Plans, decomposes, synthesizes conclusions. Never touches concrete implementation.
- **Opus** (`deep-reasoner` agent) — Architecture and complex problems: deep reasoning, design decisions, ambiguous tradeoffs. Reasoning-heavy phases only; returns concise conclusions.
- **Sonnet** (`fast-worker` agent) — Grunt work: mechanical implementation, boilerplate, tests. Executes efficiently.

## High-Stakes Decision Protocol

When a decision is high-stakes (architecture changes, irreversible actions, critical logic):

1. Run **Sonnet** and **Opus** in parallel on the same problem.
2. **Fable** reviews both outputs and synthesizes the final conclusion.

## QA Loop

- `/codex gpt-5.6-sol ultra` — adversarial QA at each milestone: challenge unreasonable steps (e.g. modifying existing code more than necessary).
- `/codex gpt-5.6-sol medium` — periodic smoke test between milestones to prevent idle drift.

## Project Constraints

- Python interpreter: `/home/zguo/miniconda/envs/evotune/bin/python` (conda env `evotune`) — always.
- Modify existing code as little as possible; prefer additive changes.
- Everything generated must stay inside `/home/zguo/Coding/baseline/EvoTune`.
- Local LLM for evolution runs: `utils/model/Qwen3-14B-Instruct` (never commit model weights).
- Incremental-compile validation target: `pyvrp_rep` (evolving `pyvrp_rep/pyvrp/cpp/crossover/selective_route_exchange.cpp`).
- Git identity for pushes: `zihao-guo <zeio99guo@gmail.com>`, remote `https://github.com/zihao-guo/EvoTune.git`.
