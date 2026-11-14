# Where each exam requirement is answered

Brief: *Take-home exam 2: Build a minimal automated AI researcher* (FML-bench, `Representation_Learning_solo_learn`).
Legend: ✅ done in repo · 🖥 goes on a slide · ⏳ still to do

## Part 1. Understand and run the baseline (5 min)

| Requirement | Status | Where |
|---|---|---|
| Explain Barlow Twins | 🖥 | Slide. Source: [`fmlbench_task/algorithm.py`](../fmlbench_task/algorithm.py) (`BarlowTwinsLoss`: cross-correlation of two views → identity; on-diagonal → 1, off-diagonal × λ=5e-3 → 0) |
| Explain linear evaluation | 🖥 | Slide. Source: `linear_eval()` in [`fmlbench_task/train_eval_baseline.py`](../fmlbench_task/train_eval_baseline.py): freeze backbone, extract 512-d features, train one linear layer 100 epochs, top-1 on val |
| One possible modification | 🖥 | Slide. Use the one the loop actually found: InfoNCE/SimCLR loss (iter 5); alternatives VICReg, lower λ, shorter-schedule lr |
| Run baseline / shortened experiment | ✅ | `runs/main/iter_000/`, `runs/flash35-clean/iter_000/` (50 epochs) and `runs/full130/iter_000/` (130 epochs) |
| Validation accuracy | ✅ | 28.67 % (50 ep) · 47.04 % (130 ep) vs reported 47.77 % — README §4 |
| Commands | ✅ | README §1–2 (`--baseline-only`, `--pretrain-epochs 130`) |
| Hardware | ✅ | README §4, `runs/main/run_config.json` (H100 NVL, EPYC 9124, torch 2.1.0+cu121) |
| Seed | ✅ | `set_seed(42)`, linear-probe seed 100, split seed 42 — README §3, `run_config.json` |
| Training budget | ✅ | 50 pretrain epochs ≈ 430 s; 130 epochs ≈ 1,070 s — README §4 |

## Part 2. Build an autonomous research loop (12 min)

| Requirement | Status | Where |
|---|---|---|
| Propose → Modify → Execute → Evaluate → Record | ✅ | [`air/loop.py`](../air/loop.py) (`propose`, `run_candidate`, `evaluate`, `_append`); README §3 table |
| LLM proposes hypothesis, edits a **copy** of algorithm.py | ✅ | `runs/main/iter_005/proposal.json`, `algorithm.py`, `diff_vs_baseline.patch`; upstream file never edited |
| Code executes, evaluates, accepts/rejects, records feedback | ✅ | [`air/executor.py`](../air/executor.py), `Run.evaluate`, `history.jsonl` field `feedback` |
| Preserve baseline and best valid code | ✅ | `runs/main/baseline/`, `runs/main/best/` (+ `meta.json`) |
| Handle failures | ✅ | static checks [`air/checks.py`](../air/checks.py), CPU smoke test [`air/smoke_check.py`](../air/smoke_check.py), LLM errors → iters 1–4 recorded as rejected |
| Handle timeouts | ✅ | `--timeout-min` (default 40), process-group kill in `executor._run` |
| Experiment limits | ✅ | `--max-iters`, `--max-hours`, `--max-consecutive-failures`, batch cap 512 |
| One baseline + one LLM candidate completed | ✅ | `runs/main`: iter 0 (28.67 %) and iter 5 (51.41 %, accepted); clean run `runs/flash35-clean`: baseline 26.75 % + 5 real candidates, best 54.19 % (iter 4) |
| Next proposal using feedback | ✅ | `runs/main/next_proposal/` (τ 0.2 → 0.1); `runs/flash35-clean/next_proposal/` (DCL → DCLW) |
| Accept only higher **comparable** val acc | ✅ | strict `>` in `Run.evaluate`; run fingerprint (budget + eval/split hashes) refuses resume with a different budget |
| Keep official evaluation unchanged | ✅ | SHA-256 pinned `train_eval_baseline.py`; [`air/eval_wrapper.py`](../air/eval_wrapper.py) only overrides budget/protocol keys |
| Fix splits, seeds, budgets | ✅ | `split_config.json` pinned (LLM may not edit), seeds in official script, budget forced by wrapper |
| Held-out test labels out of selection | ✅ | loop only runs `--split val`; static check forbids test/result-file references |

## Part 3. Show what happened (8 min)

| Requirement | Status | Where |
|---|---|---|
| One visualization from real logs | ✅ | `runs/flash35-clean/viz/progress.png` (main figure), `runs/main/viz/progress.png` + `trace.html`, built by [`scripts/visualize.py`](../scripts/visualize.py) from `history.jsonl` |
| Trace proposal → code change → execution → result → feedback | ✅ | `trace.html`, iteration 5 card (stages 1–6) |
| Explain one success or failure | 🖥 | Success: iter 5 (+22.7 pts, diff reviewed). Failure: iters 1–4 (LLM unavailable) recorded, nothing executed, best kept |
| Distinguish LLM decisions vs program checks | ✅ | purple "LLM decision" vs blue "Program check" stages in `trace.html` |

## Part 4. Test one improvement (5 min, one slide, no implementation)

| Requirement | Status | Where |
|---|---|---|
| Main limitation | 🖥 | Single-seed, single-comparison acceptance on a 3,021-image val set (SE ≈ ±0.9 pts) and multi-change proposals (iter 5 changed loss **and** lr) → gains can be noise or unattributable |
| Controlled improvement: hypothesis, comparison, evaluation criterion | 🖥 / ⏳ | Draft: *Hypothesis* — requiring k=3 seeds and a margin reduces false accepts. *Comparison* — current rule vs multi-seed rule on the same proposals. *Criterion* — false-accept rate / rank agreement with 130-epoch results. Optional seed re-runs of iter 5 would give real numbers |

## Send

| Item | Status | Where |
|---|---|---|
| Slides | ⏳ | separate slides thread |
| GitHub repo: code | ✅ | `air/`, `scripts/`, `fmlbench_task/` |
| README | ✅ | `README.md` |
| Environment | ✅ | `environment.yml` |
| Commands | ✅ | README §1–2 |
| Logs | ✅ | `runs/main/`, `runs/flash35-clean/` + `runs/flash35-clean.log` (full), `runs/full130/` + `runs/full130.log` |
| Visualization | ✅ | `runs/flash35-clean/viz/`, `runs/main/viz/` |
| Disclose LLM, libraries, interventions, failures | ✅ | README header table + §5 Disclosure; failures in §4 table |
