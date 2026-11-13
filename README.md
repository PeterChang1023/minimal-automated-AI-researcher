# Minimal automated AI researcher

A small, auditable **Propose → Modify → Execute → Evaluate → Record** loop on the
FML-bench task **`Representation_Learning_solo_learn`** (Barlow Twins, ResNet-18,
CIFAR-100, linear-probe evaluation). An LLM proposes one hypothesis per iteration and
rewrites a *copy* of `algorithm.py`; everything else (checks, training, evaluation,
the accept/reject decision, the feedback text) is deterministic program code.

| | |
|---|---|
| Benchmark | [FML-bench](https://github.com/qrzou/FML-bench) @ `d336651`, task files vendored in [`fmlbench_task/`](fmlbench_task/UPSTREAM.md) (byte-identical, SHA-256 pinned) |
| Metric | `linear_eval_acc_mean` on the validation split, higher is better. Upstream full-budget (130 epochs) val = **47.77 %** |
| Our budget | **50 pretrain epochs** (shortened from 130), everything else as upstream. Change with `--pretrain-epochs` (it starts a new, separately comparable run) |
| LLM | Gemini via Google AI Studio REST (`GEMINI_API_KEY`), default model `gemini-2.5-pro` (`--model`/`GEMINI_MODEL`). Pluggable: `--backend gemini-cli` (Google `gemini` CLI logged in with a Google account, uses the subscription quota), `claude-cli`, `file`, `mock` |
| Libraries | PyTorch 2.1.0 (cu121), torchvision 0.16.0, NumPy 1.26.4, Pillow, matplotlib. Python stdlib only for the loop / LLM client (no SDK, no RAG) |

## 1. Setup on hinton

```bash
# code lives in $HOME; data + checkpoints go to /cache/$USER (lab rule)
git clone https://github.com/PeterChang1023/minimal-automated-AI-researcher.git ~/minimal-automated-AI-researcher
cd ~/minimal-automated-AI-researcher
# keep download caches out of $HOME (lab rule: only code + packages in home)
export PIP_CACHE_DIR=/cache/$USER/pip CONDA_PKGS_DIRS=/cache/$USER/conda_pkgs TORCH_HOME=/cache/$USER/torch
mkdir -p $PIP_CACHE_DIR $CONDA_PKGS_DIRS $TORCH_HOME
conda env create -f environment.yml          # creates env "air"
conda activate air
bash scripts/env_info.sh                     # records GPU / driver / torch for the report
bash scripts/prepare_data.sh                 # CIFAR-100 -> /cache/$USER/air/data
export GEMINI_API_KEY=...                    # typed in your shell only; never committed
python -m air.llm                            # lists Gemini models your key can use (sanity check)
```

The GPU is shared, so pin one device and keep the default batch size (512, same as
upstream; estimated 10–15 GB of GPU memory, not yet measured):

```bash
export CUDA_VISIBLE_DEVICES=0
```

## 2. Commands

Run inside `tmux` (each experiment is ~10–20 min). If tmux is missing, install it into the env without root: `conda install -n air -c conda-forge tmux -y`. Every command is resumable: if it
dies, re-run the same line and it continues from `runs/<name>/history.jsonl`.

```bash
# Part 1 – baseline only, shortened budget
python -m air.loop --run-name main --baseline-only

# Part 2 – one LLM candidate (+ the next proposal, generated from the feedback, not executed)
python -m air.loop --run-name main --max-iters 1 --propose-next

# more iterations later (same run, same budget -> comparable)
python -m air.loop --run-name main --max-iters 5 --propose-next

# Part 3 – visualization from the real logs
python scripts/visualize.py runs/main        # -> runs/main/viz/progress.png, trace.html

# when finished (lab rule): remove data + checkpoints, keep logs
bash scripts/cleanup_cache.sh
rm -rf /cache/$USER/pip /cache/$USER/conda_pkgs /cache/$USER/torch
```

Pipeline test without GPU or API key (random accuracies, clearly marked as FAKE in all outputs):

```bash
python -m air.loop --run-name pipeline-test --backend mock --fake-executor --max-iters 3 --propose-next \
    --data-root /tmp/air-data --work-root /tmp/air-work
python scripts/visualize.py runs/pipeline-test
```

Useful flags: `--timeout-min 40`, `--max-hours 12`, `--max-consecutive-failures 3`,
`--max-batch-size 512`, `--keep-ckpt`, `--model gemini-3.1-pro-preview`. The API free tier may have zero quota for Pro models (HTTP 429 `limit: 0`); then use `--backend gemini-cli` or a billed key.

## 3. How one iteration works

| step | who | what |
|---|---|---|
| Propose | **LLM** | Prompt = task + fixed rules + current best `algorithm.py` (full) + every past attempt's hypothesis, status, val acc, decision and feedback ([`air/prompts.py`](air/prompts.py)). Reply = JSON (`hypothesis`, `rationale`, `expected_effect`, `change_summary`) + the complete new `algorithm.py`. |
| Modify | program | Written to `runs/<run>/iter_k/algorithm.py` (a copy). The upstream baseline in `fmlbench_task/` is never edited; diffs vs best and vs baseline are saved. |
| Check | program | Static checks ([`air/checks.py`](air/checks.py)): required functions, no data loading / test-split / result-file references, no network, shell, file I/O or pretrained weights, batch size ≤ cap. Then a CPU smoke test ([`air/smoke_check.py`](air/smoke_check.py)) of exactly the interface the official script uses. Failing candidates never reach the GPU. |
| Execute | program | Fresh sandbox under `/cache/$USER/air/work/<run>/iter_k/` with the candidate + the **unchanged** `train_eval_baseline.py` + fixed `split_config.json`. [`air/eval_wrapper.py`](air/eval_wrapper.py) overrides only the budget/protocol keys of `get_training_config()` and calls the official `main()` with `--split val`. Own process group, hard timeout. |
| Evaluate | program | Accept **iff** the run finished, produced a finite metric, and `val_acc > best_val_acc` (strict). Otherwise the best code stays. |
| Record | program | `history.jsonl` (one line per iteration) + `iter_k/` artifacts; the feedback sentence is what the LLM sees next time. `best/algorithm.py` always holds the best valid code, `baseline/algorithm.py` the original. |

### What is fixed (so numbers are comparable)

* **Evaluation**: `train_eval_baseline.py` is SHA-256 checked before every run; the linear
  probe protocol (100 epochs, lr 0.1, batch 256, seed 100) is forced by the wrapper.
* **Split**: `split_config.json` (val_ratio 0.057 → 3,021 val images, seed 42) is fixed;
  the LLM may not change it (upstream allows it, we don't, because a different split makes accuracies incomparable).
* **Seeds**: `set_seed(42)` in the official script; linear probe seed 100.
* **Budget**: `pretrain_epochs`, batch-size cap and timeout are part of the run
  fingerprint in `run_config.json`; resuming with a different budget is refused.
* **Test labels**: the loop only ever runs `--split val`. The hidden test split
  (7,000 images) is never evaluated by the loop, so it cannot influence selection.

### Run directory layout

```
runs/<run>/
  run_config.json          budget, seeds, fingerprint, LLM, hardware, argv (+ resumes)
  history.jsonl            one record per iteration (the source for the visualization)
  baseline/algorithm.py    original code          best/algorithm.py + meta.json  best valid code
  iter_000/                baseline: algorithm.py, smoke.log, stdout.log, val_info.json, record.json
  iter_001/                prompt_system.txt, prompt_user.txt, llm_response.txt, proposal.json,
                           algorithm.py, diff_vs_best.patch, diff_vs_baseline.patch,
                           smoke.log, stdout.log, effective_config.json, val_info.json, record.json
  next_proposal/           the proposal made from all feedback, not executed
  viz/                     progress.png, trace.html
```

## 4. Results

_To be filled from `runs/main` after running on hinton (commands above)._

| run | iter | what | pretrain epochs | val acc | wall time | decision |
|---|---|---|---|---|---|---|
| upstream (reported) | – | baseline | 130 | 47.77 % | – | – |
| main | 0 | baseline | 50 | | | baseline |
| main | 1 | LLM candidate | 50 | | | |

Hardware: hinton, 1× NVIDIA H100 NVL 94 GB (shared), driver 590 / CUDA 13.1; exact
strings are recorded in `runs/main/run_config.json`.

## 5. Disclosure

* **LLM used by the loop**: Gemini (model id logged per iteration in `history.jsonl`).
* **LLM used to write this repository**: Claude (Anthropic) via Claude Code, with the author reviewing.
* **Manual interventions**: none in the loop logic. Any hand-written proposal must go through
  `--backend file --response-file ...`, which is logged as `backend: file` in the history.
* **Failures**: every failed proposal (parse error, static check, smoke test, crash,
  timeout) is kept in `history.jsonl` and shown in the visualization.
* **Known limitation**: one seed per candidate on a 3,021-image validation set; the
  standard error of an accuracy near 45 % is about ±0.9 pts, so the strict
  "higher is better" rule can accept noise. See the limitation slide.
