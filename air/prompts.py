"""Prompt construction and response parsing (Propose step).

No retrieval: the prompt contains the task description, the fixed rules, the
current best code (full), and the compact history of every previous attempt
with the program's feedback."""
import json
import re

SYSTEM = (
    "You are an ambitious AI PhD student focused on improving self-supervised "
    "representation learning under compute constraints. You run one controlled "
    "experiment at a time: one clear hypothesis, a minimal code change that tests it, "
    "and you learn from the recorded results of earlier experiments."
)

TASK = """\
## Task (FML-bench: Representation_Learning_solo_learn)
Barlow Twins self-supervised pretraining of a CIFAR-adapted ResNet-18 on CIFAR-100,
evaluated by linear-probe top-1 accuracy on a fixed validation split. Goal: maximise
validation linear-probe accuracy under the FIXED compute budget below.

You may change, inside algorithm.py only: the loss (e.g. VICReg/SimCLR/BYOL-style terms
or Barlow Twins variants), projection head, backbone details, augmentation, and the
pretraining hyperparameters lr / weight_decay / lambda_param / batch_size (<= {max_bs}).

## Fixed by the harness (you cannot change these; any attempt is overridden or rejected)
- Pretraining epochs: {pretrain_epochs}. Optimizer: SGD(momentum=0.9) + cosine schedule
  over those epochs, defined in the official script (not in algorithm.py).
- Linear evaluation protocol: {linear_eval_epochs} epochs, linear_lr={linear_lr},
  linear_batch_size={linear_batch_size}, on frozen `model.backbone` features.
- Train/val split: split_config.json (val_ratio=0.057, val_seed=42) — fixed.
- Seeds: the official script calls set_seed(42); linear-probe seed 100.
- Wall-clock timeout per experiment: {timeout_min} minutes.

## Interface the official script relies on (must be kept)
- get_model() -> nn.Module with attributes .backbone, .projector, .backbone_dim;
  forward(x) returns (features [B, backbone_dim], projections [B, 2048]).
  The official script builds `BarlowTwinsLoss(lambda_param=cfg['lambda_param'], embedding_dim=2048)`
  and calls `loss = criterion(z1, z2)` on the two projection batches, so the class
  name, constructor arguments and the 2048-d projection output must stay.
  (You may implement a different objective inside that class.)
- get_training_config() -> dict with keys pretrain_epochs, linear_eval_epochs,
  batch_size, lr, weight_decay, lambda_param, linear_lr, linear_batch_size.
- get_augmentation() -> transform mapping a 32x32 PIL image to a [3,32,32] tensor.
  It is applied independently to produce the two views.
- Do not load data, read/write files, use the network, pretrained weights, or the shell.
  Only torch / torchvision / numpy / math / PIL may be imported.
"""

INSTRUCTIONS = """\
## What to do now
Propose the NEXT single experiment. Use the history: do not repeat a rejected idea
unchanged, learn from failures (fix the cause), and prefer changes likely to matter
under a SHORT budget (faster convergence). Change one idea at a time so the result is
attributable. Keep the code runnable on one GPU within the timeout.

Reply with exactly two fenced blocks and nothing else:
1. ```json {{"hypothesis": "...", "rationale": "... (cite history entries by iter id if relevant)",
   "expected_effect": "...", "change_summary": "..."}} ```
2. ```python <the COMPLETE new algorithm.py> ```
"""


def fmt_history(history: list[dict], max_entries: int = 30, show_time: bool = False) -> str:
    if not history:
        return "(no previous experiments)"
    lines = []
    for h in history[-max_entries:]:
        acc = h.get("val_acc")
        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "n/a"
        extra = ""
        if show_time:
            sec = h.get("seconds")
            extra += f" | wall={sec/60:.1f} min" if isinstance(sec, (int, float)) else ""
            if h.get("change_type"):
                extra += f" | change_type={h['change_type']}"
        lines.append(
            f"- iter {h['iter']:>3} | {h.get('kind','candidate')} | status={h['status']} | val_acc={acc_s} "
            f"| decision={h.get('decision','-')}{extra}\n"
            f"    hypothesis: {h.get('hypothesis','-')}\n"
            f"    change: {h.get('change_summary','-')}\n"
            f"    feedback: {h.get('feedback','-')}")
    return "\n".join(lines)


def build_user_prompt(budget: dict, best: dict, history: list[dict], baseline_acc,
                      variant: str = "default") -> str:
    if variant == "explore":
        return build_explore_prompt(budget, best, history, baseline_acc)
    task = TASK.format(**budget)
    base_s = f"{baseline_acc:.4f}" if baseline_acc is not None else "not yet measured"
    return (
        f"{task}\n"
        f"## Scores under this budget\n- baseline val acc: {base_s}\n"
        f"- current best val acc: {best['val_acc']:.4f} (iter {best['iter']})\n"
        f"- acceptance rule (program-checked): a candidate replaces the best only if it runs "
        f"successfully and its val acc is STRICTLY higher than the current best.\n\n"
        f"## History of experiments (oldest first)\n{fmt_history(history)}\n\n"
        f"<current_best_algorithm>\n```python\n{best['code']}```\n</current_best_algorithm>\n\n"
        f"{INSTRUCTIONS.format()}"
    )


# ---------------------------------------------------------------------------
# "explore" prompt variant (--prompt-variant explore)
#
# Motivation (Part 4 of the exam): with the default prompt, after the first big
# win (Barlow Twins -> InfoNCE, +9 pts) Gemini only fine-tuned temperature /
# augmentation / head / lr, mostly within seed noise.  The default prompt pushes
# small changes ("minimal code change", "one idea at a time", "SHORT budget")
# and does not say how much of the time limit each run actually uses.  This
# variant keeps the task, the fixed rules and the interface IDENTICAL and only
# changes the guidance: it lists structurally different directions, reports
# the real wall-clock per run vs the limit, and enforces a plateau rule.
# ---------------------------------------------------------------------------
NOISE_PTS = 1.2        # measured seed std of the baseline (3 seeds) in points
PLATEAU_WINDOW = 3     # this many consecutive candidates within noise -> structural change required

SYSTEM_EXPLORE = (
    "You are an ambitious AI PhD student focused on improving self-supervised "
    "representation learning under compute constraints. You run one experiment at a "
    "time and learn from the recorded results, but you balance exploitation (tuning the "
    "current best) with exploration (testing structurally different methods), because "
    "small tweaks quickly fall inside seed noise."
)

DIRECTIONS_EXPLORE = """## Directions you may explore (all inside algorithm.py; weights are always trained from scratch)
A. A DIFFERENT SSL objective (solo-learn implements all of these; you can implement them
   inside the `BarlowTwinsLoss` class and/or the model):
   - VICReg (variance + invariance + covariance terms), W-MSE / whitening losses
   - SimCLR / NT-Xent variants, DCL, supervised-free hard-negative weighting
   - MoCo-style: a queue of past projections kept as a buffer inside the loss module
     (more negatives than the batch); a momentum (EMA) copy of the network kept as a
     non-parameter attribute of the model and updated in forward() during training
   - BYOL / SimSiam-style: predictor head + stop-gradient. Tip: the official script
     calls model(x1), model(x2) and criterion(z1, z2) only, so you can pack several
     vectors into the 2048-d projection output (e.g. [prediction (1024) | target (1024)])
     and split them inside the loss.
   - SwAV / DINO-style: prototypes / clustering with Sinkhorn or centering + sharpening.
     Learnable prototypes must live in the MODEL (the optimizer only sees model.parameters()).
B. ENCODER ARCHITECTURE: the backbone may be redesigned (e.g. width, depth, block type,
   stem, normalisation, activation, pooling), as long as get_model() returns
   .backbone / .projector / .backbone_dim and the linear probe is trained on
   model.backbone features. No pretrained weights.
C. PROJECTION HEAD design (depth, width, BN, output normalisation).
D. AUGMENTATION policy (crop scale, colour jitter, blur, solarisation, multi-crop-like
   tricks within the two-view interface).
E. Hyperparameters lr / weight_decay / lambda_param / batch_size (<= {max_bs}).
Categories A and B are STRUCTURAL changes; C-E are TUNING.
"""

INSTRUCTIONS_EXPLORE = """## What to do now
Propose the NEXT experiment. Use the history: do not repeat a rejected idea unchanged,
learn from failures (fix the cause). Seed-to-seed noise of a single run is about
+/-{noise} pts, so a change of that size is not a real signal. Differences below that
are not worth a whole run unless nothing else is left.
A structural change (category A or B) may touch several parts of the file at once
if they belong to ONE coherent method (e.g. VICReg loss + its recommended head and
hyperparameters); describe it as one idea. Keep the code runnable on one GPU within
the timeout: use the measured wall-clock above to decide how much extra compute you
can afford.

Reply with exactly two fenced blocks and nothing else:
1. ```json {{"hypothesis": "...", "rationale": "... (cite history entries by iter id if relevant)",
   "expected_effect": "...", "change_summary": "...",
   "change_type": "structural" or "tuning", "direction": "A|B|C|D|E"}} ```
2. ```python <the COMPLETE new algorithm.py> ```
"""


def plateau_status(history: list[dict], best_acc: float,
                   window: int = PLATEAU_WINDOW, noise_pts: float = NOISE_PTS) -> tuple[bool, list[int]]:
    """True if the last `window` candidates all ran ok and landed within noise of the
    best at that time (|delta| <= noise) or were accepted by less than the noise,
    i.e. nothing clearly moved.  Returns (plateau, iter_ids)."""
    cands = [h for h in history if h.get("kind") == "candidate"]
    last = cands[-window:]
    if len(last) < window:
        return False, []
    for h in last:
        d = h.get("delta_pts")
        if h.get("status") != "ok" or d is None or abs(d) > noise_pts:
            return False, []
    return True, [h["iter"] for h in last]


def time_summary(history: list[dict], timeout_min: float) -> str:
    secs = [h["seconds"] for h in history if h.get("status") == "ok" and isinstance(h.get("seconds"), (int, float))]
    if not secs:
        return f"- per-experiment time limit: {timeout_min} min (no completed run yet)"
    last, mx = secs[-1] / 60, max(secs) / 60
    return (f"- per-experiment time limit: {timeout_min} min. Measured wall-clock of completed runs: "
            f"last {last:.1f} min, max {mx:.1f} min (pretrain + linear eval) -> about "
            f"{timeout_min - mx:.0f} min of headroom ({100 * mx / timeout_min:.0f}% of the limit used). "
            f"Heavier methods (bigger backbone, momentum encoder, extra heads) are affordable if they "
            f"stay inside the limit; a timeout counts as a failed experiment.")


def build_explore_prompt(budget: dict, best: dict, history: list[dict], baseline_acc) -> str:
    task = TASK.format(**budget)
    if budget.get("gpu_mem_gb"):
        task += (f"- GPU memory cap (shared GPU): {budget['gpu_mem_gb']:g} GB per experiment; exceeding it "
                 f"crashes the run with CUDA OOM and counts as a failure.\n")
    base_s = f"{baseline_acc:.4f}" if baseline_acc is not None else "not yet measured"
    plateau, ids = plateau_status(history, best["val_acc"])
    n_struct = sum(1 for h in history if h.get("change_type") == "structural")
    n_cand = sum(1 for h in history if h.get("kind") == "candidate")
    rule = (f"## Plateau rule (detected by the program)\n"
            f"If the last {PLATEAU_WINDOW} experiments all landed within +/-{NOISE_PTS} pts of the best, "
            f"the next experiment MUST be a structural change (category A or B).\n")
    if plateau:
        rule += (f"**PLATEAU DETECTED: iters {ids} were all within noise. This experiment MUST be "
                 f"structural (change_type=\"structural\", direction A or B).**\n")
    else:
        rule += "Status: no plateau right now; you are free to choose.\n"
    rule += f"So far {n_struct} of {n_cand} experiments were structural.\n"
    return (
        f"{task}\n"
        f"{DIRECTIONS_EXPLORE.format(**budget)}\n"
        f"## Scores and compute under this budget\n- baseline val acc: {base_s}\n"
        f"- current best val acc: {best['val_acc']:.4f} (iter {best['iter']})\n"
        f"{time_summary(history, budget['timeout_min'])}\n"
        f"- acceptance rule (program-checked): a candidate replaces the best only if it runs "
        f"successfully and its val acc is STRICTLY higher than the current best.\n\n"
        f"{rule}\n"
        f"## History of experiments (oldest first)\n{fmt_history(history, show_time=True)}\n\n"
        f"<current_best_algorithm>\n```python\n{best['code']}```\n</current_best_algorithm>\n\n"
        f"{INSTRUCTIONS_EXPLORE.format(noise=NOISE_PTS)}"
    )


class ParseError(ValueError):
    pass


def parse_response(text: str) -> tuple[dict, str]:
    """Return (proposal_dict, code). Raises ParseError."""
    jm = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    pm = re.findall(r"```python\s*\n(.*?)```", text, re.S)
    if not jm:
        raise ParseError("no ```json``` block with the proposal")
    if not pm:
        raise ParseError("no ```python``` block with algorithm.py")
    try:
        proposal = json.loads(jm.group(1))
    except json.JSONDecodeError as e:
        raise ParseError(f"proposal JSON invalid: {e}") from e
    for k in ["hypothesis", "rationale", "expected_effect", "change_summary"]:
        proposal.setdefault(k, "")
    code = max(pm, key=len)  # the full file is the longest python block
    if not code.endswith("\n"):
        code += "\n"
    return proposal, code
