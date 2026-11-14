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


def fmt_history(history: list[dict], max_entries: int = 30) -> str:
    if not history:
        return "(no previous experiments)"
    lines = []
    for h in history[-max_entries:]:
        acc = h.get("val_acc")
        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "n/a"
        lines.append(
            f"- iter {h['iter']:>3} | {h.get('kind','candidate')} | status={h['status']} | val_acc={acc_s} "
            f"| decision={h.get('decision','-')}\n"
            f"    hypothesis: {h.get('hypothesis','-')}\n"
            f"    change: {h.get('change_summary','-')}\n"
            f"    feedback: {h.get('feedback','-')}")
    return "\n".join(lines)


def build_user_prompt(budget: dict, best: dict, history: list[dict], baseline_acc) -> str:
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
