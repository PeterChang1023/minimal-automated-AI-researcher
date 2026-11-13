"""Plot per-epoch pretraining loss from real logs.

    python scripts/plot_loss_curves.py runs/main

Reads every loss_curve.json under runs/<run>/iter_*/ and runs/<run>/seed_check/*/
(written by air/eval_wrapper.py) and writes runs/<run>/viz/loss_curves.png:
one panel per code version (baseline, best, ...), one line per seed.
Losses of different objectives (Barlow Twins vs InfoNCE) have different scales,
so each panel has its own y-axis; compare shapes, not values across panels.
There is no per-epoch validation loss: the task's metric is linear-probe accuracy,
measured once after pretraining.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]


def collect(run: Path):
    panels = defaultdict(dict)  # panel -> {seed_label: curve}
    meta = json.loads((run / "best" / "meta.json").read_text()) if (run / "best" / "meta.json").exists() else {}
    for f in sorted(run.glob("iter_*/loss_curve.json")):
        it = int(f.parent.name.split("_")[1])
        name = "baseline" if it == 0 else f"best_iter{it}" if it == meta.get("iter") else f"iter{it}"
        panels[name]["seed 42"] = json.loads(f.read_text())
    for f in sorted(run.glob("seed_check/*/loss_curve.json")):
        m = re.match(r"(.+)_seed(\d+)$", f.parent.name)
        if m:
            panels[m.group(1)][f"seed {m.group(2)}"] = json.loads(f.read_text())
    return panels


def main():
    run = Path(sys.argv[1])
    panels = collect(run)
    if not panels:
        sys.exit("no loss_curve.json found (runs made before per-epoch logging was added have none)")
    names = sorted(panels, key=lambda n: (n != "baseline", n))
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.6), dpi=200, squeeze=False)
    for ax, name in zip(axes[0], names):
        for i, (label, curve) in enumerate(sorted(panels[name].items())):
            ax.plot([c["epoch"] for c in curve], [c["loss"] for c in curve], lw=2,
                    color=SERIES[i % len(SERIES)], label=label)
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_xlabel("pretrain epoch")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel("mean pretraining loss (own scale per panel)")
    fig.tight_layout()
    out = run / "viz" / "loss_curves.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
