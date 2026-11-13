"""Run the *unchanged* official FML-bench evaluation under a fixed budget.

This file is copied into each candidate's sandbox directory and executed there:

    python eval_wrapper.py --budget budget.json

It (1) imports the candidate's ``algorithm.py``, (2) overrides the keys of
``get_training_config()`` that define the compute budget and the evaluation
protocol with the harness-controlled values from ``budget.json``, and then
(3) calls ``train_eval_baseline.main()`` with ``--split val``.

``train_eval_baseline.py`` itself is byte-identical to upstream (hash-checked
by the harness).  Only ``--split val`` is ever run by the loop: the hidden test
split is never used for selection.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    with open(args.budget) as f:
        budget = json.load(f)
    overrides = budget["config_overrides"]
    seed = budget.get("pretrain_seed")  # None = official set_seed(42); only used by scripts/seed_check.py

    import algorithm  # the candidate's code

    candidate_cfg_fn = algorithm.get_training_config
    candidate_cfg = dict(candidate_cfg_fn())

    def budgeted_config():
        cfg = dict(candidate_cfg_fn())
        cfg.update(overrides)
        return cfg

    # Must happen before train_eval_baseline is imported, because it does
    # `from algorithm import get_training_config`.
    algorithm.get_training_config = budgeted_config

    effective = budgeted_config()
    with open("effective_config.json", "w") as f:
        json.dump({"candidate_config": candidate_cfg, "overrides": overrides,
                   "effective_config": effective}, f, indent=2, default=str)
    print(f"[harness] candidate config: {candidate_cfg}")
    print(f"[harness] effective config (after budget overrides): {effective}", flush=True)

    import train_eval_baseline as official

    if seed is not None:
        # Robustness check only (never used by the selection loop): replace the global
        # seed set at the start of the official main(). Split seeds and the linear-probe
        # seed are separate in the official code and stay unchanged.
        _orig_set_seed = official.set_seed
        official.set_seed = lambda _s=42: _orig_set_seed(int(seed))
        print(f"[harness] pretrain seed override: {seed} (robustness check)", flush=True)

    sys.argv = ["train_eval_baseline.py", "--split", args.split]
    t0 = time.time()
    official.main()
    print(f"[harness] official evaluation wall time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
