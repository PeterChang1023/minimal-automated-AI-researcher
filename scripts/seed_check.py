"""Seed robustness check (outside the selection loop).

Re-runs the baseline and the accepted best code of a run with extra global seeds,
under the same budget and the unchanged official evaluation (val split only):

    python scripts/seed_check.py runs/main --seeds 1 2

Writes runs/<run>/seed_check/<which>_seed<k>/ (logs + val_info.json) and
runs/<run>/seed_check/summary.json. The seed-42 numbers come from the loop's own
iterations. Nothing here changes history.jsonl or the best code.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from air.executor import Executor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-root", default=None)
    ap.add_argument("--timeout-min", type=float, default=40)
    ap.add_argument("--fake-executor", action="store_true")
    a = ap.parse_args()

    import os
    user = os.environ.get("USER", "user")
    run = Path(a.run_dir)
    cfg = json.loads((run / "run_config.json").read_text())
    hist = [json.loads(l) for l in (run / "history.jsonl").read_text().splitlines() if l.strip()]
    best_meta = json.loads((run / "best" / "meta.json").read_text())
    base = next(h for h in hist if h["iter"] == 0 and h["status"] == "ok")
    best = next(h for h in hist if h["iter"] == best_meta["iter"])
    codes = {"baseline": (run / "baseline" / "algorithm.py").read_text(),
             f"best_iter{best['iter']}": (run / "best" / "algorithm.py").read_text()}
    seed42 = {"baseline": base["val_acc"], f"best_iter{best['iter']}": best["val_acc"]}

    out_root = run / "seed_check"
    out_root.mkdir(exist_ok=True)
    summary_path = out_root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for which, code in codes.items():
        summary.setdefault(which, {"42": seed42[which]})
        for seed in a.seeds:
            if str(seed) in summary[which]:
                print(f"skip {which} seed {seed} (done: {summary[which][str(seed)]})")
                continue
            budget = dict(cfg["budget"], pretrain_seed=seed)
            ex = Executor(Path(a.work_root or f"/cache/{user}/air/work") / f"{run.name}_seedcheck",
                          Path(a.data_root or f"/cache/{user}/air/data"), budget,
                          timeout_s=int(a.timeout_min * 60), fake=a.fake_executor)
            out = out_root / f"{which}_seed{seed}"
            out.mkdir(exist_ok=True)
            print(f"running {which} seed {seed} ...", flush=True)
            res = ex.run(f"{which}_seed{seed}", code, out)
            (out / "result.json").write_text(json.dumps({k: v for k, v in res.items() if k != "sandbox"}, indent=2))
            if res["status"] != "ok":
                print(f"  FAILED: {res['status']}\n{res.get('error','')[-800:]}")
                continue
            summary[which][str(seed)] = res["val_acc"]
            summary_path.write_text(json.dumps(summary, indent=2))
            print(f"  {which} seed {seed}: val_acc={res['val_acc']:.4f}", flush=True)

    print("\nsummary (val acc, %):")
    for which, d in summary.items():
        v = [100 * x for x in d.values()]
        sd = statistics.stdev(v) if len(v) > 1 else float("nan")
        print(f"  {which:<14} n={len(v)} mean={statistics.mean(v):.2f} sd={sd:.2f} "
              f"values={', '.join(f'{s}:{100*x:.2f}' for s, x in d.items())}")


if __name__ == "__main__":
    main()
