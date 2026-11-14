"""Re-run one accepted algorithm.py under a different pretraining budget (outside the loop).

Used to check whether a candidate selected under the shortened 50-epoch budget also
beats the baseline under the full upstream budget (130 epochs):

    python scripts/run_full_budget.py runs/flash35-clean --pretrain-epochs 130

Takes runs/<run>/best/algorithm.py (or --code PATH), runs it through the same Executor
as the loop (static smoke test + unchanged official evaluation, val split only, seed 42,
linear probe forced to 100 epochs / lr 0.1 / batch 256) and writes
runs/<run>_best_full<epochs>/: result.json, stdout.log, loss_curve.json, val_info.json,
effective_config.json and a copy of the code. Compare with runs/full130 (baseline, 47.04 %).
Nothing here changes the source run's history.jsonl or best code.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from air.executor import Executor  # noqa: E402


def main():
    user = os.environ.get("USER", "user")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="source run, e.g. runs/flash35-clean")
    ap.add_argument("--code", default=None, help="algorithm.py to run (default: <run_dir>/best/algorithm.py)")
    ap.add_argument("--pretrain-epochs", type=int, default=130)
    ap.add_argument("--out", default=None, help="output dir (default: runs/<run>_best_full<epochs>)")
    ap.add_argument("--data-root", default=f"/cache/{user}/air/data")
    ap.add_argument("--work-root", default=f"/cache/{user}/air/work")
    ap.add_argument("--timeout-min", type=float, default=60)
    ap.add_argument("--fake-executor", action="store_true")
    a = ap.parse_args()

    run = Path(a.run_dir)
    code_path = Path(a.code) if a.code else run / "best" / "algorithm.py"
    code = code_path.read_text()
    src_budget = json.loads((run / "run_config.json").read_text())["budget"]
    overrides = dict(src_budget["config_overrides"], pretrain_epochs=a.pretrain_epochs)
    budget = {"config_overrides": overrides}

    out = Path(a.out) if a.out else REPO / "runs" / f"{run.name}_best_full{a.pretrain_epochs}"
    if (out / "result.json").exists() and json.loads((out / "result.json").read_text()).get("status") == "ok":
        sys.exit(f"{out}/result.json already has a finished result; pass --out to write elsewhere")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(code_path, out / "algorithm.py")
    meta = {"source_run": str(run), "code": str(code_path),
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "source_best": json.loads((run / "best" / "meta.json").read_text()) if (run / "best" / "meta.json").exists() else None,
            "budget": overrides, "seed": "official set_seed(42), linear probe seed 100",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv}

    ex = Executor(Path(a.work_root) / out.name, Path(a.data_root), budget,
                  timeout_s=int(a.timeout_min * 60), fake=a.fake_executor)
    print(f"running {code_path} with pretrain_epochs={a.pretrain_epochs} -> {out}", flush=True)
    res = ex.run(out.name, code, out)
    meta.update({k: v for k, v in res.items() if k not in ("sandbox", "loss_curve", "losses")})
    (out / "result.json").write_text(json.dumps(meta, indent=2))
    if res["status"] != "ok":
        sys.exit(f"FAILED: {res['status']}\n{res.get('error', '')[-800:]}")
    print(f"val_acc={res['val_acc']:.4f}  ({res['seconds']:.0f}s)  -> {out}/result.json", flush=True)


if __name__ == "__main__":
    main()
