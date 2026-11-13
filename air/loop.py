"""Minimal automated AI researcher:  Propose -> Modify -> Execute -> Evaluate -> Record.

    python -m air.loop --run-name demo --max-iters 1 --backend gemini

Iteration 0 is always the unmodified baseline (measured under the same fixed
budget). Every later iteration:
  Propose  [LLM]     : hypothesis + full new algorithm.py, given task, best code and history
  Modify   [program] : write it to a COPY (runs/<run>/iter_k/algorithm.py); the baseline file is never touched
  Check    [program] : static checks (interface, forbidden patterns), CPU smoke test
  Execute  [program] : official train_eval_baseline.py (--split val) under the fixed budget + timeout
  Evaluate [program] : accept iff status ok and val_acc > best_val_acc (strict)
  Record   [program] : history.jsonl + per-iteration artifacts + textual feedback for the next prompt
The run is resumable: re-running the same command continues from history.jsonl.
"""
import argparse
import datetime as dt
import difflib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from .checks import sha256_file, static_check
from .executor import TASK_DIR, Executor
from .llm import LLMError, make_backend
from .prompts import SYSTEM, ParseError, build_user_prompt, parse_response

REPO = Path(__file__).resolve().parent.parent
PINNED = {  # sha256 of the official, unchanged files (see fmlbench_task/SHA256SUMS)
    "train_eval_baseline.py": "6a4f78b42b40da3b9c39b6b5e4db587372623dba7b20f1b95ee64175a2a1d682",
    "split_config.json": "e5f83281234a2ffa594497fa6e42c38bc3cb7a326b846233c2774550128a7970",
    "algorithm.py": "2e1db633c1ec2a4bdf56debc3de1f84c25823fe01bf575a2f6b47d7398244335",
}


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def hardware_info():
    info = {"hostname": socket.gethostname(), "python": platform.python_version(), "platform": platform.platform(),
            "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
            "cpu_count": os.cpu_count(), "repo_commit": sh(f"git -C {REPO} rev-parse HEAD")}
    try:
        import torch
        import torchvision
        info.update(torch=torch.__version__, torchvision=torchvision.__version__,
                    cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    except Exception:  # noqa: BLE001
        pass
    return info


def unified_diff(a: str, b: str, a_name: str, b_name: str) -> str:
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), a_name, b_name))


class Run:
    def __init__(self, args):
        self.args = args
        self.dir = REPO / "runs" / args.run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hist_path = self.dir / "history.jsonl"
        self.budget = {
            "pretrain_epochs": args.pretrain_epochs,
            "linear_eval_epochs": 100, "linear_lr": 0.1, "linear_batch_size": 256,  # official protocol
            "max_bs": args.max_batch_size,
            "timeout_min": round(args.timeout_min, 1),
        }
        # keys forced onto the candidate's get_training_config() by eval_wrapper.py
        self.budget["config_overrides"] = {
            "pretrain_epochs": args.pretrain_epochs,
            "linear_eval_epochs": 100, "linear_lr": 0.1, "linear_batch_size": 256,
        }
        self.fingerprint = {
            "budget": self.budget["config_overrides"],
            "eval_sha256": PINNED["train_eval_baseline.py"],
            "split_sha256": PINNED["split_config.json"],
            "max_batch_size": args.max_batch_size,
        }
        self.history = self._load_history()
        work_root = Path(args.work_root) / args.run_name
        self.executor = Executor(work_root, Path(args.data_root), self.budget,
                                 timeout_s=int(args.timeout_min * 60), keep_ckpt=args.keep_ckpt,
                                 fake=args.fake_executor)

    # ---------- state ----------
    def _load_history(self):
        if not self.hist_path.exists():
            return []
        return [json.loads(l) for l in self.hist_path.read_text().splitlines() if l.strip()]

    def _append(self, rec):
        with open(self.hist_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.history.append(rec)

    def best(self):
        acc = [h for h in self.history if h.get("decision") in ("baseline", "accepted")]
        if not acc:
            return None
        b = acc[-1]
        return {"iter": b["iter"], "val_acc": b["val_acc"],
                "code": (self.dir / f"iter_{b['iter']:03d}" / "algorithm.py").read_text()}

    def baseline_acc(self):
        for h in self.history:
            if h["iter"] == 0 and h["status"] == "ok":
                return h["val_acc"]
        return None

    # ---------- setup ----------
    def verify_and_write_config(self):
        for fname, digest in PINNED.items():
            got = sha256_file(TASK_DIR / fname)
            if got != digest:
                sys.exit(f"[harness] {fname} does not match the pinned upstream hash; refusing to run.")
        cfg_path = self.dir / "run_config.json"
        if cfg_path.exists():
            old = json.loads(cfg_path.read_text())
            if old["fingerprint"] != self.fingerprint:
                sys.exit("[harness] budget/eval fingerprint differs from the existing run; results would not be "
                         f"comparable. Use a new --run-name.\nold={old['fingerprint']}\nnew={self.fingerprint}")
            old.setdefault("resumes", []).append({"time": now(), "argv": sys.argv})
            cfg_path.write_text(json.dumps(old, indent=2))
        else:
            cfg_path.write_text(json.dumps({
                "created": now(), "argv": sys.argv, "fingerprint": self.fingerprint, "budget": self.budget,
                "seeds": {"official_set_seed": 42, "linear_probe_seed_val": 100, "split": "val_ratio=0.057,val_seed=42"},
                "llm": {"backend": self.args.backend, "model": self.args.model or os.environ.get("GEMINI_MODEL", "")},
                "hardware": hardware_info(), "fake_executor": self.args.fake_executor,
                "selection_split": "val only (hidden test never run by the loop)",
            }, indent=2))

    # ---------- iteration 0: baseline ----------
    def run_baseline(self):
        it_dir = self.dir / "iter_000"
        it_dir.mkdir(exist_ok=True)
        code = (TASK_DIR / "algorithm.py").read_text()
        (it_dir / "algorithm.py").write_text(code)
        (self.dir / "baseline").mkdir(exist_ok=True)
        shutil.copy2(TASK_DIR / "algorithm.py", self.dir / "baseline" / "algorithm.py")
        print(f"[{now()}] iter 0: running unmodified baseline", flush=True)
        res = self.executor.run("iter_000", code, it_dir)
        rec = {"iter": 0, "kind": "baseline", "time": now(), "proposer": "none (unmodified upstream code)",
               "hypothesis": "unmodified FML-bench Barlow Twins baseline", "change_summary": "none",
               "static_problems": [], **self._result_fields(res), "fingerprint": self.fingerprint}
        if res["status"] == "ok":
            rec["decision"] = "baseline"
            rec["feedback"] = f"Baseline under this budget: val_acc={res['val_acc']:.4f}."
            self._save_best(0, res['val_acc'])
        else:
            rec["decision"] = "baseline_failed"
            rec["feedback"] = f"Baseline failed ({res['status']}): {res.get('error','')[-800:]}"
        (it_dir / "record.json").write_text(json.dumps(rec, indent=2))
        self._append(rec)
        print(f"[{now()}] iter 0 -> {rec['status']} val_acc={rec.get('val_acc')}", flush=True)
        return rec["decision"] == "baseline"

    def _result_fields(self, res):
        keep = ["status", "val_acc", "seconds", "smoke_seconds", "error", "losses", "fake"]
        out = {k: res[k] for k in keep if k in res}
        if "error" in out:
            out["error"] = out["error"][-3000:]
        return out

    def _save_best(self, it, val_acc):
        bd = self.dir / "best"
        bd.mkdir(exist_ok=True)
        shutil.copy2(self.dir / f"iter_{it:03d}" / "algorithm.py", bd / "algorithm.py")
        (bd / "meta.json").write_text(json.dumps({"iter": it, "val_acc": val_acc, "time": now()}, indent=2))

    # ---------- propose + modify ----------
    def propose(self, it_dir: Path):
        """LLM step. Returns (proposal, code, error)."""
        best = self.best()
        user = build_user_prompt(self.budget, best, self.history, self.baseline_acc())
        (it_dir / "prompt_system.txt").write_text(SYSTEM)
        (it_dir / "prompt_user.txt").write_text(user)
        backend = make_backend(self.args.backend, self.args.model, self.args.response_file)
        t0 = time.time()
        try:
            text = backend.complete(SYSTEM, user)
        except LLMError as e:
            return None, None, f"LLM call failed: {e}", backend.describe()
        meta = {**backend.describe(), "seconds": round(time.time() - t0, 1)}
        (it_dir / "llm_response.txt").write_text(text)
        try:
            proposal, code = parse_response(text)
        except ParseError as e:
            return None, None, f"LLM output could not be parsed: {e}", meta
        (it_dir / "proposal.json").write_text(json.dumps(proposal, indent=2, ensure_ascii=False))
        (it_dir / "algorithm.py").write_text(code)  # Modify: a COPY, never the baseline file
        (it_dir / "diff_vs_best.patch").write_text(
            unified_diff(best["code"], code, f"iter_{best['iter']:03d}/algorithm.py (best)", f"{it_dir.name}/algorithm.py"))
        (it_dir / "diff_vs_baseline.patch").write_text(
            unified_diff((TASK_DIR / "algorithm.py").read_text(), code, "baseline/algorithm.py", f"{it_dir.name}/algorithm.py"))
        return proposal, code, None, meta

    # ---------- one candidate iteration ----------
    def run_candidate(self, it: int):
        it_dir = self.dir / f"iter_{it:03d}"
        it_dir.mkdir(exist_ok=True)
        best = self.best()
        print(f"[{now()}] iter {it}: proposing (best so far {best['val_acc']:.4f} @ iter {best['iter']})", flush=True)
        proposal, code, err, llm_meta = self.propose(it_dir)
        rec = {"iter": it, "kind": "candidate", "time": now(), "parent_iter": best["iter"],
               "parent_val_acc": best["val_acc"], "llm": llm_meta, "fingerprint": self.fingerprint}
        if err:
            rec.update(status="llm_error", decision="rejected", static_problems=[], error=err,
                       feedback=f"FAILED before execution: {err}. Reply with the two required fenced blocks.")
        else:
            rec.update({k: proposal[k] for k in ["hypothesis", "rationale", "expected_effect", "change_summary"]})
            problems = static_check(code)
            bs = self._literal_batch_size(code)
            if bs is not None and bs > self.args.max_batch_size:
                problems.append(f"batch_size={bs} exceeds the shared-GPU cap {self.args.max_batch_size}")
            rec["static_problems"] = problems
            if problems:
                rec.update(status="static_fail", decision="rejected",
                           feedback="REJECTED by static checks (not executed): " + "; ".join(problems))
            else:
                print(f"[{now()}] iter {it}: executing — {proposal['change_summary'][:120]}", flush=True)
                res = self.executor.run(f"iter_{it:03d}", code, it_dir)
                rec.update(self._result_fields(res))
                rec.update(self.evaluate(res, best))
        (it_dir / "record.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        self._append(rec)
        if rec["decision"] == "accepted":
            self._save_best(it, rec['val_acc'])
        print(f"[{now()}] iter {it} -> {rec['status']} / {rec['decision']}: {rec['feedback'][:200]}", flush=True)
        return rec

    @staticmethod
    def _literal_batch_size(code):
        import re
        m = re.search(r"['\"]batch_size['\"]\s*:\s*(\d+)", code)
        return int(m.group(1)) if m else None

    def evaluate(self, res, best):
        """Program decision (no LLM involvement)."""
        st = res["status"]
        if st != "ok":
            msg = {"smoke_fail": "FAILED the CPU smoke test (interface/shape/runtime error), GPU run skipped",
                   "runtime_error": "CRASHED during the official run",
                   "timeout": f"TIMED OUT (>{self.args.timeout_min} min); make it cheaper",
                   "no_metric": "finished but produced no valid metric"}[st]
            return {"decision": "rejected", "feedback": f"{msg}. Error tail:\n{res.get('error','')[-1200:]}"}
        acc, ref = res["val_acc"], best["val_acc"]
        delta = 100 * (acc - ref)
        degenerate = " (near chance: representation collapsed?)" if acc < 0.05 else ""
        if acc > ref:
            return {"decision": "accepted", "delta_pts": round(delta, 2),
                    "feedback": f"ACCEPTED: val_acc {acc:.4f} > best {ref:.4f} ({delta:+.2f} pts). Now the best."}
        return {"decision": "rejected", "delta_pts": round(delta, 2),
                "feedback": f"REJECTED: val_acc {acc:.4f} <= best {ref:.4f} ({delta:+.2f} pts){degenerate}. Best code kept."}

    # ---------- the final 'next proposal' (not executed) ----------
    def propose_next_only(self):
        it_dir = self.dir / "next_proposal"
        if it_dir.exists():
            shutil.rmtree(it_dir)
        it_dir.mkdir()
        proposal, code, err, meta = self.propose(it_dir)
        rec = {"kind": "next_proposal", "time": now(), "status": "proposed_not_executed", "llm": meta,
               "based_on_history_iters": [h["iter"] for h in self.history], "error": err,
               **(proposal or {}), "static_problems": static_check(code) if code else None}
        (it_dir / "record.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        print(f"[{now()}] next proposal written to {it_dir}: {rec.get('hypothesis', err)}", flush=True)

    # ---------- main loop ----------
    def main(self):
        self.verify_and_write_config()
        if not any(h["iter"] == 0 and h["decision"] == "baseline" for h in self.history):
            if self.history:  # failed baseline previously -> retry it
                print("[harness] previous baseline failed; re-running it", flush=True)
                self.history = [h for h in self.history if h["iter"] != 0]
            if not self.run_baseline():
                sys.exit("[harness] baseline failed; fix the environment before running candidates (see iter_000/).")
        if self.args.baseline_only:
            return
        t_start = time.time()
        fails_in_row = 0
        done = sum(1 for h in self.history if h["kind"] == "candidate")
        while done < self.args.max_iters:
            if (time.time() - t_start) / 3600 > self.args.max_hours:
                print("[harness] wall-clock limit reached", flush=True)
                break
            it = max(h["iter"] for h in self.history) + 1
            rec = self.run_candidate(it)
            done += 1
            fails_in_row = fails_in_row + 1 if rec["status"] != "ok" else 0
            if fails_in_row >= self.args.max_consecutive_failures:
                print(f"[harness] {fails_in_row} consecutive failures; stopping", flush=True)
                break
        if self.args.propose_next:
            self.propose_next_only()
        b = self.best()
        print(f"[harness] done. best val_acc={b['val_acc']:.4f} (iter {b['iter']}); baseline={self.baseline_acc()}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    user = os.environ.get("USER", "user")
    p.add_argument("--run-name", required=True)
    p.add_argument("--backend", default="gemini", choices=["gemini", "gemini-cli", "claude-cli", "file", "mock"])
    p.add_argument("--model", default="", help="LLM model id (gemini default: $GEMINI_MODEL or gemini-2.5-pro)")
    p.add_argument("--response-file", default="", help="for --backend file")
    p.add_argument("--max-iters", type=int, default=1, help="number of LLM candidates (excluding baseline)")
    p.add_argument("--max-hours", type=float, default=12.0)
    p.add_argument("--max-consecutive-failures", type=int, default=3)
    p.add_argument("--pretrain-epochs", type=int, default=50, help="fixed shortened budget (upstream: 130)")
    p.add_argument("--max-batch-size", type=int, default=512, help="cap for the shared GPU")
    p.add_argument("--timeout-min", type=float, default=40.0)
    p.add_argument("--data-root", default=f"/cache/{user}/air/data", help="contains cifar-100-python/")
    p.add_argument("--work-root", default=f"/cache/{user}/air/work", help="sandboxes + checkpoints")
    p.add_argument("--keep-ckpt", action="store_true")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--propose-next", action="store_true", help="after the loop, ask for one more proposal (not executed)")
    p.add_argument("--fake-executor", action="store_true", help="pipeline test: random accuracy, no training")
    return p.parse_args(argv)


def main(argv=None):
    Run(parse_args(argv)).main()


if __name__ == "__main__":
    main()
