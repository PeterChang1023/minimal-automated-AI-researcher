"""Execute step: run one candidate algorithm.py through the official evaluation
in an isolated sandbox directory, with a smoke test first and a hard timeout."""
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
TASK_DIR = PKG.parent / "fmlbench_task"


def _run(cmd, cwd, log_path, timeout, env):
    """Run cmd in its own process group; stream output to log_path. Kill the group on timeout."""
    t0 = time.time()
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_group(p)
            rc, timed_out = None, True
        except BaseException:
            # Ctrl-C / SIGTERM of the loop: the child runs in its own session, so it
            # would otherwise keep training on the shared GPU as an orphan.
            _kill_group(p)
            raise
    return rc, timed_out, time.time() - t0


def _kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    except ProcessLookupError:
        pass


def tail(path, n=25):
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-n:])


def parse_losses(log_text):
    return [{"epoch": int(e), "loss": float(l)} for e, l in
            re.findall(r"Pretrain Epoch (\d+)/\d+: Loss=([-+0-9.eEnaif]+)", log_text)
            if l.lower() not in ("nan", "inf", "-inf")] if log_text else []


class Executor:
    def __init__(self, work_root: Path, data_root: Path, budget: dict, timeout_s: int,
                 smoke_timeout_s: int = 300, keep_ckpt: bool = False, fake: bool = False):
        self.work_root = Path(work_root)
        self.data_root = Path(data_root)
        self.budget = budget
        self.timeout_s = timeout_s
        self.smoke_timeout_s = smoke_timeout_s
        self.keep_ckpt = keep_ckpt
        self.fake = fake  # pipeline test without GPU: random accuracy

    def _prepare_sandbox(self, tag: str, code: str) -> Path:
        sb = self.work_root / tag
        if sb.exists():
            shutil.rmtree(sb)
        sb.mkdir(parents=True)
        (sb / "algorithm.py").write_text(code)
        shutil.copy2(TASK_DIR / "train_eval_baseline.py", sb / "train_eval_baseline.py")
        shutil.copy2(TASK_DIR / "split_config.json", sb / "split_config.json")
        shutil.copy2(PKG / "eval_wrapper.py", sb / "eval_wrapper.py")
        shutil.copy2(PKG / "smoke_check.py", sb / "smoke_check.py")
        b = {"config_overrides": self.budget["config_overrides"]}
        if self.budget.get("pretrain_seed") is not None:
            b["pretrain_seed"] = self.budget["pretrain_seed"]
        (sb / "budget.json").write_text(json.dumps(b, indent=2))
        os.symlink(self.data_root.resolve(), sb / "data")
        return sb

    def run(self, tag: str, code: str, out_dir: Path) -> dict:
        """Returns a result dict with status in {ok, smoke_fail, runtime_error, timeout, no_metric}."""
        out_dir = Path(out_dir)
        sb = self._prepare_sandbox(tag, code)
        res = {"sandbox": str(sb)}

        # 1) smoke test on CPU (program check, cheap)
        env_cpu = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        rc, to, dt = _run([sys.executable, "smoke_check.py"], sb, out_dir / "smoke.log",
                          self.smoke_timeout_s, env_cpu)
        res["smoke_seconds"] = round(dt, 1)
        smoke_txt = tail(out_dir / "smoke.log", 40)
        if rc != 0:
            res.update(status="smoke_fail", error=("smoke test timed out" if to else smoke_txt))
            self._cleanup(sb)
            return res
        try:
            res["smoke"] = json.loads(smoke_txt.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            res["smoke"] = {}

        # 2) official evaluation (val split only) under the fixed budget
        if self.fake:
            import random
            time.sleep(0.2)
            (out_dir / "stdout.log").write_text("[fake executor] no training performed\n")
            acc = round(random.Random(hash(code) % 2**32).uniform(0.30, 0.40), 4)
            res.update(status="ok", val_acc=acc, seconds=0.2, fake=True, losses=[])
            self._cleanup(sb)
            return res

        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        rc, to, dt = _run([sys.executable, "eval_wrapper.py", "--budget", "budget.json"], sb,
                          out_dir / "stdout.log", self.timeout_s, env)
        res["seconds"] = round(dt, 1)
        log_text = (out_dir / "stdout.log").read_text(errors="replace")
        res["losses"] = parse_losses(log_text)
        if (sb / "loss_curve.json").exists():
            shutil.copy2(sb / "loss_curve.json", out_dir / "loss_curve.json")
            try:
                res["loss_curve"] = json.loads((sb / "loss_curve.json").read_text())
            except Exception:  # noqa: BLE001
                pass
        if (sb / "effective_config.json").exists():
            shutil.copy2(sb / "effective_config.json", out_dir / "effective_config.json")
        if to:
            res.update(status="timeout", error=f"killed after {self.timeout_s}s\n" + tail(out_dir / "stdout.log"))
        elif rc != 0:
            res.update(status="runtime_error", error=tail(out_dir / "stdout.log"))
        else:
            vf = sb / "results_tmp" / "val_info.json"
            try:
                acc = json.loads(vf.read_text())["cifar100_ssl"]["means"]["linear_eval_acc_mean"]
                shutil.copy2(vf, out_dir / "val_info.json")
                if not (isinstance(acc, (int, float)) and math.isfinite(acc)):
                    raise ValueError(f"non-finite accuracy {acc}")
                res.update(status="ok", val_acc=float(acc))
            except Exception as e:  # noqa: BLE001
                res.update(status="no_metric", error=f"{e}\n" + tail(out_dir / "stdout.log"))
        self._cleanup(sb)
        return res

    def _cleanup(self, sb: Path):
        if not self.keep_ckpt:
            shutil.rmtree(sb / "model_checkpoint", ignore_errors=True)
