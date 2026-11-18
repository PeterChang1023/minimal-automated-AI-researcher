"""Remove iterations that never ran an experiment (LLM call failed: status llm_error)
from a run, so the run can be resumed with a clean history.

    python scripts/drop_failed_iters.py runs/<run>            # dry run: show what would go
    python scripts/drop_failed_iters.py runs/<run> --apply    # back up, then remove

Removes: their records in history.jsonl, their iter_XXX/ folders, the next_proposal/
folder (re-created at the end of the next run), and their lines in runs/<run>.log.
Iteration numbers of the remaining experiments are NOT changed (Gemini's rationales cite
them), so numbering keeps gaps. A full copy is first written to ~/air_backups/<run>_<time>/ (outside the repo).
By default only llm_error iterations are removed; pass --status llm_error,timeout to also
remove runs killed by the time limit (e.g. before resuming with a larger --timeout-min).
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--apply", action="store_true")
ap.add_argument("--status", default="llm_error",
                help="comma-separated statuses to remove (default llm_error; e.g. llm_error,timeout)")
a = ap.parse_args()
run = Path(a.run_dir).resolve()
log = run.parent / f"{run.name}.log"
hist = [json.loads(l) for l in (run / "history.jsonl").read_text().splitlines() if l.strip()]
drop = set(a.status.split(","))
bad = sorted(h["iter"] for h in hist if h.get("kind") == "candidate" and h.get("status") in drop)
keep = [h for h in hist if h["iter"] not in bad]
print(f"{a.status} iters to remove: {bad}; kept iters: {[h['iter'] for h in keep]}")

log_lines = log.read_text().splitlines(True) if log.exists() else []
out, skip, tail = [], False, False
for line in log_lines:
    m = re.search(r"\] iter (\d+): proposing", line)
    if m:
        skip, tail = int(m.group(1)) in bad, False
    if tail:  # multi-line error tail printed after a removed iteration's result line
        if re.match(r"^\[\d{4}-", line):
            tail = False
        else:
            continue
    if skip:
        if re.search(r"\] iter (\d+) -> ", line):
            skip, tail = False, True
        continue
    if "consecutive failures; stopping" in line or "next proposal written to" in line or line.startswith("[harness] done."):
        continue
    out.append(line)
# drop [llm] retry lines of the final next-proposal call (they follow the last kept record)
while out and out[-1].startswith("[llm] "):
    out.pop()
print(f"log: {len(log_lines)} -> {len(out)} lines")

if not a.apply:
    print("dry run; add --apply to change files")
    raise SystemExit
bk = Path.home() / "air_backups" / f"{run.name}_{time.strftime('%Y%m%d-%H%M%S')}"
shutil.copytree(run, bk / run.name)
if log.exists():
    shutil.copy2(log, bk / log.name)
for i in bad:
    shutil.rmtree(run / f"iter_{i:03d}", ignore_errors=True)
shutil.rmtree(run / "next_proposal", ignore_errors=True)
(run / "history.jsonl").write_text("".join(json.dumps(h) + "\n" for h in keep))
if log.exists():
    log.write_text("".join(out))
print(f"done; backup in {bk}")
