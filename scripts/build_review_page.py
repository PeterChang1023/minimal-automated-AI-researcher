"""Build a self-contained review page for one loop run, from its real logs.

    python scripts/build_review_page.py runs/<run-name> [--noise-pts 1.2]

Writes runs/<run-name>/viz/review.html. Open it in any browser (no server needed).

What the page shows, all read from history.jsonl and iter_*/ (nothing typed by hand):
  1. overview chart   - val acc per iteration; hover for the change, click to jump to its card
  2. noise band       - +/- one seed standard deviation around the running best
  3. lineage graph    - which version each candidate was built from (record.json parent_iter)
  4. code diff        - side-by-side, toggle "vs parent (best at the time)" / "vs baseline"
  6. loss curve       - pretraining loss per epoch, per iteration
  8. feedback chain   - where iteration k's program-written feedback appears in iteration k+1's prompt
 11. review panel     - per-iteration verdicts and notes; only active when the page is opened as a
                        claude.ai Artifact with the `db` + `user` capabilities, hidden otherwise
"""
import argparse
import json
import re
from pathlib import Path


def read(p: Path):
    return p.read_text(encoding="utf-8") if p.exists() else ""


def history_block(prompt: str, k: int):
    """Return (history_section, start, end) where [start, end) is iteration k's entry inside it."""
    m = re.search(r"## History of experiments.*?(?=\n<current_best_algorithm>|\n## |\Z)", prompt, re.S)
    if not m:
        return "", -1, -1
    sec = m.group(0).rstrip()
    b = re.search(rf"^- iter\s+{k} \|.*?(?=^- iter\s+\d+ \||\Z)", sec, re.S | re.M)
    if not b:
        return sec, -1, -1
    return sec, b.start(), b.start() + len(b.group(0).rstrip())


def collect(run_dir: Path, noise_pts: float):
    hist = [json.loads(l) for l in read(run_dir / "history.jsonl").splitlines() if l.strip()]
    cfg = json.loads(read(run_dir / "run_config.json") or "{}")
    iters = []
    for h in hist:
        k = int(h["iter"])
        d = run_dir / f"iter_{k:03d}"
        rec = json.loads(read(d / "record.json") or "{}")
        rec = {**h, **rec}
        lc = rec.get("loss_curve") or json.loads(read(d / "loss_curve.json") or "[]")
        iters.append({
            "iter": k,
            "kind": rec.get("kind"),
            "status": rec.get("status"),
            "decision": rec.get("decision") or ("baseline" if rec.get("kind") == "baseline" else rec.get("status")),
            "val_acc": rec.get("val_acc"),
            "delta_pts": rec.get("delta_pts"),
            "parent_iter": rec.get("parent_iter"),
            "parent_val_acc": rec.get("parent_val_acc"),
            "time": rec.get("time"),
            "model": (rec.get("llm") or {}).get("model"),
            "llm_seconds": (rec.get("llm") or {}).get("seconds"),
            "seconds": rec.get("seconds"),
            "hypothesis": rec.get("hypothesis", ""),
            "rationale": rec.get("rationale", ""),
            "expected_effect": rec.get("expected_effect", ""),
            "change_summary": rec.get("change_summary", ""),
            "static_problems": rec.get("static_problems") or [],
            "feedback": rec.get("feedback", ""),
            "loss": [[p["epoch"], p["loss"]] for p in lc if p.get("loss") is not None],
            "diff_best": read(d / "diff_vs_best.patch"),
            "diff_base": read(d / "diff_vs_baseline.patch"),
            "prompt_user": read(d / "prompt_user.txt"),
        })
    nd = run_dir / "next_proposal"
    nxt = None
    if (nd / "record.json").exists():
        r = json.loads(read(nd / "record.json"))
        nxt = {
            "status": r.get("status"), "time": r.get("time"),
            "hypothesis": r.get("hypothesis", ""), "rationale": r.get("rationale", ""),
            "expected_effect": r.get("expected_effect", ""), "change_summary": r.get("change_summary", ""),
            "diff_best": read(nd / "diff_vs_best.patch"), "diff_base": read(nd / "diff_vs_baseline.patch"),
            "prompt_user": read(nd / "prompt_user.txt"),
        }
    # feedback chain: iteration k's entry inside the NEXT prompt (k+1, or next_proposal after the last)
    by_k = {it["iter"]: it for it in iters}
    for it in iters:
        nxt_prompt = by_k[it["iter"] + 1]["prompt_user"] if it["iter"] + 1 in by_k else (nxt or {}).get("prompt_user", "")
        sec, s, e = history_block(nxt_prompt, it["iter"])
        it["fed_into"] = (it["iter"] + 1) if it["iter"] + 1 in by_k else ("next" if nxt else None)
        it["fed_section"], it["fed_start"], it["fed_end"] = sec, s, e
    budget = cfg.get("budget") or (cfg.get("fingerprint") or {}).get("budget", {})
    return {
        "run": run_dir.name,
        "created": cfg.get("created"),
        "argv": " ".join(Path(a).name if i == 0 else a for i, a in enumerate(cfg.get("argv", []))),
        "budget": budget,
        "noise_pts": noise_pts,
        "iters": iters,
        "next": nxt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--noise-pts", type=float, default=1.2,
                    help="seed std-dev in val-acc points used for the noise band (default 1.2)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--artifact", action="store_true",
                    help="write page content only (no <!doctype>/<html> wrapper), for publishing as a claude.ai Artifact")
    a = ap.parse_args()
    data = collect(a.run_dir, a.noise_pts)
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    tpl = (Path(__file__).with_name("review_page_template.html")).read_text(encoding="utf-8")
    html = tpl.replace("__RUN_NAME__", data["run"]).replace("/*__DATA__*/null", blob)
    if not a.artifact:
        html = ('<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>\n'
                + html + "\n</body></html>\n")
    out = a.out or (a.run_dir / "viz" / "review.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
