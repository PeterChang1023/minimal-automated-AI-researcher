"""Build the Part-3 visualization from a run's real logs.

    python scripts/visualize.py runs/<run-name>

Writes into runs/<run-name>/viz/:
  * progress.png  – val accuracy per iteration, best-so-far line, accept/reject/fail markers
  * trace.html    – self-contained page tracing every iteration:
                    proposal (LLM) -> code diff (LLM) -> checks/execution (program)
                    -> result + decision (program) -> feedback fed to the next prompt (program)
Everything is read from history.jsonl and the iter_*/ folders; nothing is typed in by hand.
"""
import html
import re
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, RED, GRAY = "#2a78d6", "#eb6834", "#e34948", "#8a8984"


def load(run_dir: Path):
    hist = [json.loads(l) for l in (run_dir / "history.jsonl").read_text().splitlines() if l.strip()]
    cfg = json.loads((run_dir / "run_config.json").read_text())
    nxt = run_dir / "next_proposal" / "record.json"
    return hist, cfg, (json.loads(nxt.read_text()) if nxt.exists() else None)


def progress_png(hist, cfg, out: Path):
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)

    ok = [h for h in hist if h["status"] == "ok"]
    accs = [h["val_acc"] for h in ok] or [0.0]
    lo, hi = min(accs), max(accs)
    pad = max(0.01, (hi - lo) * 0.25)
    fail_y = lo - pad * 1.6

    # best-so-far step line (program state)
    xs, ys, best = [], [], None
    for h in hist:
        if h.get("decision") in ("baseline", "accepted"):
            best = h["val_acc"]
        if best is not None:
            xs.append(h["iter"]); ys.append(best)
    ax.step(xs, ys, where="post", color=BLUE, lw=2, label="best so far (kept code)")

    for h in hist:
        i, d = h["iter"], h.get("decision")
        if h["status"] != "ok":
            ax.scatter([i], [fail_y], marker="x", s=70, color=RED, lw=2, zorder=3)
            ax.annotate(h["status"], (i, fail_y), textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=7, color="#52514e")
            continue
        if d == "baseline":
            ax.scatter([i], [h["val_acc"]], s=90, marker="s", color=GRAY, zorder=3, edgecolor="white", lw=1.5)
        elif d == "accepted":
            ax.scatter([i], [h["val_acc"]], s=90, color=BLUE, zorder=3, edgecolor="white", lw=1.5)
        else:
            ax.scatter([i], [h["val_acc"]], s=80, facecolor="white", edgecolor=ORANGE, lw=2, zorder=3)
        ax.annotate(f"{100*h['val_acc']:.2f}%", (i, h["val_acc"]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="#0b0b0b")

    # legend proxies (shape + color, so identity is not color-alone)
    ax.scatter([], [], s=90, marker="s", color=GRAY, label="baseline (iter 0)")
    ax.scatter([], [], s=90, color=BLUE, label="accepted (strictly > best)")
    ax.scatter([], [], s=80, facecolor="white", edgecolor=ORANGE, lw=2, label="rejected (<= best)")
    ax.scatter([], [], marker="x", s=70, color=RED, label="failed check / run")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5,
              handletextpad=0.4, columnspacing=1.2)

    ax.set_ylim(fail_y - pad, hi + pad)
    ax.set_xticks([h["iter"] for h in hist])
    ax.set_xlabel("iteration")
    ax.set_ylabel("validation linear-probe accuracy")
    b = cfg["budget"]
    ax.set_title(f"FML-bench solo-learn · CIFAR-100 · {b['pretrain_epochs']} pretrain epochs · "
                 f"LLM: {llm_label(hist, cfg)}", fontsize=10, loc="left")
    # failed iterations have no accuracy: label that band instead of a fake value
    ticks = [t for t in ax.get_yticks() if t > fail_y + pad * 0.6 and t <= hi + pad]
    ax.set_yticks(ticks + [fail_y])
    ax.set_yticklabels([f"{t:.2f}" for t in ticks] + ["no result"])
    ax.axhline(fail_y + pad * 0.6, color="#c3c2b7", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def err_block(e):
    e = ANSI.sub("", str(e or ""))
    if len(e) <= 300:
        return f"<pre class='err'>{html.escape(e)}</pre>"
    return (f"<pre class='err'>{html.escape(e[:300])}…</pre>"
            f"<details><summary>full error</summary><pre class='err'>{html.escape(e)}</pre></details>")


def llm_label(hist, cfg):
    models = []
    for h in hist:
        m = (h.get("llm") or {}).get("model")
        if m and h["status"] == "ok" and m not in models:
            models.append(m)
    return ", ".join(models) or f"{cfg['llm']['backend']} {cfg['llm']['model']}"


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def esc(x):
    return html.escape(ANSI.sub("", str(x if x is not None else "–")))


def diff_html(patch: str) -> str:
    if not patch.strip():
        return "<p class='muted'>(no diff)</p>"
    rows = []
    for line in patch.splitlines():
        cls = "add" if line.startswith("+") and not line.startswith("+++") else \
              "del" if line.startswith("-") and not line.startswith("---") else \
              "hunk" if line.startswith("@@") else ""
        rows.append(f"<span class='{cls}'>{esc(line)}</span>")
    return "<pre class='diff'>" + "\n".join(rows) + "</pre>"


def stage(owner, title, body):
    tag = "LLM decision" if owner == "llm" else "Program check"
    return (f"<div class='stage {owner}'><div class='stage-h'><span class='tag'>{tag}</span>"
            f"<span class='stage-t'>{esc(title)}</span></div>{body}</div>")


def card(h, run_dir: Path):
    it_dir = run_dir / f"iter_{h['iter']:03d}"
    parts = []
    if h["kind"] == "baseline":
        parts.append(stage("prog", "Proposal", "<p>None: unmodified upstream FML-bench <code>algorithm.py</code>.</p>"))
    elif h["status"] == "llm_error":
        llm = h.get("llm") or {}
        parts.append(stage("prog", f"1 · Propose: LLM call failed ({esc(llm.get('backend'))} {esc(llm.get('model'))})",
            f"<p>No code was produced; nothing was executed.</p>{err_block(h.get('error'))}"))
    else:
        llm = h.get("llm") or {}
        parts.append(stage("llm", f"1 · Propose ({esc(llm.get('backend'))} {esc(llm.get('model'))})",
            f"<dl><dt>Hypothesis</dt><dd>{esc(h.get('hypothesis'))}</dd>"
            f"<dt>Rationale</dt><dd>{esc(h.get('rationale'))}</dd>"
            f"<dt>Expected effect</dt><dd>{esc(h.get('expected_effect'))}</dd></dl>"))
        patch = (it_dir / "diff_vs_best.patch").read_text() if (it_dir / "diff_vs_best.patch").exists() else ""
        parts.append(stage("llm", f"2 · Modify: diff vs best (iter {h.get('parent_iter')})",
            f"<p>{esc(h.get('change_summary'))}</p><details><summary>show diff "
            f"({patch.count(chr(10)+'+')} + / {patch.count(chr(10)+'-')} − lines)</summary>{diff_html(patch)}</details>"))
        sp = h.get("static_problems") or []
        parts.append(stage("prog", "3 · Static checks",
            "<p class='pass'>passed</p>" if not sp else
            "<ul>" + "".join(f"<li>{esc(p)}</li>" for p in sp) + "</ul>"))
    ran = h["status"] in ("ok", "runtime_error", "timeout", "no_metric", "smoke_fail")
    if ran:
        losses = h.get("losses") or []
        loss_s = ", ".join(f"ep{l['epoch']}: {l['loss']:.1f}" for l in losses) or "–"
        err = f"<pre class='err'>{esc(h.get('error'))}</pre>" if h.get("error") else ""
        parts.append(stage("prog", "4 · Execute (smoke test + official eval, val split)",
            f"<p>status <b>{esc(h['status'])}</b> · wall {esc(h.get('seconds'))} s · "
            f"pretrain loss {esc(loss_s)}{' · <b>FAKE executor</b>' if h.get('fake') else ''}</p>{err}"))
    acc = h.get("val_acc")
    acc_s = f"{100*acc:.2f}%" if isinstance(acc, (int, float)) else "–"
    delta = h.get("delta_pts")
    parts.append(stage("prog", "5 · Evaluate (accept iff val acc strictly > best)",
        f"<p class='result'><span class='acc'>{acc_s}</span>"
        f"{f' <span class=delta>{delta:+.2f} pts vs best</span>' if delta is not None else ''}"
        f" <span class='dec {esc(h.get('decision'))}'>{esc(h.get('decision'))}</span></p>"))
    parts.append(stage("prog", "6 · Record → feedback in the next prompt", (f"<pre class='fb'>{esc(h.get('feedback'))}</pre>" if len(str(h.get('feedback'))) < 400 else f"<pre class='fb'>{esc(str(h.get('feedback'))[:300])}…</pre>")))
    return (f"<section class='card'><h2>Iteration {h['iter']} <span class='muted'>· {esc(h['kind'])} · "
            f"{esc(h.get('time'))}</span></h2><div class='stages'>{''.join(parts)}</div></section>")


def next_card(n, run_dir):
    if not n:
        return ""
    patch_p = run_dir / "next_proposal" / "diff_vs_best.patch"
    patch = patch_p.read_text() if patch_p.exists() else ""
    body = (stage("llm", f"Propose (uses feedback from iters {esc(n.get('based_on_history_iters'))})",
                  f"<dl><dt>Hypothesis</dt><dd>{esc(n.get('hypothesis'))}</dd>"
                  f"<dt>Rationale</dt><dd>{esc(n.get('rationale'))}</dd>"
                  f"<dt>Expected effect</dt><dd>{esc(n.get('expected_effect'))}</dd></dl>")
            + stage("llm", "Modify: diff vs best", f"<p>{esc(n.get('change_summary'))}</p>"
                    f"<details><summary>show diff</summary>{diff_html(patch)}</details>")
            + stage("prog", "Static checks", "<p class='pass'>passed</p>" if not n.get("static_problems")
                    else esc(n.get("static_problems")))
            + stage("prog", "Status", "<p>proposed, <b>not executed</b></p>"))
    return f"<section class='card next'><h2>Next proposal</h2><div class='stages'>{body}</div></section>"


CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e0;--llm:#6250d6;--llm-bg:#f1effc;
--prog:#2a78d6;--prog-bg:#eef4fc;--add:#e6f6ee;--del:#fdecec;--acc:#2a78d6;--rej:#b4461d;--bad:#c53030}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;
--line:#3a3a37;--llm:#9085e9;--llm-bg:#2a2740;--prog:#3987e5;--prog-bg:#1f2b3b;--add:#1d3328;--del:#3b2222;--acc:#3987e5;--rej:#e07a4f;--bad:#e66767}}
:root[data-theme="dark"]{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37;--llm:#9085e9;--llm-bg:#2a2740;
--prog:#3987e5;--prog-bg:#1f2b3b;--add:#1d3328;--del:#3b2222;--acc:#3987e5;--rej:#e07a4f;--bad:#e66767}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:0 0 12px}
.muted{color:var(--ink2);font-weight:400;font-size:13px}.meta{color:var(--ink2);font-size:13px}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0}.legend span{display:flex;align-items:center;gap:6px;font-size:13px}
.sw{width:12px;height:12px;border-radius:3px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:16px 0}
.stages{display:grid;gap:8px}.stage{border-left:4px solid;border-radius:6px;padding:8px 12px}
.stage.llm{border-color:var(--llm);background:var(--llm-bg)}.stage.prog{border-color:var(--prog);background:var(--prog-bg)}
.stage-h{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}.tag{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.llm .tag{color:var(--llm)}.prog .tag{color:var(--prog)}.stage-t{font-weight:600}
dl{margin:6px 0}dt{font-size:12px;color:var(--ink2)}dd{margin:0 0 6px}
pre{white-space:pre-wrap;word-break:break-word;font:12px/1.45 ui-monospace,Menlo,monospace;margin:6px 0;overflow-x:auto}
.diff{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px;max-height:480px;overflow:auto}
.diff .add{background:var(--add);display:block}.diff .del{background:var(--del);display:block}.diff .hunk{color:var(--ink2);display:block}
.result{font-size:16px}.acc{font-weight:700;font-size:20px}.delta{color:var(--ink2)}
.dec{font-weight:700;margin-left:8px}.dec.accepted,.dec.baseline{color:var(--acc)}.dec.rejected{color:var(--rej)}
.err{color:var(--bad)}.pass{color:var(--acc);font-weight:600}.next{border-style:dashed}summary{cursor:pointer;color:var(--ink2)}
"""


def trace_html(hist, cfg, nxt, run_dir: Path, out: Path):
    b, hw = cfg["budget"], cfg.get("hardware", {})
    best = max((h for h in hist if h.get("decision") in ("baseline", "accepted")), key=lambda h: h["iter"])
    head = (f"<h1>Automated research trace · {esc(run_dir.name)}</h1>"
            f"<p class='meta'>FML-bench Representation_Learning_solo_learn · {b['pretrain_epochs']} pretrain epochs "
            f"(upstream 130) · linear eval {b['linear_eval_epochs']} ep · split val_ratio 0.057 seed 42 · "
            f"LLM {esc(cfg['llm']['backend'])} {esc(cfg['llm']['model'])} · GPU {esc(hw.get('gpu'))} · "
            f"torch {esc(hw.get('torch'))}<br>Best: iter {best['iter']} at {100*best['val_acc']:.2f}% · "
            f"{sum(1 for h in hist if h['kind']=='candidate')} candidates · selection on val only, hidden test never run by the loop"
            f"{' · <b>FAKE executor (pipeline test, numbers are random)</b>' if cfg.get('fake_executor') else ''}</p>"
            "<div class='legend'><span><i class='sw' style='background:var(--llm)'></i>LLM decision</span>"
            "<span><i class='sw' style='background:var(--prog)'></i>Program check (deterministic code)</span></div>"
            "<img src='progress.png' alt='Validation accuracy per iteration with best-so-far line'>")
    body = head + "".join(card(h, run_dir) for h in hist) + next_card(nxt, run_dir)
    out.write_text(f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                   f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>Research loop trace</title>"
                   f"<style>{CSS}</style></head><body><main>{body}</main></body></html>")


def main():
    run_dir = Path(sys.argv[1])
    hist, cfg, nxt = load(run_dir)
    viz = run_dir / "viz"
    viz.mkdir(exist_ok=True)
    progress_png(hist, cfg, viz / "progress.png")
    trace_html(hist, cfg, nxt, run_dir, viz / "trace.html")
    print(f"wrote {viz/'progress.png'} and {viz/'trace.html'}")


if __name__ == "__main__":
    main()
