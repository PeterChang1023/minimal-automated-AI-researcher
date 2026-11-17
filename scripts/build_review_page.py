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
import ast
import difflib
import json
import re
from collections import Counter
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


# ---------- key-change extraction: turn a code diff into short "what changed" chips ----------
KW_RE = re.compile(r"""['"]?\b([A-Za-z_]\w*)['"]?\s*(?<![=!<>])[=:](?!=)\s*(\([^()]*\)|\[[^\[\]]*\]|[^,()\[\]{}]+)""")
CALL_RE = re.compile(r"\b(?:nn|transforms|T|F|torch)\.([A-Z]\w+)\s*\(")


def _noise_lines(src: str):
    """Line numbers (1-based) that are docstrings, comments or blank: changes there are not code changes."""
    out = set()
    for i, ln in enumerate(src.splitlines(), 1):
        if not ln.strip() or ln.strip().startswith("#"):
            out.add(i)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            out.update(range(body[0].lineno, body[0].end_lineno + 1))
    return out


def _scopes(src: str):
    """Map line number -> 'Class.method' (innermost def/class around it)."""
    names = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return names

    def visit(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}.{ch.name}" if prefix else ch.name
                for ln in range(ch.lineno, ch.end_lineno + 1):
                    names[ln] = q
                visit(ch, q)
    visit(tree, "")
    return names


def _code(line: str):
    return re.sub(r"\s+", "", line.split("#")[0])


def _kwargs(line: str):
    """keyword arguments `f(name=value)` and dict entries `'name': value` (not plain assignments)."""
    line = line.split("#")[0]
    out = {}
    for m in KW_RE.finditer(line):
        before = line[:m.start()].rstrip()
        quoted = m.group(0).lstrip()[:1] in "'\""
        if quoted or before.endswith(("(", ",", "{")):
            out[m.group(1)] = m.group(2).strip()
    return out


def _nums(line: str):
    return re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?:e-?\d+)?", line.split("#")[0])


def key_changes(patch: str, old_src: str, new_src: str):
    """Return [{scope, text}] — parameter changes as 'name old → new', component adds/removals,
    otherwise 'rewritten (+a −d lines)'. Docstring / comment edits are ignored."""
    if not patch.strip():
        return []
    skip_old, skip_new = _noise_lines(old_src), _noise_lines(new_src)
    sc_old, sc_new = _scopes(old_src), _scopes(new_src)
    blocks, buf, a, b = [], ([], []), 0, 0
    for ln in patch.splitlines():
        h = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", ln)
        if h:
            if buf[0] or buf[1]: blocks.append(buf)
            buf, a, b = ([], []), int(h.group(1)) - 1, int(h.group(2)) - 1
        elif ln.startswith(("---", "+++")):
            continue
        elif ln.startswith("-"):
            a += 1
            if a not in skip_old: buf[0].append((a, ln[1:]))
        elif ln.startswith("+"):
            b += 1
            if b not in skip_new: buf[1].append((b, ln[1:]))
        elif ln.startswith(" "):
            a += 1; b += 1
            if buf[0] or buf[1]: blocks.append(buf); buf = ([], [])
    if buf[0] or buf[1]: blocks.append(buf)

    per_scope = {}  # scope -> {"items": [...], "add": n, "del": n}
    for dels, adds in blocks:
        scope = (sc_new.get(adds[0][0]) if adds else None) or (sc_old.get(dels[0][0]) if dels else None) or "module"
        s = per_scope.setdefault(scope, {"items": [], "add": 0, "del": 0, "explained": 0, "noop": 0})
        s["add"] += len(adds); s["del"] += len(dels)
        # 1) paired lines: keyword / dict-key value changes
        used_d, used_a = set(), set()
        for i, (_, dl) in enumerate(dels):
            for j, (_, al) in enumerate(adds):
                if j in used_a: continue
                if _code(dl) == _code(al):  # only a comment / spacing changed
                    used_d.add(i); used_a.add(j); s["noop"] += 2; break
                if difflib.SequenceMatcher(None, _code(dl), _code(al)).ratio() < 0.6:
                    continue
                kd, ka = _kwargs(dl), _kwargs(al)
                if not kd and not ka: continue
                if _nums(dl) == _nums(al) and all(k not in kd or kd[k] == ka[k] for k in ka) and not set(kd) - set(ka):
                    used_d.add(i); used_a.add(j); s["noop"] += 2; break  # same values, only spelled differently
                for k in kd:
                    if k in ka and kd[k] != ka[k]:
                        s["items"].append(f"{k}: {kd[k]} → {ka[k]}")
                for k in ka:
                    if k not in kd:
                        s["items"].append(f"新增 {k} = {ka[k]}")
                for k in kd:
                    if k not in ka:
                        s["items"].append(f"移除 {k}")
                used_d.add(i); used_a.add(j)
                break
        # 2) components added / removed (nn.X(...), transforms.X(...))
        rest_d = [l for i, (_, l) in enumerate(dels) if i not in used_d]
        rest_a = [l for j, (_, l) in enumerate(adds) if j not in used_a]
        cd = Counter(n for l in rest_d for n in CALL_RE.findall(l))
        ca = Counter(n for l in rest_a for n in CALL_RE.findall(l))
        for n, c in (cd - ca).items():
            s["items"].append(f"移除 {n}" + (f" ×{c}" if c > 1 else ""))
        for n, c in (ca - cd).items():
            extra = ""
            for l in rest_a:
                m = re.search(rf"\b{n}\s*\(([^()]*)\)", l)
                if m and "=" in m.group(1):
                    extra = "(" + ", ".join(f"{k}={v}" for k, v in _kwargs(m.group(1)).items()) + ")"
                    break
            s["items"].append(f"新增 {n}{extra}" + (f" ×{c}" if c > 1 else ""))
        explained = len(used_d) + len(used_a) + sum(1 for l in rest_d + rest_a if CALL_RE.search(l))
        s["explained"] += explained
    out = []
    for scope, s in per_scope.items():
        unexplained = s["add"] + s["del"] - s["explained"]
        items = list(dict.fromkeys(s["items"]))
        if s["add"] + s["del"] - s["noop"] <= 0:
            continue
        if unexplained > 3 or not items:
            items.append(f"改寫程式（+{s['add']} −{s['del']} 行）")
        for t in items:
            out.append({"scope": scope, "text": t})
    return out


# ---------- automatic checks: did a candidate touch anything that must stay fixed? ----------
REPO = Path(__file__).resolve().parents[1]


def _pinned_sha():
    out = {}
    for ln in read(REPO / "fmlbench_task" / "SHA256SUMS").splitlines():
        parts = ln.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def _static_check(code: str):
    try:
        import sys
        sys.path.insert(0, str(REPO))
        from air.checks import static_check
        return static_check(code)
    except Exception as e:  # keep the page buildable without the package
        return None


def _split_line(stdout: str):
    m = re.search(r"Agent train: \d+, Agent val: \d+.*", stdout)
    return m.group(0).strip() if m else ""


def run_checks(d: Path, rec: dict, run_budget: dict, base_info: dict):
    """List of {name, status: ok|warn|fail|info, detail}. Everything comes from files in d."""
    pins = _pinned_sha()
    fp = rec.get("fingerprint") or {}
    cfg = json.loads(read(d / "effective_config.json") or "{}")
    eff, cand = cfg.get("effective_config") or {}, cfg.get("candidate_config") or {}
    smoke = {}
    for ln in read(d / "smoke.log").splitlines():
        if ln.startswith("{"):
            try: smoke = json.loads(ln)
            except ValueError: pass
    code = read(d / "algorithm.py")
    out = []

    def add(name, ok, detail, warn=False, info=False):
        out.append({"name": name, "status": "info" if info else ("ok" if ok else ("warn" if warn else "fail")), "detail": detail})

    want = pins.get("train_eval_baseline.py")
    got = fp.get("eval_sha256")
    add("評估程式沒被改", bool(want) and got == want,
        f"train_eval_baseline.py SHA-256 {str(got)[:12]}… {'＝' if got == want else '≠'} 鎖定值 {str(want)[:12]}…")
    want, got = pins.get("split_config.json"), fp.get("split_sha256")
    sl = _split_line(read(d / "stdout.log"))
    same_split = sl == base_info.get("split_line") and bool(sl)
    add("資料切分沒被改", bool(want) and got == want and same_split,
        f"split_config.json SHA-256 {'＝' if got == want else '≠'} 鎖定值；這輪實際切分「{sl or '找不到'}」"
        + ("，和 baseline 相同" if same_split else "，和 baseline 不同"))
    keys = ["pretrain_epochs", "linear_eval_epochs", "linear_lr", "linear_batch_size"]
    bad = [k for k in keys if k in run_budget and eff.get(k) != run_budget[k]]
    tried = [f"{k}={cand.get(k)}" for k in keys if k in cand and k in run_budget and cand.get(k) != run_budget[k]]
    if bad:
        add("訓練預算固定", False, "實際設定和預算不同：" + "、".join(f"{k}={eff.get(k)}" for k in bad))
    elif tried:
        add("訓練預算固定", False, "Gemini 想改 " + "、".join(tried) + "，已被程式覆蓋回預算值", warn=True)
    else:
        add("訓練預算固定", True, f"pretrain {eff.get('pretrain_epochs')} ep · linear eval {eff.get('linear_eval_epochs')} ep · linear_lr {eff.get('linear_lr')} · linear batch {eff.get('linear_batch_size')}")
    mbs = run_budget.get("max_bs") or fp.get("max_batch_size")
    bs = eff.get("batch_size")
    add("Batch size 沒超過上限", bs is not None and mbs is not None and bs <= mbs, f"batch_size {bs}（上限 {mbs}）")
    code_nc = "\n".join(l.split("#")[0] for l in code.splitlines())
    has_opt = re.search(r"torch\.optim|optim\.(SGD|Adam\w*|LARS|LBFGS)|lr_scheduler", code_nc)
    add("Optimizer 沒被換掉", not has_opt,
        "algorithm.py 裡沒有 optimizer / scheduler（固定用官方程式的 SGD + cosine）" if not has_opt else f"algorithm.py 出現 `{has_opt.group(0)}`")
    probs = _static_check(code) if code else None
    rec_probs = rec.get("static_problems") or []
    if probs is None:
        add("靜態檢查（禁止的寫法）", not rec_probs, "；".join(rec_probs) or "loop 當時的檢查沒有問題（本機無法重跑 air.checks）")
    else:
        add("靜態檢查（禁止的寫法）", not probs and not rec_probs,
            "沒有讀資料、讀測試集、連網、用預訓練權重、執行 shell 等寫法（重新跑 air/checks.py 確認）" if not probs and not rec_probs else "；".join(probs or rec_probs))
    vi = json.loads(read(d / "val_info.json") or "{}")
    v = ((vi.get("cifar100_ssl") or {}).get("final_info_dict") or {}).get("linear_eval_acc")
    if rec.get("val_acc") is not None:
        add("分數對得上原始輸出", v is not None and abs(v - rec["val_acc"]) < 1e-9,
            f"record.json {rec['val_acc']:.4f}，官方程式輸出的 val_info.json {v if v is None else f'{v:.4f}'}")
    n = smoke.get("n_params")
    if n:
        b = base_info.get("n_params")
        delta = "" if not b or n == b else f"（比 baseline {'+' if n > b else '−'}{abs(n - b) / 1e6:.2f}M）"
        add("模型參數量", True, f"{n / 1e6:.2f}M{delta or '（和 baseline 一樣）'}，含 projection head", info=True)
    return out


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
    # automatic checks
    run_budget = cfg.get("budget") or {}
    base_dir = run_dir / "iter_000"
    base_smoke = next((json.loads(l) for l in read(base_dir / "smoke.log").splitlines() if l.startswith("{")), {})
    base_info = {"split_line": _split_line(read(base_dir / "stdout.log")), "n_params": base_smoke.get("n_params")}
    for it in iters:
        d = run_dir / f"iter_{it['iter']:03d}"
        rec = json.loads(read(d / "record.json") or "{}")
        it["checks"] = run_checks(d, rec, run_budget, base_info)
    # key changes vs the parent (best at the time); docstring/comment edits ignored
    src = lambda k: read(run_dir / f"iter_{int(k):03d}" / "algorithm.py") if k is not None else ""
    for it in iters:
        if it["kind"] != "baseline":
            it["key_changes"] = key_changes(it["diff_best"], src(it["parent_iter"]), src(it["iter"]))
    if nxt:
        best_k = max((i for i in iters if i["decision"] in ("accepted", "baseline")), key=lambda i: i["iter"])["iter"]
        code = read(nd / "algorithm.py")
        code_nc = "\n".join(l.split("#")[0] for l in code.splitlines())
        probs = _static_check(code)
        has_opt = re.search(r"torch\.optim|optim\.(SGD|Adam\w*|LARS|LBFGS)|lr_scheduler", code_nc)
        nxt["checks"] = [
            {"name": "Optimizer 沒被換掉", "status": "fail" if has_opt else "ok",
             "detail": f"algorithm.py 出現 `{has_opt.group(0)}`" if has_opt else "algorithm.py 裡沒有 optimizer / scheduler"},
            {"name": "靜態檢查（禁止的寫法）", "status": "info" if probs is None else ("fail" if probs else "ok"),
             "detail": "本機無法重跑 air.checks" if probs is None else ("；".join(probs) or "沒有禁止的寫法（重新跑 air/checks.py 確認）")},
            {"name": "其他檢查", "status": "info", "detail": "這個提案還沒執行，評估程式、切分、分數的檢查要等跑完才有"},
        ]
        nxt["key_changes"] = key_changes(nxt["diff_best"], src(best_k), read(nd / "algorithm.py"))
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
