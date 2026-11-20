# flash35-e130-20iter-explore — disclosure

Extension of `runs/flash35-e130-10iter-explore` (experimental, not the main result) from 10 to 20
executed experiments. **Iterations are renumbered 0–20** (0 = baseline); see the mapping below.

* **Same as the 10-iter explore run:** `--prompt-variant explore`, gemini-3.5-flash, 130 pretrain /
  100 linear epochs, seed 42, same split and eval, epoch/optimizer locked, `--gpu-mem-gb 24`.
  Iterations 0–10 here are the 10-iter explore run's experiments (there numbered 0–12 with gaps).
* **Changed for iterations 11–20:** per-experiment time limit raised from 60 to 120 min
  (`--timeout-min 120`, `--max-hours 30`); Gemini's prompt reports that limit. The longest kept run
  took 51.5 min, so all kept runs would also fit the old 60-min limit.
* **Removed and renumbered:** the LLM (Gemini API) failed 4 times (503 high demand / 429 free-tier daily
  quota; no training happened), at original iterations 5, 8, 18, 21. Those were removed
  (`scripts/drop_failed_iters.py`) and the remaining iterations renumbered 0–20. Iteration 11 (original 13)
  first exceeded the 60-min limit; it was re-run after the limit was raised. One aborted start (API key
  variable set to a placeholder; no request sent) and the final next-proposal call (503) were also removed.
  8 iterations were accepted. Raw logs and backups are on hinton (`~/air_backups/`, `explore20_raw.tgz`).
* **Gemini saw the original numbers.** Prompts, Gemini's responses/rationales, and code comments inside
  `iter_*/` (`prompt_user.txt`, `llm_response.txt`, `proposal.json`, patches) still use the original
  numbering. Directory names, `history.jsonl`, `record.json` (`iter`, `parent_iter`), `best/meta.json`,
  the harness log lines and the review page use the new numbering.

  | original | 0–4 | 6 | 7 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 19 | 20 | 22 | 23 | 24 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | **new** | 0–4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |

* **API keys:** free tier (20 requests/day, retries count); free keys of several Google accounts.
* **Capacity and probe input not held fixed:** iter 3 widened the backbone (ablation:
  58.09 % at width 1.0x vs 59.88 %); from iter 17 the backbone returns concatenated multi-stage
  features, so the linear probe input grows 512 → 1152 (iter 17) → 2304 (19) → 4032 (20). The probe
  is still one `nn.Linear` trained by the unchanged official script, but much of the late gain comes
  from giving it more features. Peak GPU memory 15.9 GB.
* **Result:** baseline 46.77 % → 61.44 % after 10 experiments (iter 10) → 69.22 % after 20 (iter 20).
  Main run: 47.27 % → 57.73 %. Single seed.
