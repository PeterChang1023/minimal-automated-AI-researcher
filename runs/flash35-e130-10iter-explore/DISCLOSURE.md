# flash35-e130-10iter-explore — disclosure

Experimental variant of `runs/flash35-e130-10iter` (the main result). Not the main result.

* **Only difference:** `--prompt-variant explore` (air/prompts.py): lists structural directions
  (other SSL objectives, encoder architecture), reports measured wall-clock per run vs the 60-min
  limit, plateau rule (3 results within ±1.2 pt → next must be structural; it never triggered).
  Same model (gemini-3.5-flash), budget (130 pretrain / 100 linear epochs), seed 42, split, eval.
  Added `--gpu-mem-gb 24` (shared GPU); peak use was 15.9 GB.
* **Removed iterations:** every iteration whose Gemini call failed (status `llm_error`, nothing
  executed) was removed with `scripts/drop_failed_iters.py` before resuming, so numbering skips
  5 and 8 (earlier attempts at iters 1–7, 10 and 12 had also failed before execution and were removed the same way). Backups with
  the raw logs are kept outside the repo on hinton (`~/air_backups/`, `~/explore_failed_backup*`).
  Iterations 6, 7 and 9 were proposed while those failure records were still in the history Gemini saw.
* **API keys:** the free tier allows 20 requests/day and 503 retries count; the run used free keys
  of three Google accounts over 2026-11-17/18.
* **Capacity not held fixed:** iter 3 widened the backbone 1.5x (768-d features). Ablation
  `runs/flash35-e130-10iter-explore_ablation_width1x` (iter 3 code, width 1.0x): 58.09 % vs 59.88 %.
  Later bests are deeper/wider still; wall-clock per run rose from ~18 to ~40 min.
* **Result:** baseline 46.77 % → best 61.44 % (iter 12), vs main run 47.27 % → 57.73 %. Single seed.
