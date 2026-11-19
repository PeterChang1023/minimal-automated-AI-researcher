# flash35-e130-20iter-explore — disclosure

Extension of `runs/flash35-e130-10iter-explore` (experimental, not the main result) from 10 to 20
executed experiments. Iterations 0–12 are copied unchanged from that run.

* **Same as the 10-iter explore run:** `--prompt-variant explore`, gemini-3.5-flash, 130 pretrain /
  100 linear epochs, seed 42, same split and eval, epoch/optimizer locked, `--gpu-mem-gb 24`.
* **Changed for iterations 13–24:** per-experiment time limit raised from 60 to 120 min
  (`--timeout-min 120`, `--max-hours 30`); Gemini's prompt reports that limit. The longest kept run
  took 51.5 min, so all kept runs would also fit the old 60-min limit.
* **Removed iterations** (`scripts/drop_failed_iters.py --status llm_error,timeout`): every Gemini
  call failure (503 high demand, 429 free-tier daily quota, 402 on a billing-enabled key) and one
  first attempt at iteration 13 (ECA attention) that exceeded the 60-min limit. Numbering therefore
  skips 5, 8, 18 and 21. One aborted start (API key variable set to a placeholder; no request sent)
  and the final next-proposal call (503) were removed from the log. Raw logs and backups are on hinton
  (`~/air_backups/`, `explore20_raw.tgz`).
* **API keys:** free tier (20 requests/day, retries count); free keys of several Google accounts.
* **Capacity and probe input not held fixed:** iter 3 widened the backbone (ablation:
  58.09 % at width 1.0x vs 59.88 %); from iter 20 the backbone returns concatenated multi-stage
  features, so the linear probe input grows 512 → 1152 (iter 20) → 2304 (23) → 4032 (24). The probe
  is still one `nn.Linear` trained by the unchanged official script, but much of the late gain comes
  from giving it more features. Peak GPU memory 15.9 GB.
* **Result:** baseline 46.77 % → 61.44 % after 10 experiments (iter 12) → 69.22 % after 20 (iter 24).
  Main run: 47.27 % → 57.73 %. Single seed.
