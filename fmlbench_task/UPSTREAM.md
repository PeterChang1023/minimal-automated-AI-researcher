# Vendored FML-bench task: `Representation_Learning_solo_learn`

Source: https://github.com/qrzou/FML-bench (Apache-2.0, see `LICENSE.FML-bench`)
Commit: `d336651ebea50c622c256f02ded82b68b4451fdc`
Path:   `ml_tasks/Representation_Learning_solo_learn/`

Files are copied byte-for-byte. `SHA256SUMS` pins them; the loop refuses to run
if `train_eval_baseline.py` (the official evaluation) or `split_config.json`
(the train/val split) no longer match.

| file | role | who may change it |
|---|---|---|
| `algorithm.py` | baseline Barlow Twins (model, loss, augmentation, config) | the LLM edits a **copy** per candidate; this file is never modified |
| `train_eval_baseline.py` | official pretrain + linear-probe evaluation | nobody (hash-checked) |
| `split_config.json` | train/val split (`val_ratio=0.057`, `val_seed=42`) | nobody (fixed so val accuracies are comparable) |
| `prompt.json`, `config.json` | upstream task description and metric | read only |
| `upstream_baseline_results/` | upstream reported numbers (full 130-epoch budget): val 0.4777, test 0.4896 | read only |

Note: the upstream task repo (`vturrisi/solo-learn@b69b4bd`) is cloned by FML-bench's
`setup.py`, but neither `algorithm.py` nor `train_eval_baseline.py` imports it.
The evaluation only needs PyTorch + torchvision + CIFAR-100, so we do not clone it.
