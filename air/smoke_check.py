"""Cheap CPU interface check of a candidate algorithm.py (run as a subprocess
inside the candidate sandbox, before any GPU time is spent).

Checks exactly what the official train_eval_baseline.py relies on:
  * get_model() -> module with .backbone, .projector, .backbone_dim; forward returns (features, projections)
  * features have shape [B, backbone_dim]; projections have shape [B, 2048]
    (the official script builds BarlowTwinsLoss(..., embedding_dim=2048))
  * BarlowTwinsLoss(lambda_param=..., embedding_dim=2048)(z1, z2) is a finite scalar and backprops
  * get_augmentation()(PIL 32x32 image) -> float tensor [3, 32, 32]
  * get_training_config() has the keys the official script reads
Prints one JSON line on success; exits non-zero with a message on failure.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())

REQUIRED_CFG_KEYS = ["pretrain_epochs", "linear_eval_epochs", "batch_size", "lr",
                     "weight_decay", "lambda_param", "linear_lr", "linear_batch_size"]


def fail(msg):
    print(f"SMOKE_FAIL: {msg}")
    sys.exit(1)


def main():
    import numpy as np
    import torch
    from PIL import Image

    torch.manual_seed(0)
    try:
        import algorithm
    except Exception as e:  # noqa: BLE001
        fail(f"import algorithm failed: {type(e).__name__}: {e}")

    for name in ["get_model", "get_training_config", "get_augmentation", "BarlowTwinsLoss"]:
        if not hasattr(algorithm, name):
            fail(f"algorithm.py must export {name}")

    cfg = algorithm.get_training_config()
    missing = [k for k in REQUIRED_CFG_KEYS if k not in cfg]
    if missing:
        fail(f"get_training_config() missing keys: {missing}")

    model = algorithm.get_model()
    for attr in ["backbone", "projector", "backbone_dim"]:
        if not hasattr(model, attr):
            fail(f"model has no attribute .{attr}")
    n_params = sum(p.numel() for p in model.parameters())

    aug = algorithm.get_augmentation()
    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8"))
    x = aug(img)
    if not (torch.is_tensor(x) and tuple(x.shape) == (3, 32, 32)):
        fail(f"augmentation must return a [3,32,32] tensor, got {getattr(x, 'shape', type(x))}")

    B = 8
    model.train()
    xb1 = torch.stack([aug(img) for _ in range(B)])
    xb2 = torch.stack([aug(img) for _ in range(B)])
    out = model(xb1)
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        fail("model.forward must return (features, projections)")
    f1, z1 = out
    if tuple(f1.shape) != (B, model.backbone_dim):
        fail(f"features shape {tuple(f1.shape)} != ({B}, backbone_dim={model.backbone_dim})")
    if tuple(z1.shape) != (B, 2048):
        fail(f"projections shape {tuple(z1.shape)} != ({B}, 2048); the official script uses embedding_dim=2048")
    _, z2 = model(xb2)

    crit = algorithm.BarlowTwinsLoss(lambda_param=cfg["lambda_param"], embedding_dim=2048)
    loss = crit(z1, z2)
    if not (torch.is_tensor(loss) and loss.dim() == 0 and torch.isfinite(loss)):
        fail(f"loss must be a finite scalar tensor, got {loss}")
    loss.backward()

    model.eval()
    with torch.no_grad():
        f_eval, _ = model(xb1)
    if tuple(f_eval.shape) != (B, model.backbone_dim):
        fail("eval-mode features have wrong shape")

    print(json.dumps({"smoke_ok": True, "n_params": n_params, "loss": float(loss),
                      "candidate_config": {k: cfg[k] for k in cfg}}, default=str))


if __name__ == "__main__":
    main()
