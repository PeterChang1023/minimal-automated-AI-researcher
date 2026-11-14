#!/usr/bin/env bash
# Download CIFAR-100 once to /cache (hinton rule: datasets live under /cache/$USER, not $HOME).
set -euo pipefail
DATA_ROOT="${1:-/cache/$USER/air/data}"
mkdir -p "$DATA_ROOT"
python - "$DATA_ROOT" <<'PY'
import sys, torchvision
root = sys.argv[1]
torchvision.datasets.CIFAR100(root=root, train=True, download=True)
torchvision.datasets.CIFAR100(root=root, train=False, download=True)
print("CIFAR-100 ready in", root)
PY
du -sh "$DATA_ROOT"
