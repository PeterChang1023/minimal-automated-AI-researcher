#!/usr/bin/env bash
# Print the hardware / software facts reported in the README (Part 1).
hostname; nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
python -c "import torch,torchvision,sys;print('python',sys.version.split()[0],'torch',torch.__version__,'tv',torchvision.__version__,'cuda',torch.version.cuda,'cudnn',torch.backends.cudnn.version(),'gpu_ok',torch.cuda.is_available())"
lscpu | grep -E 'Model name|^CPU\(s\)'; free -g | head -2
