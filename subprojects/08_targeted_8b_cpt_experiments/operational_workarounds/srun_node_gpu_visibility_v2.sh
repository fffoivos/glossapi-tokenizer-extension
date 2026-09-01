#!/usr/bin/env bash
set -euo pipefail

# An allocation requested as four one-GPU tasks per node inherits
# gres/gpu:per_task:1. The canonical torchrun path deliberately starts one
# controller task per node, so that controller must own all four local GPUs.
# These exact flags were probed on both nodes of allocation 3141832 and exposed
# CUDA_VISIBLE_DEVICES=0,1,2,3 with torch.cuda.device_count() == 4.
exec /usr/bin/srun --gpus-per-node=4 --gpus-per-task=4 --gpu-bind=none "$@"
