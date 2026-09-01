#!/usr/bin/env bash
#SBATCH --job-name=r1_roundtrip
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/r1_roundtrip_%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/r1_roundtrip_%j.err
#
# R1: HF -> Megatron -> HF roundtrip on Apertus-8B-2509.
# Validates loader_apertus_hf + saver_swissai_hf preserve all non-R17 weights
# bit-exactly, and measures R17 drift (xIELU + QK-Norm reset through saver_core
# protocol).
#
# Passed 2026-05-21 (job 2333864): standard max abs diff = 0.0, R17 changed = 128.
# Result: 32 layers x 4 xIELU params = 128. Standard weights are bit-identical.
# Confirms the bakeoff plan's two-V4-run design (V4-HF + V4-post-conversion).
#
# Compute justification (per [[feedback_compute_sweet_spot_justify]]):
#
# 1. Parallelism strategy.
#    The convert.py protocol is loader-process <-> saver-process via multi-
#    processing Queue (Megatron's own send/recv). Per-layer streaming is
#    inherently serial — loader sends layer N, saver writes layer N, then
#    layer N+1. No data parallelism within the tool.
#
# 2. GPU saturation.
#    Both legs use ONE H100 (load HF in bf16 ≈ 16 GB, send to saver, GPU is
#    mostly idle during the IO-heavy save). --gpus-per-node=1 saturates the
#    bottleneck (the HF load_pretrained call); more GPUs would idle.
#
# 3. CPU + memory ceiling.
#    --cpus-per-task=72 (one socket) handles the two python processes plus
#    pyarrow IO comfortably. --mem=400G covers two full state_dicts in RAM
#    (orig + roundtrip for the two-tier diff at the end ≈ 35 GB) plus
#    Megatron's per-layer working set; the rest is headroom for the
#    `from_pretrained` peak allocation pattern in transformers >=4.50.
#
# 4. Known efficiency gaps.
#    convert.py is single-process per leg by design. No tensor parallelism
#    in the loader (we run TP=1, PP=1 for the bakeoff). Wallclock ≈ 5-10 min
#    end-to-end on an H100; well within the 1 h debug-partition cap.

set -euo pipefail
APERTUS_HF=/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509
MEGATRON_DIR=/iopsstor/scratch/cscs/fffoivos/code/training/Megatron-LM-Swiss-AI
WORK=/capstor/scratch/cscs/fffoivos/runs/r1_roundtrip_$SLURM_JOB_ID
mkdir -p "$WORK"

uenv start pytorch/v2.9.1:v2 --view=default --ignore-tty <<INNER
set -euo pipefail
export PYTHONPATH=$MEGATRON_DIR
cd $MEGATRON_DIR

# --loader-transformer-impl transformer_engine: needed for validate_args
# in saver_core and loader_core, because Apertus has qknorm_impl=apex which
# triggers the OP-args assertion when transformer_impl != transformer_engine
# (megatron/training/arguments.py:811).
echo "=== HF -> Megatron ==="
python3 tools/checkpoint/convert.py --model-type GPT \
    --loader apertus_hf --saver core \
    --load-dir $APERTUS_HF --save-dir $WORK/apertus_megatron \
    --tokenizer-model $APERTUS_HF --bf16 \
    --loader-transformer-impl transformer_engine

# saver_core writes iter_0000000 + latest_checkpointed_iteration.txt='0',
# but loader_core's read_metadata asserts iteration > 0 OR file contents == 'release'.
# Convert to a release checkpoint between legs.
echo "=== Mark Megatron checkpoint as release ==="
mv $WORK/apertus_megatron/iter_0000000 $WORK/apertus_megatron/release
echo release > $WORK/apertus_megatron/latest_checkpointed_iteration.txt
ls -la $WORK/apertus_megatron/

echo "=== Megatron -> HF (back) ==="
python3 tools/checkpoint/convert.py --model-type GPT \
    --loader core --saver swissai_hf \
    --load-dir $WORK/apertus_megatron --save-dir $WORK/apertus_hf_roundtrip \
    --hf-tokenizer $APERTUS_HF \
    --loader-transformer-impl transformer_engine

echo "=== Two-tier diff ==="
python3 - <<PYEOF
import torch, re
from transformers import AutoModelForCausalLM
orig = AutoModelForCausalLM.from_pretrained("$APERTUS_HF", torch_dtype=torch.bfloat16)
trip = AutoModelForCausalLM.from_pretrained("$WORK/apertus_hf_roundtrip", torch_dtype=torch.bfloat16)
R17 = re.compile(r"(act_fn\.(alpha_p|alpha_n|beta|eps)|self_attn\.(q_norm|k_norm)\.weight)")
QK = re.compile(r"self_attn\.(q_norm|k_norm)\.weight")
ok = set(orig.state_dict().keys()); tk = set(trip.state_dict().keys())
print("orig only:", sorted(ok - tk))
print("trip only:", sorted(tk - ok))
std_max, r17, r17_max, qk_max = 0.0, 0, 0.0, 0.0
for k in sorted(ok & tk):
    a, b = orig.state_dict()[k].float(), trip.state_dict()[k].float()
    if a.shape != b.shape: print("SHAPE MISMATCH:", k); continue
    d = (a-b).abs().max().item()
    if R17.search(k):
        r17_max = max(r17_max, d)
        if QK.search(k): qk_max = max(qk_max, d)
        if d > 1e-3: r17 += 1
    else:
        std_max = max(std_max, d)
        if d > 1e-3: print("DRIFT (std):", k, d)
print("standard max abs diff:", std_max)
print("R17 changed:", r17)
print("R17 max abs diff:", r17_max)
print("QK norm max abs diff:", qk_max)
print("PASS" if std_max < 1e-3 else "FAIL")
PYEOF
INNER
