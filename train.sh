#!/bin/bash
# Launch training with venv nvidia libs prepended to LD_LIBRARY_PATH.
# This prevents dlopen inside libcudnn_graph.so.9 from loading the system
# CUDA 12.9 libcublasLt instead of the CUDA 12.8 version bundled with torch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NVIDIA="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/nvidia"

export LD_LIBRARY_PATH=\
"$VENV_NVIDIA/cublas/lib:"\
"$VENV_NVIDIA/cudnn/lib:"\
"$VENV_NVIDIA/cuda_runtime/lib:"\
"$VENV_NVIDIA/cufft/lib:"\
"$VENV_NVIDIA/curand/lib:"\
"$VENV_NVIDIA/cusolver/lib:"\
"$VENV_NVIDIA/cusparse/lib:"\
"$VENV_NVIDIA/nccl/lib:"\
"${LD_LIBRARY_PATH:-}"

source "$SCRIPT_DIR/.venv/bin/activate"
exec python3 -m train.train "$@"