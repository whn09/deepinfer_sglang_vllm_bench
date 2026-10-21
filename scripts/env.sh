# 在 B200 上 source 这个文件：DeepInfer 编译/运行环境 + 本评测用到的路径。
# 机器不固定：所有路径都在这里集中定义，换机器只改这里。
#   source scripts/env.sh

# ---- 路径 ----
export BENCH_ROOT="${BENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export DEEPINFER_ROOT="${DEEPINFER_ROOT:-$HOME/DeepInfer}"
export MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/DeepSeek-V4.1-Flash-1017}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-ai/DeepSeek-V4.1-Flash-1017}"
# 数据集和结果都放 nvme 上的临时目录；跑完用 scripts/pull_results.sh 拉回本地再删
export WORK="${WORK:-/opt/dlami/nvme/bench_tmp}"
export DATA_DIR="${DATA_DIR:-$WORK/data}"
export RESULTS_DIR="${RESULTS_DIR:-$WORK/results}"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/hf}"

# ---- DeepInfer 环境（见 docs/setup_deepinfer.md）----
source /opt/pytorch/bin/activate
export CUDA_HOME=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CUDA_HOME/include/cccl:${CPLUS_INCLUDE_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$CUDA_HOME/lib:${LIBRARY_PATH:-}
export DEEPINFER_ACCELERATOR=nvidia
export FLASH_MLA_SKIP_REG_SPILL_CHECK=1
export DEEPINFER_HF_MODEL_DIR="$MODEL_DIR"
hash -r

mkdir -p "$DATA_DIR" "$RESULTS_DIR"
