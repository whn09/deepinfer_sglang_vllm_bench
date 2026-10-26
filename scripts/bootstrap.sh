#!/bin/bash
# Fresh B200 (DLAMI PyTorch 2.12, Ubuntu 24.04) bootstrap for deepinfer_sglang_vllm_bench. Idempotent.
set -u
W=/opt/dlami/nvme/bench_tmp; M=/opt/dlami/nvme/models/DeepSeek-V4.1-Flash-1017
mkdir -p $W/logs $W/data $W/results $W/cache /opt/dlami/nvme/hf
# 1. no unattended upgrades / needrestart service restarts (they restarted Fabric Manager mid-run before)
sudo systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer > /dev/null 2>&1
echo '$nrconf{restart} = "l";' | sudo tee /etc/needrestart/conf.d/99-no-auto.conf > /dev/null
# 2. model weights (public, 510 GB)
/opt/pytorch/bin/pip install -q "huggingface_hub[hf_xet]" > /dev/null 2>&1
if [ ! -f $M/.done ]; then
    ( HF_HOME=/opt/dlami/nvme/hf HF_XET_HIGH_PERFORMANCE=1 /opt/pytorch/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
        --local-dir $M --max-workers 32 > $W/logs/hf_download.log 2>&1 && touch $M/.done ) &
fi
# 3. DeepInfer base image from ECR (same region)
REG=579019700964.dkr.ecr.ap-south-2.amazonaws.com
aws ecr get-login-password --region ap-south-2 | docker login --username AWS --password-stdin $REG > /dev/null 2>&1
( docker pull -q $REG/deepinfer:efa-774ce24-deepep-b5a7378 && docker tag $REG/deepinfer:efa-774ce24-deepep-b5a7378 deepinfer:efa-b5a7378 ) > $W/logs/docker_pull.log 2>&1 &
# 4. InferenceX (metrics code + recipes)
[ -d $W/InferenceX ] || git clone -q https://github.com/SemiAnalysisAI/InferenceX.git $W/InferenceX
[ -e $W/inferencex-e2e ] || ln -s $W/InferenceX/inferencex-e2e $W/inferencex-e2e
# 5. AgentX harness venv (Python 3.13 from /opt/pytorch)
V=$W/agentx-venv-1.0.12
[ -x $V/bin/agentx ] || { /opt/pytorch/bin/python -m venv $V && $V/bin/pip install -q "agentx-harness==1.0.12" "datasets>=4.7.0"; } > $W/logs/agentx_venv.log 2>&1
wait
echo "BOOTSTRAP_DONE $(date +%H:%M:%S) model=$( [ -f $M/.done ] && echo ok || echo pending ) image=$(docker images -q deepinfer:efa-b5a7378) agentx=$( [ -x $V/bin/agentx ] && echo ok )"
