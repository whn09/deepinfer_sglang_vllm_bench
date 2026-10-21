#!/bin/bash
# 停掉本评测起的 server 容器，释放 GPU。跑 DeepInfer 之前必须先停。
docker rm -f bench-sglang bench-vllm 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used --format=csv
