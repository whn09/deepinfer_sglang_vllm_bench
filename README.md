# DeepInfer vs SGLang vs vLLM: DeepSeek-V4.1-Flash on B200

This repo compares three inference stacks serving `DeepSeek-V4.1-Flash-1017` on one 8×B200 box: same weights, same inputs, same metrics. The primary workload is the [InferenceX](https://inferencex.semianalysis.com/inference/deepseek-v41-flash?i_xmode=interactivity&i_seq=agentic-traces&i_prec=fp8) **AgentX** benchmark, and we first reproduce the published B200 points before comparing anything.

**Roadmap and status: [#8](https://github.com/whn09/deepinfer_sglang_vllm_bench/issues/8).** **Latest results: [`results/20261023_agentx/summary.md`](results/20261023_agentx/summary.md).**

## Results (AgentX, 4 GPUs per server, 3600 s per point)

![pareto](results/20261023_agentx/agentx_pareto.png)
![ttft](results/20261023_agentx/agentx_ttft.png)

- **The reproduction matches InferenceX.** Our vLLM TP4 runs match all 7 published points (throughput 0.95–1.02×). Our SGLang runs match all 12 published points (0.98–1.06×), including DEP4 + HiCache.
- **DeepInfer with the #147 server defaults was 2–5× behind** because EPLB rearrangement was hard-wired on (#5). It is fixed upstream (EPLB off by default).
- **DeepInfer TTFT was 3–4× vLLM's** because HF `tokenizers` re-encoded the whole conversation on every turn (#17). With `fastokens`, as vLLM's Rust frontend uses, it is within 10–40 ms of vLLM. This is now upstream in #147 as `--tokenizer-backend`.
- **DeepInfer today**, vs vLLM TP4:
  - throughput 0.97–0.98× at c ≥ 16;
  - interactivity 0.62–0.76× at c ≤ 16 (DP attention gives one request one GPU) and 0.92–0.95× at c ≥ 64.
- **SGLang** has the highest throughput at high concurrency, but its TTFT is 1.3–1.6 s on the DEP4 points.
- **DeepInfer PD 4P+4D** (8 GPUs, #18) gives the best per-user speed at c ≥ 64, at about half the throughput per GPU.

## Status

| Track | Issue | State |
|---|---|---|
| Reproduce InferenceX AgentX (vLLM, SGLang) | #4 | done (19/19 points) |
| DeepInfer on AgentX (OpenAI server, routing, prefix cache, synthetic acceptance) | #5 | done; headline `deepinfer-ix-ep4-noeplb-fastok` |
| Fixed-length closed-loop comparison | #6 | sg-1k/8k/32k done |
| DeepInfer TTFT breakdown and fix | #17 | done, upstreamed |
| YoCo truncation (lossless?) | #16 | keep=140 changes greedy outputs; quality eval open |
| PD split 4P+4D, Mooncake NVLink transport | #18 | done: Mooncake = TCP (identical outputs, same perf); #214 ready for review |
| Mooncake Store as L3 under the prefix cache | #20 | correct per request; TRANSFER_FAIL under multi-process TCP load, investigating |
| Mooncake `nvlink_intra` VMM buffers | #19 | brief for a separate agent |
| Final report | #7 | interim snapshot |

## How DeepInfer runs here

DeepInfer's open-source release is offline-only, so the AgentX runs use upstream PRs plus a few patches of ours. All of them are in [`patches/`](patches), and each is built as a Docker image ([`docker/`](docker)):

| Image | Contents |
|---|---|
| `deepinfer:efa-b5a7378` | DeepInfer @ 774ce24 + EFA installer + aws-ofi-nccl + `whn09/DeepEP@b5a7378` (Engram over NCCL GIN on EFA, #10) |
| `deepinfer:agentx-v2` | + upstream #144 (host prefix cache) and #147 (OpenAI server, EPLB off, `--tokenizer-backend`) + ours: synthetic acceptance length (InferenceX golden AL), sticky-balanced DP routing, per-request TTFT tracing |
| `deepinfer:agentx-l3` | + Mooncake Store L3 under the prefix cache (#20, upstream draft #216) |
| `deepinfer:pd` | upstream #195 + #197 (PD split) + Mooncake Transfer Engine KV transport (#18, upstream draft #214) |

## Reproduce

The B200 host is not fixed. Everything lives under `/opt/dlami/nvme/bench_tmp` (`$WORK`, set in `scripts/env.sh`). `scripts/sync_from_host.sh` pulls results back periodically; delete the remote directory when done.

```bash
rsync -av --exclude results --exclude .git ./ B200-1:/opt/dlami/nvme/bench_tmp/repo/   # local -> host
ssh B200-1 && cd /opt/dlami/nvme/bench_tmp/repo && source scripts/env.sh

# AgentX, vLLM / SGLang: one fresh server per concurrency from InferenceX's own recipe (bench/ix_recipe.py)
bash scripts/serve_ix.sh vllm 64
IX_JSON=$WORK/logs/ix_vllm_c64.json bash scripts/run_agentx.sh vllm vllm-ix-tp4 4 tp4 64
bash scripts/serve_ix.sh sglang 128                    # DEP4 + HiCache + sglang-router, as published
IX_JSON=$WORK/logs/ix_sglang_c128.json bash scripts/run_agentx.sh sglang sglang-ix 4 dep4 128

# AgentX, DeepInfer (EP4 = 4 DP ranks, synthetic AL 3.51, host prefix cache)
DI_IMAGE=deepinfer:agentx-v2 bash scripts/serve_deepinfer_agentx.sh          # HALF=b for GPUs 4-7
PUBLISHED_AS=vllm bash scripts/run_agentx.sh deepinfer deepinfer-ix-ep4 4 tp4 64

# PD split, PE on GPUs 0-3 + CE on GPUs 4-7 (DI_KV_TRANSPORT=tcp|mooncake)
DI_KV_TRANSPORT=mooncake bash scripts/serve_deepinfer_pd.sh
python3 bench/pd_parity.py --port 8888 --out mooncake.json                  # greedy output fingerprints

# Prefix cache with a Mooncake Store L3
bash scripts/serve_mooncake_store.sh && DI_IMAGE=deepinfer:agentx-l3 DI_L3=1 bash scripts/serve_deepinfer_agentx.sh

# Report: tables + Pareto and TTFT-by-concurrency plots (InferenceX's own metric code)
python bench/agentx_report.py results/b200_live/results/agentx --series "DeepInfer=deepinfer-ix-ep4-noeplb-fastok:dep4" \
    --published "vLLM (InferenceX)=results/inferencex_public/dsv41flash_b200_vllm_agentic.json" --out-dir results/<run>
```

Fixed-length closed loop (`docs/fairness.md`): `scripts/sweep_deepinfer.sh` drives the offline engine, and `serve_sglang.sh` / `serve_vllm.sh` + `sweep_server.sh` drive the servers. Scenarios are in `scripts/scenarios.sh`, and `bench/summarize.py` builds the tables.

**Host pitfalls we hit.**
- `--privileged` containers ignore `--gpus device=`.
- Parallel servers of the same stack need distinct internal ports.
- `--ipc=host` containers leak `/dev/shm` segments when killed (`scripts/clean_orphan_shm.sh`).
- Cross-container CUDA IPC needs `--pid=host`.
- An unattended upgrade can restart Fabric Manager mid-run; recover with stop FM → `nvidia-smi -r` → start FM.

Details are in the issues and `docs/`.

## Upstream contributions

- **deepseek-ai/DeepInfer**
  - #147: EPLB stalls ([comment](https://github.com/deepseek-ai/DeepInfer/pull/147#issuecomment-6334191794)) and fast prompt tokenization ([comment](https://github.com/deepseek-ai/DeepInfer/pull/147#issuecomment-6355311603)). Both were adopted in d7844a2 and 269a61f.
  - [#214](https://github.com/deepseek-ai/DeepInfer/pull/214): Mooncake KV transport for the PD split.
  - [#216](https://github.com/deepseek-ai/DeepInfer/pull/216): Mooncake Store L3 for the prefix cache.
- **kvcache-ai/Mooncake**
  - [#4873](https://github.com/kvcache-ai/Mooncake/pull/4873): nvlink_intra diagnostics for VMM buffers and PID namespaces.
  - [#4803 comment](https://github.com/kvcache-ai/Mooncake/issues/4803#issuecomment-6360317607): DeepInfer as a consumer of the per-sequence state contract.
- Drafts and posted texts are kept in [`docs/upstream/`](docs/upstream).

## Layout

```
bench/
  common.py, make_dataset.py, client_openai.py, run_deepinfer.py, summarize.py   fixed-length harness
  ix_recipe.py        expand InferenceX agentic.yaml into server args / env for one concurrency
  agentx_metrics.py   metrics with InferenceX's own code, compared with the published point
  agentx_report.py    AgentX table + Pareto and TTFT plots across configs
  ttft_breakdown.py   per-stage TTFT from DeepInfer server tracing (#17)
  render_bench.py, fastokens_bench.py   tokenizer microbenchmarks (#17)
  pd_parity.py        greedy output fingerprints for transport A/B (#18)
  mc_*_probe.py, cuda_probes/          Mooncake / CUDA IPC / VMM probes (#18, #19)
scripts/
  env.sh                 every path and the DeepInfer environment
  serve_ix.sh            vLLM / SGLang server (+ router) from the InferenceX recipe
  serve_deepinfer_agentx.sh, serve_deepinfer_pd.sh, serve_mooncake_store.sh
  serve_sglang.sh, serve_vllm.sh, sweep_*.sh, scenarios.sh, make_datasets.sh   fixed-length
  run_agentx.sh          AgentX (agentx-harness 1.0.12) against any of the servers
  sync_from_host.sh, pull_results.sh, clean_orphan_shm.sh, wait_*.sh, stop_servers.sh
docker/    image definitions (see the table above)
patches/   DeepInfer patch stacks used by the images
docs/      setup_deepinfer.md, fairness.md (read first), reference_configs.md, agentx.md, upstream/
results/   one directory per run; inferencex_public/ = published InferenceX points; b200_live/ = synced, git-ignored
```

<details><summary>中文</summary>

# DeepInfer vs SGLang vs vLLM：DeepSeek-V4.1-Flash 在 B200 上的对比

本仓库在一台 8×B200 上对比三个推理框架运行 `DeepSeek-V4.1-Flash-1017` 的表现：同一份权重、同一批输入、同一套指标。主要负载是 InferenceX 的 **AgentX** benchmark，比较之前先复现官方公布的 B200 数据点。

**Roadmap 和进度：[#8](https://github.com/whn09/deepinfer_sglang_vllm_bench/issues/8)。最新结果：[`results/20261023_agentx/summary.md`](results/20261023_agentx/summary.md)。**

## 结果（AgentX，每个 server 4 卡，每点 3600 s）

- **复现结果与 InferenceX 一致：** 我们跑的 vLLM TP4 与官方 7 个点全部对上（吞吐 0.95–1.02×）；SGLang 与官方 12 个点全部对上（0.98–1.06×），包括 DEP4 + HiCache。
- **DeepInfer 用 #147 server 的默认配置时落后 2–5 倍：** 原因是 EPLB 重排被写死开启（#5）。上游已修复，现在默认关闭 EPLB。
- **DeepInfer 的 TTFT 曾是 vLLM 的 3–4 倍：** 原因是每一轮都用 HF `tokenizers` 把整段对话重新编码（#17）。改用 vLLM Rust 前端同款的 `fastokens` 后，和 vLLM 只差 10–40 ms。这已经以 `--tokenizer-backend` 参数的形式合入上游 #147。
- **DeepInfer 现状**（相对 vLLM TP4）：
  - c ≥ 16 时吞吐 0.97–0.98×；
  - interactivity 在 c ≤ 16 时为 0.62–0.76×（DP attention 下一个请求只用一张卡），c ≥ 64 时为 0.92–0.95×。
- **SGLang：** 高并发下吞吐最高，但 DEP4 各点的 TTFT 有 1.3–1.6 s。
- **DeepInfer PD 4P+4D**（8 卡，#18）：c ≥ 64 时每用户速度最快，代价是每卡吞吐约减半。

## 进展

见上方英文表格，每一项对应一个 issue：复现（#4）、DeepInfer 跑 AgentX（#5）、定长对比（#6）、TTFT（#17）、YoCo（#16）、PD + Mooncake（#18）、Mooncake Store L3（#20）、VMM（#19）、最终报告（#7）。

## DeepInfer 在这里怎么跑

DeepInfer 开源版只能离线推理，所以跑 AgentX 用的是上游 PR 加上我们自己的几个补丁，全部放在 `patches/` 下，每套都构建成一个 Docker 镜像（`docker/`），对应关系见上方英文表格。

## 复现

B200 机器不固定，所有东西都放在 `/opt/dlami/nvme/bench_tmp`（`$WORK`，在 `scripts/env.sh` 里设置）。`scripts/sync_from_host.sh` 会定期把结果拉回本地，跑完后删掉远端目录。命令见上方英文部分，依次是：vLLM/SGLang 的 AgentX（按 InferenceX 自己的 recipe 起 server）、DeepInfer 的 AgentX、PD 分离、带 Mooncake Store L3 的前缀缓存，以及出报告。定长闭环对比见 `docs/fairness.md`。

**踩过的坑：**
- `--privileged` 容器会忽略 `--gpus device=`。
- 同一个框架并行起多个 server 时，内部端口要错开。
- `--ipc=host` 的容器被强杀后会在 `/dev/shm` 里留下共享内存段（`scripts/clean_orphan_shm.sh`）。
- 跨容器的 CUDA IPC 需要 `--pid=host`。
- 系统自动升级可能在运行中途重启 Fabric Manager，恢复方法是停 FM → `nvidia-smi -r` → 启动 FM。

详情见各 issue 和 `docs/`。

## 提交给上游的内容

见上方英文部分：给 DeepInfer #147 的两条建议都已被采纳（EPLB 默认关闭、fast tokenizer）；DeepInfer Draft PR #214（PD 的 Mooncake KV 传输）和 #216（前缀缓存的 Mooncake Store L3）；Mooncake PR #4873（nvlink_intra 诊断提示和文档），以及在 RFC #4803 下的评论。草稿和已发布的文字都保存在 `docs/upstream/`。

## 目录

见上方英文部分。
</details>
