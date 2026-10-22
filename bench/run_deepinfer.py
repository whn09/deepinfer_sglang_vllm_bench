"""DeepInfer 侧的压测：离线引擎 + 闭环固定并发，逐请求记时间戳，指标口径与 client_openai.py 相同。

DeepInfer 没有 HTTP server，只有离线 LLM / LLMEngine。这里直接驱动 LLMEngine：
一个请求结束就补一个新请求（add_request），始终保持 C 个在飞，然后 step()——
和 server 侧"并发 C 的闭环 client"语义一致。

并行方式与 DeepInfer 自己的脚本一样：每张卡一个 DP rank 进程（attention DP + MoE EP），
全局并发 C 和请求按 rank 均分；所有 rank 每轮都要调用 has_unfinished_requests()
（它内部是跨 rank 的 all_reduce），所以没有请求的 rank 也得陪着 step。

刻意**不**照搬 examples/benchmark.py 的几个设置（它们会让数字和真实服务不可比）：
  - DEEPINFER_FORCE_LOAD_BALANCE=1：MoE gate 改成随机路由（gate_force_random），专家负载
    被人为摊平，不是模型真实路由；
  - 吞吐按"投机草稿全部接受"折算（1+nextn tokens/step）：这里只数实际输出的 token；
  - 随机 token prompt：由 make_dataset.py 决定，开投机解码时用 sharegpt 真实文本；
  - DEEPINFER_BATCH_INVARIANT=1（offline_inference.py 在 NVIDIA 上默认开）：为逐位可复现
    牺牲性能，压测不开。

一个进程按 --plan 依次跑多个 (数据集, 并发) 点，只加载一次 476G 权重。

用法（在 DeepInfer 仓库根目录、执行过 scripts/env_deepinfer.sh 之后）：
  python $BENCH/bench/run_deepinfer.py --num-gpus 8 --nextn 0 --config deepinfer-ep8-nospec \
      --plan data/rand_1k_1k_s1.jsonl@1 data/rand_1k_1k_s8.jsonl@8 ... --out results/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
DEEPINFER_ROOT = Path(os.environ.get("DEEPINFER_ROOT", Path.home() / "DeepInfer")).resolve()
sys.path.insert(0, str(DEEPINFER_ROOT / "examples"))
sys.path.insert(0, str(BENCH_DIR))

from common import append_result, compute_metrics, load_dataset, load_dataset_meta, summary_line, write_records  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", default=os.environ.get("DEEPINFER_HF_MODEL_DIR"), help="本地模型目录，默认取 DEEPINFER_HF_MODEL_DIR")
p.add_argument("--num-gpus", type=int, required=True, help="EP/DP 规模 = 使用的 GPU 数")
p.add_argument("--device-offset", type=int, default=0, help="从第几号卡开始用（如 4 表示用 4-7 号卡）")
p.add_argument("--nextn", type=int, required=True, help="DSpark 投机解码 block 大小（DEEPINFER_USE_NEXTN_DECODE）；0 = 关闭")
p.add_argument("--config", required=True, help="配置名，写进结果用于分组，如 deepinfer-ep8-nospec")
p.add_argument("--plan", nargs="+", required=True, metavar="DATASET@CONCURRENCY", help="依次测量的点")
p.add_argument("--out", required=True, help="结果 jsonl（追加）")
p.add_argument("--records-dir", default=None)
p.add_argument("--num-requests", type=int, default=None, help="每个数据集只用前 N 条")
p.add_argument("--warmup-requests", type=int, default=None, help="每个点正式测量前的预热请求数，默认 = 该点并发")
p.add_argument("--temperature", type=float, default=0.0)
p.add_argument("--top-p", type=float, default=1.0)
p.add_argument("--max-num-batched-tokens", type=int, default=16384, help="prefill chunk 大小，对齐 server 侧的 chunked prefill")
p.add_argument("--max-model-len", type=int, default=None, help="默认按数据集最长 输入+输出 自动取")
p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
p.add_argument("--sampler-force-topk", type=int, default=4096)
p.add_argument("--enable-eplb", action="store_true", help="开 EPLB（专家重排 + 冗余专家）；参考配置的 sglang/vllm 都没开")
p.add_argument(
    "--prefill-any-interval",
    type=int,
    default=2,
    help="DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL：其他 rank 空闲时每隔多少个 decode step 接纳一次等待中的 prompt。"
    "引擎默认 100 会给每个请求加一个 ~0.5-1.5s 的 TTFT 下限（以 decode 吞吐换 TTFT）；对齐服务端行为用 2，"
    "见 deepseek-ai/DeepInfer#147 和本仓库 #6",
)
p.add_argument(
    "--yoco-keep-tokens",
    type=int,
    default=140,
    help="YoCo prefill 截断（DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact）保留的末尾 token 数；0 = 关。"
    "V4.1 是 YoCo checkpoint（n_prefill_layer=20 / 40）：consumer 层的 prefill 只需算最后 KEEP 个 token，"
    "KEEP >= 滑窗 128 + 回溯时与不截断等价。DeepInfer 自己的长上下文 benchmark 和 #147 的 server 默认都开（#16）",
)
p.add_argument("--dump-tokens", action="store_true", help="把每个请求的输出 token ids 写进逐请求记录（用于对比开/关某个优化时输出是否一致）")
p.add_argument("--log-interval", type=int, default=50, help="引擎日志间隔（step），日志里有 avg_nextn_accept_rate（引擎自报的投机接受率）；0 关闭")
p.add_argument("--note", default="")
args = p.parse_args()

if not args.model:
    raise SystemExit("需要 --model 或 DEEPINFER_HF_MODEL_DIR")


def parse_plan() -> list[tuple[str, int]]:
    plan = []
    for entry in args.plan:
        path, _, c = entry.rpartition("@")
        plan.append((path, int(c)))
    return plan


PLAN = parse_plan()
DP = args.num_gpus
# spawn 出来的 rank 子进程 pid 各不相同，临时目录名必须用父进程定下的标记（经环境变量继承）
os.environ.setdefault("BENCH_RUN_TAG", f"{os.getpid()}_{int(time.time())}")
TMP_DIR = Path(args.out).parent / f".tmp_{args.config}_{os.environ['BENCH_RUN_TAG']}"


def split_for_rank(n_items: int, concurrency: int, rank: int) -> tuple[list[int], int]:
    """全局并发 C 均分给各 rank（余数给前几个 rank）；请求按 rank 的并发份额轮流分配。"""
    shares = [concurrency // DP + (1 if r < concurrency % DP else 0) for r in range(DP)]
    active = [r for r in range(DP) if shares[r] > 0]
    idx = [i for i in range(n_items) if active[i % len(active)] == rank]
    return idx, shares[rank]


def max_model_len_for_plan() -> int:
    if args.max_model_len:
        return args.max_model_len
    longest = 0
    for path, _ in PLAN:
        for it in load_dataset(path, args.num_requests):
            longest = max(longest, len(it["prompt_token_ids"]) + it["output_len"])
    # 引擎的 profile run 要求 max_model_len ≥ 单条序列可能分到的 prefill chunk（max_num_seqs=1 时就是整个 chunk）
    return max(((longest + 1024 + 4095) // 4096) * 4096, args.max_num_batched_tokens)


def rank_main(dp_size: int, dp_rank: int) -> None:
    os.environ["DEEPINFER_USE_NEXTN_DECODE"] = str(args.nextn)
    os.environ["DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL"] = str(args.prefill_any_interval)
    if args.yoco_keep_tokens > 0:
        os.environ["DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE"] = "compact"
        os.environ["DEEPINFER_YOCO_PREFILL_KEEP_TOKENS"] = str(args.yoco_keep_tokens)
        os.environ["DEEPINFER_PREFILL_STORE_EXTRA_STATE"] = "1"
    else:
        os.environ["DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE"] = "off"
    os.environ["DEVICE_OFFSET"] = str(args.device_offset)

    from _config import create_llm, setup_common_env

    setup_common_env(
        model_name=args.model,
        sampler_force_topk=args.sampler_force_topk,
        set_override_experts=args.enable_eplb,
        enable_eplb=args.enable_eplb,
    )
    max_conc_local = max(split_for_rank(c, c, dp_rank)[1] for _, c in PLAN)
    llm = create_llm(
        args.model,
        dp_rank,
        ep_size=dp_size,
        cp_size=1,
        env={"DEEPINFER_LOGGING_INTERVAL_STEP": str(args.log_interval)},
        data_parallel_size=dp_size,
        max_num_seqs=max(1, max_conc_local),
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=max_model_len_for_plan(),
        gpu_memory_utilization=args.gpu_memory_utilization,
        lookahead_tokens=min(1, args.nextn),
    )

    import numpy as np

    from deepinfer._C.engine import RequestOutput  # pyright: ignore[reportMissingModuleSource]
    from deepinfer.sampling_params import SamplingParams

    engine = llm.llm_engine
    counter = [0]

    def add(item: dict[str, Any], max_tokens: int) -> str:
        counter[0] += 1
        rid = f"r{dp_rank}_{counter[0]}"
        sp = SamplingParams(max_tokens=max_tokens, ignore_eos=True, temperature=args.temperature, top_p=args.top_p)
        mm = dict(image_bytes=[], image_sizes=[]) if engine.multi_modal_processor is not None else None
        engine.add_request(
            user_id="0",
            request_id=rid,
            sampling_params=sp,
            prompt_token_ids=np.asarray(item["prompt_token_ids"], dtype=np.int32),
            multi_modal_data=mm,
        )
        return rid

    def closed_loop(items: list[dict[str, Any]], conc: int, override_out: int | None = None) -> list[dict[str, Any]]:
        pending = deque(items)
        inflight: dict[str, dict[str, Any]] = {}
        done: list[dict[str, Any]] = []
        while True:
            while pending and len(inflight) < conc:
                it = pending.popleft()
                out_len = override_out or it["output_len"]
                t = time.time()
                rid = add(it, out_len)
                inflight[rid] = {
                    "id": it["id"],
                    "prompt_len": len(it["prompt_token_ids"]),
                    "expected_output_len": out_len,
                    "t_submit": t,
                    "chunks": [],
                    "output_len": 0,
                    "accept_tok": 0,
                    "draft_tok": 0,
                    **({"tokens": []} if args.dump_tokens else {}),
                    "error": None,
                }
            unfinished = engine.has_unfinished_requests()  # 跨 rank all_reduce，每个 rank 每轮都要调
            step_outputs = engine.step() if unfinished else engine.flush_inflight_reqs()
            now = time.time()
            per_req: dict[str, int] = {}
            finished: list[str] = []
            for raw in step_outputs:
                out = RequestOutput.clone(raw)
                rec = inflight.get(out.request_id)
                if rec is None:
                    continue
                per_req[out.request_id] = per_req.get(out.request_id, 0) + 1
                rec["accept_tok"] += int(out.accept_tok)
                if args.dump_tokens:
                    rec["tokens"].append(int(out.token_id))
                rec["draft_tok"] += int(out.draft_tok)
                if out.finished:
                    finished.append(out.request_id)
            for rid, n in per_req.items():
                rec = inflight[rid]
                rec["chunks"].append([now, n])
                rec["output_len"] += n
                if "t_first" not in rec:
                    rec["t_first"] = now
            for rid in finished:
                rec = inflight.pop(rid)
                rec["t_end"] = now
                done.append(rec)
            if not unfinished:
                break
        for rid, rec in inflight.items():  # 理论上不会发生
            rec["error"] = "not finished"
            done.append(rec)
        return done

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for point, (path, conc) in enumerate(PLAN):
        items = load_dataset(path, args.num_requests)
        idx, conc_local = split_for_rank(len(items), conc, dp_rank)
        local = [items[i] for i in idx]

        n_warm = args.warmup_requests if args.warmup_requests is not None else conc
        warm_idx, _ = split_for_rank(min(n_warm, len(items)), conc, dp_rank)
        # 预热请求用逆序 token，避免和正式请求共享前缀
        warm = [{"id": -1, "prompt_token_ids": list(reversed(items[i]["prompt_token_ids"])), "output_len": 16} for i in warm_idx]
        closed_loop(warm, max(conc_local, 1), override_out=16)

        recs = closed_loop(local, max(conc_local, 1))
        write_records(TMP_DIR / f"p{point}_rank{dp_rank}.jsonl", recs)
        if dp_rank == 0:
            print(f"[{args.config}] point {point} {Path(path).name} c={conc} done", flush=True)

    llm.close()


def merge_and_report() -> None:
    rec_dir = Path(args.records_dir) if args.records_dir else Path(args.out).parent / "records"
    for point, (path, conc) in enumerate(PLAN):
        records: list[dict[str, Any]] = []
        for r in range(DP):
            f = TMP_DIR / f"p{point}_rank{r}.jsonl"
            if f.exists():
                records.extend(json.loads(line) for line in f.read_text().splitlines() if line)
        records.sort(key=lambda x: x["id"])
        metrics = compute_metrics(records, DP)
        acc = sum(r.get("accept_tok", 0) for r in records)
        draft = sum(r.get("draft_tok", 0) for r in records)
        result = {
            "system": "deepinfer",
            "config": args.config,
            "dataset": str(path),
            "dataset_meta": load_dataset_meta(path),
            "concurrency": conc,
            "sampling": {"temperature": args.temperature, "top_p": args.top_p},
            "nextn": args.nextn,
            # 引擎原始计数，nextn=0 时也会计数（每个 token 各记 1），不能直接当接受率用；
            # 跨系统对比看 compute_metrics 里的 tokens_per_chunk（= 平均每步输出 token 数）
            "engine_accept_tok": acc,
            "engine_draft_tok": draft,
            "engine_args": {
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enable_eplb": args.enable_eplb,
                "sampler_force_topk": args.sampler_force_topk,
                "prefill_any_interval": args.prefill_any_interval,
                "yoco_keep_tokens": args.yoco_keep_tokens,
            },
            "note": args.note,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            **metrics,
        }
        append_result(args.out, result)
        write_records(rec_dir / f"{args.config}_{Path(path).stem}_c{conc}.jsonl", records)
        print(summary_line(args.config, conc, metrics), flush=True)
    for f in TMP_DIR.glob("*.jsonl"):
        f.unlink()
    TMP_DIR.rmdir()


if __name__ == "__main__":
    from _launch import run_dp

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    run_dp(rank_main, DP)
    merge_and_report()
