"""生成三家共用的请求文件：固定输入长度、固定输出长度，直接存 token ids。

三家都直接吃 token ids（DeepInfer 的 prompt_token_ids、OpenAI /v1/completions 的
prompt=[int,...]），绕开各家 tokenizer / chat template 的差异，保证输入逐 token 相同。

两种来源：
  random    从非特殊 token 里均匀采样。只适合关投机解码的对比：随机文本上投机解码的
            接受率没有意义。
  sharegpt  把 ShareGPT 对话文本拼成一条长 token 流，每个请求从随机位置截取 input_len
            个 token。真实文本，投机解码接受率有意义；开投机解码的对比必须用这个。

每个请求的截取起点都不同，请求之间没有公共前缀，不会命中 prefix cache。
不同扫描点请用不同 --seed，避免同一个 server 跨扫描点命中上一个点留下的缓存。

用法：
  python bench/make_dataset.py --tokenizer $MODEL_DIR --source sharegpt \
      --input-len 1024 --output-len 1024 --num-requests 512 --seed 1 --out data/sg_1k_1k_s1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"


def load_sharegpt_texts(path: str | None, max_convs: int) -> list[str]:
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(SHAREGPT_REPO, SHAREGPT_FILE, repo_type="dataset")
    data = json.loads(Path(path).read_text())
    texts = []
    for conv in data[:max_convs]:
        for turn in conv.get("conversations", []):
            value = turn.get("value", "")
            if value:
                texts.append(value)
    return texts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="模型目录（含 tokenizer.json）")
    p.add_argument("--source", choices=["random", "sharegpt"], required=True)
    p.add_argument("--sharegpt-path", default=None, help="本地 ShareGPT json；不给就从 HF Hub 下载")
    p.add_argument("--sharegpt-max-convs", type=int, default=20000, help="只取前 N 条对话，够拼出足够长的 token 流即可")
    p.add_argument("--input-len", type=int, required=True)
    p.add_argument("--output-len", type=int, required=True)
    p.add_argument("--num-requests", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)

    if args.source == "random":
        special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
        pool = [i for i in range(tok.vocab_size) if i not in special]
        prompts = [[rng.choice(pool) for _ in range(args.input_len)] for _ in range(args.num_requests)]
    else:
        texts = load_sharegpt_texts(args.sharegpt_path, args.sharegpt_max_convs)
        stream: list[int] = []
        for t in texts:
            stream.extend(tok.encode(t, add_special_tokens=False))
            # 每段之间插个换行，避免两段文本粘连成不自然的 token
            stream.extend(tok.encode("\n\n", add_special_tokens=False))
        need = args.input_len + 1
        if len(stream) < need:
            raise SystemExit(f"ShareGPT token 流只有 {len(stream)} 个 token，不够截 {args.input_len}；调大 --sharegpt-max-convs")
        prompts = []
        for _ in range(args.num_requests):
            start = rng.randrange(0, len(stream) - args.input_len)
            prompts.append(stream[start : start + args.input_len])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, ids in enumerate(prompts):
            f.write(json.dumps({"id": i, "prompt_token_ids": ids, "output_len": args.output_len}) + "\n")
    meta = {
        "source": args.source,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_requests": args.num_requests,
        "seed": args.seed,
        "tokenizer": str(args.tokenizer),
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(prompts)} requests -> {out}")


if __name__ == "__main__":
    main()
