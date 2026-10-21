# 场景与并发点定义，sweep.sh 会 source 这个文件。
# 场景名 -> "数据来源 输入长度 输出长度"
#   random   ：随机 token，只用于关投机解码的对比
#   sharegpt ：真实文本，开投机解码的对比必须用它（接受率才有意义）
declare -A SCENARIOS=(
    [rand-1k-1k]="random 1024 1024"
    [sg-1k-1k]="sharegpt 1024 1024"
    [sg-8k-1k]="sharegpt 8192 1024"
    [sg-32k-1k]="sharegpt 32768 1024"
)

# 闭环并发点（全局并发，DeepInfer 侧会按 GPU 均分）
CONCURRENCIES=(${CONCURRENCIES:-1 4 8 16 32 64 128})

# 每个点的请求数：至少 4 倍并发，保证稳态窗口够长；下限 16
num_requests_for() {
    local c=$1
    local n=$((c * 4))
    [ "$n" -lt 16 ] && n=16
    echo "$n"
}
