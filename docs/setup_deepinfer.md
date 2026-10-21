# DeepInfer 在 B200 上的编译与模型下载指南

本文记录在 AWS B200 机器（DLAMI，`/opt/pytorch` venv）上从零编译 DeepInfer、下载 `DeepSeek-V4.1-Flash-1017` 权重的正确步骤，以及每一步背后踩过的坑。2026-10-21 实测通过。

## 1. 环境概况

| 项目 | 值 |
|---|---|
| GPU | NVIDIA B200 (SM100) |
| 驱动 | 595.91.07，**最高支持 CUDA 13.2** |
| Python 环境 | DLAMI 自带的 `/opt/pytorch` venv（Python 3.13） |
| PyTorch | 2.13.0+cu132（由 `requirements-nvidia.txt` 安装） |
| CUDA 工具链 | venv 里 pip 安装的 `nvidia/cu13`，**没有** `/usr/local/cuda` |
| 代码 | `~/DeepInfer` |
| 数据盘 | `/opt/dlami/nvme`（28T） |

> README 写的是 Python 3.12，实测 3.13 也可以正常编译、import。

**核心问题：** venv 自带的 CUDA 编译工具链（nvcc / cicc 等）是 13.0，但 torch cu132 会把 CUDA runtime 和头文件升级到 13.2。两边版本不一致时，编译 CUDA kernel 会接连出错。所以编译前必须把工具链也对齐到 13.2。

## 2. 一次性准备（每台新机器只做一次）

### 2.1 对齐 CUDA 工具链到 13.2

```bash
source /opt/pytorch/bin/activate

pip install --no-deps \
    nvidia-cuda-nvcc==13.2.78 \
    nvidia-cuda-crt==13.2.78 \
    nvidia-cuda-cccl==13.2.75 \
    nvidia-nvvm==13.2.78 \
    nvidia-cuda-cuobjdump==13.2.78
```

每个包装它的原因：

| 包 | 不装/不对齐的报错 |
|---|---|
| `nvidia-cuda-nvcc`、`nvidia-cuda-crt` | `CUDA compiler and CUDA toolkit headers are incompatible`（nvcc 13.0 遇到 13.2 的头文件） |
| `nvidia-cuda-cccl` | 同上，cccl 头文件要和 nvcc 版本一致 |
| `nvidia-nvvm` | `macro '__cudaLaunch' passed 2 arguments, but takes just 1`（CUDA 13 把 cicc 拆到单独的 nvvm 包里，很容易漏） |
| `nvidia-cuda-cuobjdump` | `error: cuobjdump not found`（DeepSelect / FlashMLA 编完后要用它检查寄存器溢出，venv 默认没装） |

装完后检查一下：

```bash
/opt/pytorch/cuda/bin/nvcc --version | tail -2   # 应显示 release 13.2, V13.2.78
pip list | grep -iE "nvidia-(cuda-(nvcc|crt|cccl|runtime|cuobjdump)|nvvm)"   # 全部应为 13.2.x
```

> 版本必须是 13.2，不要装 13.3 / 13.4：驱动最高只支持 13.2，而且 torch 是 cu132。

### 2.2 安装系统依赖

```bash
sudo apt-get install -y libdw-dev
```

DeepEP 里的 deep_jit 要用到 `elfutils/libdwfl.h`，不装会报 `fatal error: elfutils/libdwfl.h: No such file or directory`。

## 3. 编译

### 3.1 环境变量

每次新开 shell、编译或运行前都执行下面这段（机器不固定，换机器后照抄即可，不依赖机器上预先存放的任何文件）。本仓库的 `scripts/env.sh` 已经包含这段内容，另外还定义了评测要用的路径：

```bash
source /opt/pytorch/bin/activate

export CUDA_HOME=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CUDA_HOME/include/cccl:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH

export DEEPINFER_ACCELERATOR=nvidia
export FLASH_MLA_SKIP_REG_SPILL_CHECK=1

hash -r   # 清掉 shell 缓存的命令路径，确保 hf/python 用的是 venv 里的
```

其中两处和官方 README 不同：

- **`CPLUS_INCLUDE_PATH` 里加了 `$CUDA_HOME/include/cccl`。**
  FlashMLA 和 DeepSelect 的 `setup.py` 按标准 CUDA toolkit 的目录结构，到 `$CUDA_HOME/targets/x86_64-linux/include/cccl` 找 cccl。pip 装的 CUDA 没有 `targets/` 目录，cccl 在 `$CUDA_HOME/include/cccl`。nvcc 编 `.cu` 时会通过 `nvcc.profile` 自动加上这个路径，但 g++ 编 `.cpp` 时不会，于是报 `fatal error: cuda/std/utility: No such file or directory`。

- **`FLASH_MLA_SKIP_REG_SPILL_CHECK=1`。**
  FlashMLA 编完会检查 kernel 有没有寄存器溢出，有就让 build 失败。用 CUDA 13.2 的 ptxas 编，sm100 head64 FP8 sparse decode kernel 有两个模板实例各用了 16 字节栈（`STACK:16 LOCAL:0`），其他 kernel 都是零溢出。FlashMLA README 写明他们是在 CUDA 13.3 上调到零溢出的，但这台机器的驱动只支持到 13.2，所以只能跳过这个检查。
  按我的理解，这不影响计算结果，只可能让这个 decode kernel 稍微慢一点，具体慢多少还没实测。如果要拿这个 kernel 跟官方性能数字比，要记住这一点。

### 3.2 执行编译

```bash
cd ~/DeepInfer
# 先执行 3.1 的环境变量
./build.sh 2>&1 | tee /tmp/di_build.log
```

`build.sh` 依次做这些事：

1. `pip install -r requirements-nvidia.txt`（装 torch 2.13.0+cu132、huggingface_hub 0.36.2 等）
2. 把 NCCL 换成 2.32.3
3. `scripts/build_vendor.sh`：编译 vendor 依赖，依次是 DeepSelect → FlashMLA → DeepEP 等
4. `scripts/build.sh build`：编译 DeepInfer 本体 `_C`

成功时最后一行是：

```
✅ DeepInfer is successfully built
```

### 3.3 验证

```bash
cd ~/DeepInfer      # deepinfer 是源码树内构建，不装进 site-packages，必须在仓库根目录下运行
python -c "import torch, deepinfer, deepinfer._C, deep_select, flash_mla, deep_ep; \
print('OK', torch.__version__, torch.cuda.get_device_name(0))"
# 期望输出：OK 2.13.0+cu132 NVIDIA B200
```

> 在其他目录下运行会报 `ModuleNotFoundError: No module named 'deepinfer'`。这是正常的，`cd ~/DeepInfer` 再运行即可（`deep_select` / `flash_mla` / `deep_ep` 是装进 venv 的，在哪个目录都能 import）。

## 4. 下载模型

### 4.1 正确的下载命令

```bash
# 先执行 3.1 的环境变量（关键：要用 venv 里的 hf）
export HF_HOME=/opt/dlami/nvme/hf
# export HF_TOKEN=hf_xxx           # 可选，设置后限速更宽松、下载更快

which hf                           # 必须是 /opt/pytorch/bin/hf，不能是 /usr/bin/hf

nohup hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --local-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash-1017 \
    > /tmp/hf_dl.log 2>&1 &

tail -f /tmp/hf_dl.log
```

- 权重约 **476G**，共 48 个 safetensors，务必放在 `/opt/dlami/nvme`，根分区放不下。
- 用 `nohup` 放到后台跑，SSH 断开也不会中断。
- 中途断了就重新执行同一条命令，已下完的文件会自动跳过。

### 4.2 为什么之前会报错

报错是：

```
ValueError: The file is too large to be downloaded using the regular download method. Install `hf_xet` ...
```

原因是调到了**系统自带的 `/usr/bin/hf`**：它跑在系统 Python 3.14 上，huggingface_hub 是 1.2.2，**没装 `hf_xet`**。这个模型仓库用 Xet 存储，单个文件太大，没有 hf_xet 只能退回普通 HTTP 下载，而普通 HTTP 不支持这么大的文件，所以直接报错。

venv 里的 `/opt/pytorch/bin/hf` 自带 `hf-xet 1.7.0`，可以正常下载。但它是 `build.sh` 安装依赖时才生成的。如果你的 shell 在那之前已经执行过 `hf`，bash 会继续用缓存的 `/usr/bin/hf`。执行一次 `hash -r` 或者直接写全路径 `/opt/pytorch/bin/hf` 即可。

> 不要往系统 Python 里 `pip install hf_xet`：Ubuntu 的系统 Python 受 PEP 668 保护，而且没有必要。

### 4.3 检查下载是否完整

```bash
D=/opt/dlami/nvme/models/DeepSeek-V4.1-Flash-1017
ls $D/*.safetensors | wc -l                                     # 应为 48
ls $D/.cache/huggingface/download | grep -c incomplete          # 应为 0
du -sh $D                                                        # 约 476G
```

> 下载过程中 `du` 显示的大小可能远超实际进度（hf_xet 会先占好文件空间），以日志和 `incomplete` 文件数为准。

### 4.4 使用本地模型

DeepInfer 默认会在第一次运行时把模型下载到 HF 缓存。要用本地已下载的副本，可以这样指定：

```bash
cd ~/DeepInfer
# 先执行 3.1 的环境变量
export DEEPINFER_HF_MODEL_DIR=/opt/dlami/nvme/models/DeepSeek-V4.1-Flash-1017
python examples/offline_inference.py
```

## 5. 报错速查

| 报错 | 原因 | 解决 |
|---|---|---|
| `CUDA compiler and CUDA toolkit headers are incompatible` | nvcc 13.0 遇到 13.2 头文件 | 2.1：对齐 nvcc / crt / cccl |
| `macro '__cudaLaunch' passed 2 arguments, but takes just 1` | cicc（nvvm 包）还是 13.0 | 2.1：`nvidia-nvvm==13.2.78` |
| `error: cuobjdump not found` | venv 没装 cuobjdump | 2.1：`nvidia-cuda-cuobjdump==13.2.78` |
| `fatal error: cuda/std/utility: No such file or directory` | g++ 找不到 pip 布局下的 cccl | 3.1：`CPLUS_INCLUDE_PATH` 加 `$CUDA_HOME/include/cccl` |
| `Register spilling detected. Build failed!`（FlashMLA） | CUDA 13.2 ptxas 编出 16 字节栈 | 3.1：`FLASH_MLA_SKIP_REG_SPILL_CHECK=1` |
| `fatal error: elfutils/libdwfl.h: No such file or directory` | 缺系统头文件 | 2.2：`sudo apt-get install -y libdw-dev` |
| `The file is too large to be downloaded using the regular download method` | 用了没有 hf_xet 的 `/usr/bin/hf` | 4.1：`hash -r`，用 `/opt/pytorch/bin/hf` |
| `ModuleNotFoundError: No module named 'deepinfer'` | 不在仓库根目录下运行 | `cd ~/DeepInfer` |
| `Usage: DEEPINFER_ACCELERATOR=nvidia\|ascend` | 没设置加速器类型 | `export DEEPINFER_ACCELERATOR=nvidia` |

排查编译错误时，在日志里找第一个出错的位置：

```bash
grep -n -E "error|FAILED|not found" /tmp/di_build.log | head
```
