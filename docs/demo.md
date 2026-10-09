# 在这台 RTX 5090 上跑 Token Rush demo

只覆盖两件事：一条命令出一段回复，以及把同一个引擎开成 OpenAI / Anthropic 兼容的本地服务。基准、量化、对手复现不在这里。

仓库根目录执行下面的命令。这台机器是本机 Ubuntu（内核 `7.0.0-34-generic`），不是 vast 容器。

## 这台机器现在的状态

2026-10-08 装完环境之后：

| | |
|---|---|
| GPU | PCI `01:00.0`，RTX 5090，32 GB。旁边还有一块 AMD 核显，桌面在那块卡上 |
| 驱动 | `nvidia-smi` 可用。驱动 **595.91.07**，CUDA 13.2。内核模块是 `linux-modules-nvidia-595-open-7.0.0-34-generic`，对应当前内核 `7.0.0-34-generic` |
| Python | 项目虚拟环境 `.venv`，解释器是系统的 **3.12.3**。miniconda 默认仍是 3.13，没有往里面装任何东西 |
| `uv` | `~/.local/bin/uv` 0.12.23，只属于 jesse。没有改 `.bashrc` |
| torch | `2.14.0+cu130`，能看到这张 5090 |
| 权重 | 还没下。对话 demo 第一次会拉 17 GB + 3.9 GB |

内核当初升到 `7.0.0-34` 时，NVIDIA 模块还停在 `7.0.0-31`，所以 `nvidia-smi` 连不上驱动。补上对应模块时，apt 把已经装好的 595.84 用户态包一起升到了 595.91（模块包依赖新版本）。没有重启，当时登录的图形会话还在。以后内核再升级，要装 `linux-modules-nvidia-595-open-$(uname -r)`，否则 `nvidia-smi` 会再失效。

显存尽量空着。服务默认上下文是 256k，两份 draft 一起驻留大约 30 GB。只跑下面的对话 demo（默认 32k 上下文）则宽松得多。

## 环境已经装好

虚拟环境在仓库里的 `.venv`，依赖按 `uv.lock` 装齐：torch cu130、Triton、transformers、flash-linear-attention。conda 和系统 Python 都没动。确认一下：

```bash
cd /home/jesse/workspace/token-rush
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

应看到 `2.14.0+cu130` 和 `NVIDIA GeForce RTX 5090`。

第一次 `import` 引擎时，Marlin 扩展（`tokenrush/csrc/`）用本机 nvcc 现场编译，大约几秒到十几秒，只要成功一次。

换一台机器重装时：把 `uv` 装进自己的 `~/.local/bin`（`INSTALLER_NO_MODIFY_PATH=1`，避免改 shell 配置），在仓库里 `uv sync`。不要 `pip install torch` 到 conda 3.13 里。PyPI 官方源在这台机器上很慢，大包可以走 `https://pypi.tuna.tsinghua.edu.cn/packages` 下到本地再用 `uv pip install --no-index --find-links`。

## 3. 对话 demo

第一次会从 Hugging Face 拉两份权重进 Hub 缓存（默认 `~/.cache/huggingface`）：

| 仓库 | 作用 | 大小 |
|---|---|---|
| `zyhector/Qwen3.8-27B-TokenRush-int4g128` | 4.25 bpw 的正文 | 17 GB |
| `z-lab/Qwen3.8-27B-DFlash2` | 投机解码的 draft | 3.9 GB |

需要能访问 huggingface.co。缓存放别处时设 `HF_HOME`。已经下过、不想再联网时加 `--no-download`。

```bash
uv run python -m tokenrush.run --chat \
  --prompt "Explain speculative decoding in three sentences."
```

这条命令做的事：

1. 加载 int4 权重，编译并捕获整步 CUDA graph，再捕获 DFlash2 的投机 graph（英文 prompt 走 DFlash2；中文默认改走模型自带的 MTP head）。
2. 用 checkpoint 里的 chat template 包住 prompt，thinking 关着。
3. 贪心解码最多 200 个新 token，最后打一行 JSON：token 数、耗时、tok/s。

第一次加载加上 graph capture，冷启动大约一分钟量级，随后才开始吐字。短上下文、贪心、散文，预期在 **200 tok/s 以上**（README 同卡测量是 229）。数字随功耗墙和是否独占 GPU 浮动，差一截不代表装错了。

常用开关：

```bash
# 中文 prompt；--draft auto 时 CJK 占比 ≥ 20% 会改用 MTP
uv run python -m tokenrush.run --chat --prompt "用三句话解释投机解码。"

# 关掉投机，只看原始 decode（大约 100 tok/s）
uv run python -m tokenrush.run --chat --no-spec --prompt "Say hello in one sentence."

# 强制某一路 draft
uv run python -m tokenrush.run --chat --draft dflash --prompt "..."
uv run python -m tokenrush.run --chat --draft mtp --prompt "..."

# 采样而不是贪心
uv run python -m tokenrush.run --chat --temperature 0.7 --top-p 0.9 --prompt "..."

# 多生成一些；默认 --max-new 200，--max-len 32768
uv run python -m tokenrush.run --chat --max-new 512 --prompt "..."
```

权重不在默认 Hub id 上时：`--model` 接受仓库 id 或本地已打包目录，`--dflash-path` 同理。

## 4. 本地服务

引擎常驻，一次只处理一个请求，后面的排队。对话上下文留在 GPU 上，下一轮只 prefill 新增的 token。

```bash
uv run python -m tokenrush.serve --port 8000
```

默认 `--max-len 262144`（显存大约 30 GB）、KV 用 FP8、温度 0.7、top-p 0.9。旁边还有别的程序用 GPU 时，先把窗口缩小：

```bash
uv run python -m tokenrush.serve --port 8000 --max-len 32768
```

另开一个终端：

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"token-rush","messages":[{"role":"user","content":"Write a haiku about GPUs."}]}'
```

模型名不校验，请求里写什么，响应里就回什么。流式加上 `"stream": true`。

OpenAI SDK：`base_url` 设为 `http://127.0.0.1:8000/v1`，`api_key` 随便填（没加 `--api-key` 时不检查）。

Claude Code：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=anything
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768   # 与启动时的 --max-len 一致
claude
```

服务只听 `127.0.0.1`。要给别的机器用，加 `--host 0.0.0.0`，并设 `--api-key`（或环境变量 `TOKENRUSH_API_KEY`），客户端用 `Authorization: Bearer <key>` 或 `x-api-key`。

协议细节、tool calling、前缀复用的耗时在 [serving.md](serving.md)。

## 启动失败时先看这几条

| 现象 | 原因 |
|---|---|
| `nvidia-smi` 连不上驱动 | 第 1 节的内核模块没装上或没 `modprobe`。内核若再升级，模块包要跟着装 `linux-modules-nvidia-595-open-<uname -r>` |
| `no kernel image is available` | torch 不是 cu130 构建。用 `uv sync`，不要装 cu124 wheel |
| CUDA out of memory | 服务默认 256k。改 `--max-len 32768`，或停掉别的 GPU 进程 |
| Hub 下载停在 0%，或 `CAS Client Error` / `error decoding response body` | Xet 通道（`cas-server.xethub.hf.co`）在这条链路上会卡住或中途断掉。同一次运行会改走普通 HTTP，一次只下一个文件。进度条一直不动时停掉，设 `HF_HUB_DISABLE_XET=1` 再跑。需要 token 时 `hf auth login` 或设 `HF_TOKEN` |
| Marlin 编译失败 | 需要 nvcc（这台在 `/usr/local/cuda/bin`）和 `ninja`（依赖里已有）。临时绕过：`--backend triton`，原始 decode 略快、验证步略慢 |
| import 用了 Python 3.13 | 那是 conda。命令都走 `uv run`，它用项目自己的 3.12 |
