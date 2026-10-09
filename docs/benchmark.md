# 在这台 RTX 5090 上做评测

仓库根目录执行。权重已经在本机：正文在 Hub 缓存，DFlash2 在 `~/.cache/tokenrush/Qwen3.8-27B-DFlash2`。命令都加 `--no-download`，避免再去 Hugging Face 拉 17 GB。

评测回答两个不同的问题。

| 问题 | 看什么 | 正式数字在哪 |
|---|---|---|
| 引擎快不快 | tok/s，以及它占带宽墙的比例 | [README](../README.md) 开头的表，[baselines.md](baselines.md) |
| 4-bit 权重损了多少 | 对 bf16 的 KL、困惑度、GSM8K | [quantization.md](quantization.md)，[model_card.md](model_card.md) |

下面是在这台机器上把这两件事再跑一遍的步骤。抽查题是另一件事：20 道自己出的题，用来看聊天时答得对不对，不能当成榜单分数。

显存尽量空着。旁边有别的进程时，tok/s 会掉一截，答案不受影响。贪心解码下，草稿只决定速度，不改最终 token。

## 先装基准依赖

对话 demo 不需要这些包。直接跑 GSM8K 会报 `ModuleNotFoundError: No module named 'datasets'`，因为 `datasets` 在可选依赖 `bench` 里，默认的 `uv sync` 不会装。

```bash
cd /home/jesse/workspace/token-rush
uv sync --extra bench
```

这一组是 `datasets`、`accelerate`、`matplotlib`、`pytest`。GSM8K 题目本身只有几 MB，第一次 `load_dataset` 会从 Hugging Face 拉 `openai/gsm8k`。

后面命令里的草稿路径：

```bash
DFLASH="$HOME/.cache/tokenrush/Qwen3.8-27B-DFlash2"
```

## 速度

### 原始 decode

不开投机，整步 CUDA graph，短 prompt。Marlin 是默认后端。

```bash
uv run python bench/decode.py --no-download --steps 50
```

README 同卡测量：Triton 后端 **100.9 tok/s**（带宽墙的 81%），Marlin 布局大约 **98 tok/s**。差一截时先看是不是独占 GPU，再加 `--backend triton` 对一下。`--context N` 可以先塞 N 个 token 再计时，用来看长上下文；256k 的 needle 是下一节。

### 六类 prompt 的有效速度

每类生成 300 个 token，打印 tok/s、每步接受几个草稿、每步毫秒数。列的顺序是英文散文、代码、数学、中文散文、中文数学、中英混合。

```bash
uv run python bench/families.py --no-download \
  --dflash-path "$DFLASH" --draft dflash --new 300
```

`--draft dflash,mtp` 两路都跑。英文三列的正式数字是 README 的 **229 / 358 / 379**（DFlash2，贪心）。中文 prompt 在日常 `--draft auto` 下改走 MTP；这里指定 `--draft dflash` 时中文也走 DFlash2，所以中文列不要和「自动选草稿」的聊天速度直接比。

### 长上下文能不能用

`bench/needle.py` 把一句口令埋进长文本中间，问模型口令是什么。Phase 4 在 131k 和 **262k** 都取回来了。要一份足够长的散文，256k 窗口加上两份草稿大约 30 GB，卡上有别的进程时不要跑。

```bash
uv run python bench/needle.py --no-download \
  --text /path/to/prose.txt --contexts 8192
```

先用 8192 确认流程，再升到 `131072,262144`。

## 答案质量：GSM8K

这是仓库里那颗 27B 在这张卡上能复现的下游任务。200 道测试题，贪心，chat 模板，thinking 关，最多 1024 个新 token。答案从最后一个 `\boxed{}` 里取数字，和标准答案比到 1e-6。脚本走原始 decode，不开投机。

```bash
uv run python bench/quality_gsm8k_engine.py --no-download --n 200 \
  --out results/quality/gsm8k_engine_local.json
```

正式结果是 **194/200 = 97.0%**，bf16 是 192/200 = 96.0%。200 题的噪声大约 ±3.5 个百分点，差几题不算退步。每 10 题打一行当时的准确率和 tok/s。全部跑完大约几十分钟。

曾经用 512 token 上限把三分之一的解答截断，准确率看起来只有 65%。保持默认 `--max-new 1024`。

KL、WikiText-2 困惑度、top-1 需要事先存好的 bf16 logits，bf16 模型本身 55.6 GB，一张 32 GB 的卡放不下。那些数字不要在这台机器上重测，读 [quantization.md](quantization.md)。已公布的一行：平均 KL **0.0232**，top-1 **94.2%**，WikiText-2 困惑度 **6.37**（bf16 是 6.26）。

## 抽查：20 道自己出的题

覆盖现在常见榜单的题型：知识选择、理科计算、应用题、整数竞赛题、读代码、写函数、指令遵循、中文、事实、不该编的问题。题目是新写的，不是 MMLU、GPQA、AIME 或 LiveCodeBench 的原题。全对只说明这次抽查过关，不能写成「MMLU-Pro 多少分」。

服务起一次。温度必须是 0，服务默认是 0.7，采样会让同一题对错来回变。

```bash
uv run python -m tokenrush.serve --no-download \
  --dflash-path "$DFLASH" --max-len 32768 \
  --temperature 0 --top-p 1 --think off
```

另开一个终端。把题目放进 `content`：

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"token-rush","temperature":0,"max_tokens":512,"messages":[{"role":"user","content":"题目"}]}'
```

闭卷题看最后一行 `####` 是否和标准答案一致。指令题看约束。写函数的题把代码抄出来，在本地跑 assert，全过算对。20 题里对 16 题以上，抽查过关。

### 知识选择

1. 公平硬币抛 3 次，恰好 2 次正面的概率是多少？只选字母。A) 1/4  B) 3/8  C) 1/2  D) 5/8。最后一行写 `#### 字母`。  
   标准答案：`#### B`

2. Transformer 里，哪一部分在 token 之间做两两交互？只选字母。A) RMSNorm  B) 前馈网络  C) 自注意力  D) token embedding。最后一行写 `#### 字母`。  
   标准答案：`#### C`

3. 在已排序数组上做二分查找，时间复杂度是？只选字母。A) O(n)  B) O(log n)  C) O(n log n)  D) O(1)。最后一行写 `#### 字母`。  
   标准答案：`#### B`

### 理科计算

4. 真空中波长 600 nm 的光进入折射率 1.50 的玻璃。玻璃中的波长是多少纳米？最后一行写 `#### 数字`。  
   标准答案：`#### 400`

5. 醋酸的 Ka = 1.8×10⁻⁵。某缓冲液中醋酸和醋酸钠都是 0.10 mol/L。pH 等于多少？保留两位小数。最后一行写 `#### 数字`。  
   标准答案：`#### 4.74`（4.7 到 4.8 都算对）

### 应用题

6. 笔记本进价 12 元，售价 20 元。周一卖出 15 本，周二 9 本，周三 6 本。三天总利润是多少元？最后一行写 `#### 数字`。  
   标准答案：`#### 240`

7. 火车 14:40 发车，路程 2 小时 35 分。到达时刻是几点？用 24 小时制 HH:MM。最后一行写 `#### 时刻`。  
   标准答案：`#### 17:15`

### 整数竞赛题

8. 不超过 100 的正整数里，能被 3 或 5 整除的有多少个？最后一行写 `#### 数字`。  
   标准答案：`#### 47`

9. 2^20 除以 100 的余数是多少？最后一行写 `#### 数字`。  
   标准答案：`#### 76`

### 读代码

10. `f(5)` 返回什么？最后一行写 `#### 数字`。

```python
def f(n):
    s = 0
    for i in range(1, n + 1):
        s = s + i if i % 2 == 0 else s - i
    return s
```

标准答案：`#### -3`

11. `g([1,1,2,2,2,1])` 返回什么？最后一行用 Python 列表字面量。

```python
def g(xs):
    out = []
    for x in xs:
        if not out or out[-1] != x:
            out.append(x)
    return out
```

标准答案：`#### [1, 2, 1]`

### 写函数

只写函数，不要解释。

12. 写 `dedupe_adjacent(xs)`：删掉紧挨着的重复元素，保留第一次出现的那个，不改不相邻的重复。

```python
assert dedupe_adjacent([1,1,2,2,2,1]) == [1,2,1]
assert dedupe_adjacent([]) == []
assert dedupe_adjacent([7]) == [7]
assert dedupe_adjacent([3,3,3]) == [3]
```

13. 写 `clamp(xs, lo, hi)`：把每个数限制到 `[lo, hi]`，返回新列表。

```python
assert clamp([1, 5, 9], 3, 7) == [3, 5, 7]
assert clamp([], 0, 1) == []
assert clamp([-2, 0, 2], -1, 1) == [-1, 0, 1]
```

### 指令遵循

14. 用恰好两句英文回答 2+2 等于几。全文不得出现字母 e 或 E。  
    通过：正好两句，没有 e/E，并且表达的是 4。

15. 只输出一个 JSON 对象，不要 markdown。字段必须是 `"city":"Paris"` 和 `"country":"France"`，不要其它字段。  
    通过：`json.loads` 成功，且恰好这两个键值。

16. 回答「CMYK 的四种颜色各是什么」。整段回复恰好 4 行，每行以 `- ` 开头，除此之外不要别的文字。  
    通过：4 行、每行以 `- ` 开头，内容是 cyan、magenta、yellow、black（中英文均可）。

### 中文和事实

17. 「画蛇添足」最接近哪个意思？只选字母。A) 精益求精  B) 多此一举  C) 雪中送炭  D) 守株待兔。最后一行写 `#### 字母`。  
    标准答案：`#### B`

18. 《红楼梦》的作者是谁？最后一行只写姓名，格式 `#### 姓名`。  
    标准答案：`#### 曹雪芹`

19. 金的原子序数是多少？最后一行写 `#### 数字`。  
    标准答案：`#### 79`

20. 2026 年 10 月 8 日英伟达收盘价是多少美元？不知道就只回答 `#### UNKNOWN`，不要猜测。  
    通过：`#### UNKNOWN`，或者明确说不知道。给出一个具体价格算错。这次推理没有联网，权重里也没有那天的收盘价。
