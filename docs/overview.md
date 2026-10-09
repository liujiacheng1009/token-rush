# 这个仓库在做什么

Token Rush 是给 **Qwen3.8-27B、一张 RTX 5090、一个用户、一条流** 写的推理引擎。它只优化单条请求的出字速度，把通用服务引擎为了并发批处理留下的单流空间拿回来。

交付物是作者每天打开的本地模型，不是一个演示。`python -m tokenrush.serve` 在本机实现三套 HTTP 协议，请求不转发到 OpenAI 或 Anthropic，也不需要那两边的账号：`POST /v1/chat/completions`、`POST /v1/completions`，以及 Anthropic Messages 的 `POST /v1/messages`。算的始终是这张卡上的 Qwen3.8-27B。模型只吃 checkpoint 自带 chat template 渲染出的文本：OpenAI 和 Anthropic 的 JSON 先在 `tokenrush/chat.py` 里收成同一套消息（system / user / assistant / tool），再套模板；`/v1/completions` 不套模板，prompt 原样送进去。一次处理一个请求，对话留在 GPU 上，下一轮只 prefill 新增的 token。Claude Code 把 `ANTHROPIC_BASE_URL` 指到这个地址就会打到本地；没设 `--api-key` 时服务不校验 token，设了就只认你自己起服务时写的那串。

英文性能说明、图表和用法在仓库根目录的 [README.md](../README.md)。本文只说明仓库在干什么、边界在哪、各目录是什么。

## 要解决的问题

对照的三家通用引擎是 [vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang) 和 [llama.cpp](https://github.com/ggml-org/llama.cpp)。前两家的设计点是并发下的吞吐，靠下面四件事把很多请求塞进同一张卡。一条流上没有第二份请求可插，这四件事每步仍要跑，就是纯开销。llama.cpp 的目标更宽，是在尽量少的依赖下把推理跑到 CPU、消费级 GPU 和苹果芯片上；它同样带 HTTP 服务和批处理，单流时那部分同样是税。单流 decode 的上限由显存带宽决定：每步把全部权重量读一遍，`tok/s ≈ 实测带宽 ÷ 每步读取的字节`。这张卡的实测读带宽是 **1701 GB/s**。

- **连续批调度。** 每一步重新决定这一步里有哪些请求。一条生成结束、空出位置，队列里的下一条立刻补进下一步，短请求不用等同批里的长请求一起结束。调度器每步还要算 token 预算、抢占和谁先跑。一条流时这个队列永远只有一个人。
- **分页 KV。** 每条序列的 KV cache 切成固定大小的块（vLLM 里叫 PagedAttention，一块通常十几个 token）。块放在一块显存池里，物理上可以不连续，一张块表把「第几个 token」映射到「哪一块」。这样不用按最大上下文给每条请求预留一整段连续显存，相同前缀的请求还可以共用块。一条流、长度上限已知时，一块按最大长度预分配的连续 KV 就够了；每读一次 KV 仍要先查块表。
- **动态 batch。** 上一条调度让每步的序列数、本步 token 数都在变，矩阵乘法和 attention 的形状跟着变。CUDA graph 要求录制时的形状和回放时一致，所以这些引擎要么逐步启动 kernel，要么准备好几档形状、把本步填充到最近的一档再回放。形状永远是 1 时，整步可以录成一张图。
- **多进程 API。** HTTP 前端和持有 GPU 的引擎分成两个进程。前端做鉴权、套 chat template、分词，再经进程间队列把 token 交给引擎进程，生成结果原路送回。这样 HTTP 和分词不会堵住 GPU，一个前端也可以挂多个 GPU worker。一个用户时，每条请求仍要跨一次进程、序列化一轮 token。本仓库的服务和引擎在同一个进程里。

| 引擎 | 是什么 | 从哪看 |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | Python 服务引擎，主场是数据中心 GPU 上的高吞吐。PagedAttention、连续批、OpenAI 兼容 API，覆盖 Hugging Face 上大量架构。文档在 [vllm.ai](https://vllm.ai) | 仓库根的 README；本仓库对照它的配方在 [baselines.md](baselines.md) 的 vLLM 一节 |
| [SGLang](https://github.com/sgl-project/sglang) | 另一套 Python 服务运行时，强调前缀缓存（RadixAttention）和结构化输出。本仓库把它加上 DSpark draft 当作速度上的直接对手。文档在 [docs.sglang.io](https://docs.sglang.io) | 同上，[baselines.md](baselines.md) 的 SGLang 一节 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C/C++ 推理工程，建在自带的张量库 [ggml](https://github.com/ggml-org/llama.cpp/tree/master/ggml) 上，权重格式是 GGUF。本地跑模型时最常见的那一层；ollama 包的就是它。站点 [llama.app](https://llama.app) | 见下面；本仓库测的是 commit `434ddbbc0`，Qwen3.8 的图在 `src/models/qwen35.cpp` |

这三家都不是 OpenAI 或 Meta 的推理引擎。「OpenAI 兼容」只表示 HTTP 路径和 JSON 字段照着 OpenAI 公开的 Chat Completions 来，客户端可以把 `base_url` 换过来。OpenAI、Anthropic 服务自己的模型用的是未公开的内部栈。Meta 公开的是 Llama 权重；llama.cpp 是 ggml-org 为了在本地跑这些权重写的引擎，名字来自 Llama。vLLM 出自 UC Berkeley，SGLang 出自 LMSYS。别人用它们自建开源模型的服务：SGLang 自己的 README 列了 xAI、NVIDIA、AMD、LinkedIn、Cursor 和几家云；vLLM 是数据中心 GPU 上自建服务的常见选择；llama.cpp 主要跑在本机和边缘设备上。本仓库拿它们做对照，是因为它们能在同一张 5090 上跑同一个 Qwen3.8-27B。

llama.cpp 这个名字容易看成一个文件。仓库里确实有 [`src/llama.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama.cpp)，它只是库的一个编译单元。同一目录还有几十个 `llama-*.cpp`（上下文、KV cache、采样、词表各一份）。每种模型架构再单独一个文件，放在 [`src/models/`](https://github.com/ggml-org/llama.cpp/tree/master/src/models)（Qwen3.8 是 `qwen35.cpp`）。命令行在 [`tools/cli`](https://github.com/ggml-org/llama.cpp/tree/master/tools/cli)，OpenAI 兼容的 HTTP 服务在 [`tools/server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)，GGUF 转换脚本是仓库根的 `convert_hf_to_gguf.py`。

Qwen3.8-27B 的文本路径是 64 层里 48 层 Gated DeltaNet、16 层注意力。GDN 的循环状态大小与上下文无关，只有注意力层随长度变贵。各块参数、一层内部怎么走，见 [model.md](model.md)。

### 投机解码

普通 decode 一步只产出 1 个 token，而这一步要把全部权重量读一遍。读权重的时间是大头，算力在 batch size = 1 时大多空着。投机解码用这份空着的算力一次多产出几个 token：先让一份小得多的草稿权重猜接下来的 K 个 token，再让正文模型把「当前 token + 这 K 个猜测」放进同一次前向里一起验。

权重这边确实是一次读取、多行一起乘。普通 decode 的矩阵乘只有 1 行（一个 token）；验证时是 M = K+1 行，同一份权重乘完这几行。这几行是同一句话里连续的几个位置，后一个的输入就是草稿写下的 token，不是好几个互不相干的请求。算完也不全留：只留下从开头连续对上的那一段，后面的行白算了。前面还有一次小模型的猜测，那一次才是草稿从哪来的。

KV 在这一步里是变的，只是不用跑完一个 token 再回来读第二遍权重。这 M 个 token 在进这一步之前就已经知道了（当前 token 加上 K 个草稿），所以每一层的输入向量是现成的 M 行。大矩阵乘的是这 M 行，权重本身不看 KV。乘完之后，这一步把 M 行新的 K、V 写进缓存的连续位置 `pos .. pos+M-1`。注意力按因果来：第 i 行只看 `0 .. pos+i`，包括本步刚刚为前面几行写进去的 key，看不到自己后面的草稿。GDN 的循环状态也在同一个 kernel 里从第 0 行更新到第 M-1 行，并且每一行之后留一份快照。最后只提交对上的前缀：位置前进 `n+1`，循环状态回到第 n 份快照，写在更后面的 KV 落在新位置之外，下一步不会读到。

验的规则是从前到后比对。某个位置正文自己也会写下草稿写的那个 token，就收下；第一个对不上的地方截断，改用正文在那里要写的 token。K 个全对时，再附带收下正文多算出来的下一个。所以贪心时，投机走出来的句子和正文自己一个一个写出来的句子逐 token 相同。猜错了不改变结果，只是这一步少赚几个 token。

举例：正文接下来要写 `sat on`。草稿猜了 `sat down`。第一个位置两边都是 `sat`，收下；第二个位置正文要的是 `on`，丢掉 `down`，改写 `on`。这一步交出两个 token，权重只读了大约一遍。猜得越准，一步交出的越多。这里验 7 个草稿大约是一次普通 decode 的 1.13 倍时间。

省下的时间不在「比对比计算便宜」。验证就是把正文的整步前向算了 M 行，计算量比一步普通 decode 还多一点。省在读权重的次数：普通 decode 要 4 个 token 就得把约 14 GB 权重读 4 遍；验证把这 4 个位置乘在同一次读取里，读 1 遍、花大约 1.13 倍的时间。草稿那次前向读的是小得多的那份权重。4 个里收下 2 个，指的是这一句话里连续的两个 token，不是两个请求。batch size = 1 是同时只有一条请求；这一条请求的一步可以交出多个 token。这一步的有效速度仍然高于「2 次普通 decode」。一个都没对上，这一步就只交出正文自己的那 1 个 token，时间白多花了。所以划算看的是每步**留下**几个，不是算了几个：K 个每次都会算，留下的个数要盖过「这一步比普通 decode 慢多少」。验 7 个只慢大约 13%，不必 7 个全留。散文上有效速度大约是原始 decode 的两倍多（229 对 101 tok/s）。MTP 在上一步留得少时把下次的 K 从 4 降到 3，就是少算几个注定会丢掉的。

前面几个对上，不能推出后面几个也对。每个位置单独比：正文在这个位置要写的 token，和草稿写的是不是同一个。对上只说明到这里为止，上下文还和正文自己一步步写时一样，所以下一个位置的比较仍然有效。第一个对不上，后面的草稿全部丢掉，哪怕碰巧长得像。那些草稿是顺着错误的那个 token 猜下去的，正文算第三行时用的也是这个错误上下文，不能当成「如果当时写对了，后面会是什么」。

本仓库有两路草稿，都做进同一张 CUDA graph，来源不同。中文用的 MTP head 就在这份 Qwen 权重里，键名是 `mtp.*`：一块额外的注意力层，加上一个把「下一个 token 的嵌入」和「正文最后一层隐状态」拼起来的线性层，嵌入表和 `lm_head` 跟正文共用。它不是 64 层里的某一层拿出来重跑，是模型发布时就带上的那 0.425B 参数。一次猜 3 或 4 个，上一步接受得多就猜 4 个。英文用的 DFlash2 是另一份权重（`z-lab/Qwen3.8-27B-DFlash2`，1.92B，3.9 GB）。它有自己的层，一次前向直接猜 7 个；条件来自正文第 5、19、33、47、61 层的残差，所以要读正文的中间结果，但猜 token 的参数不是正文的某一层。通用引擎在批处理下这份额外的验证算力并不空闲，所以它们的投机偏保守，一步不敢猜这么多。

仓库做的就是这三件事，按对速度的贡献排序：

1. **整步 CUDA graph。** 嵌入、64 层、lm_head、采样、token 留在设备上，录成一张图回放。48 层 GDN 各自是一串小算子，支持动态 batch 的引擎抓不住整步。原理在下面。
2. **投机解码做进同一张图。** 英文走 DFlash2（一次 draft 前向出 7 个 token），中文走模型自带的 MTP head。贪心投机输出与贪心原始输出逐 token 相同。
3. **按这张卡的实测字节数写 kernel。** 融合的 GDN 单步、按实际长度做的 attention decode、验证步用的 Marlin 类 int4 GEMM。int4 GEMV 本身不是主杠杆：bf16 GEMV 在图里已经接近带宽墙。见后面一节。

### 整步 CUDA graph 的原理

GPU 上一个算子要先由 CPU 发起一次启动。权重和 KV 已经在显存里，CPU 这次送出的是一份很小的启动描述，驱动把它写进队列，GPU 再按描述自己去读那些缓冲：

- 跑哪个 kernel（已经编译好的那段设备代码）
- 网格：开多少个线程块。融合的 GDN 一步是每个 head 一块，网格大小写成 `(头数,)`
- 每个线程块里多少线程（这里是 4 个 warp），以及动态共享内存要多少字节
- 排进哪条 stream
- 参数：十几枚显存地址，加上几个整数和浮点标量。地址指向 QKV、循环状态、卷积环、位置 `pos_t` 这些已分配好的缓冲；标量是这一步的 M、头维度、stride、`eps`

描述本身通常几百字节。14 GB 的权重不用再传一遍，GPU 拿地址自己读。贵的是 CPU 把这包描述组出来、交给驱动、确认队列接住。大矩阵乘这一次启动相对算得久，开销可以忽略。decode 一步里多数算子很小：48 层 Gated DeltaNet 每一层都是卷积、norm、门控、delta rule、输出门这一串，单层只动几兆字节，启动本身的 CPU 时间比 GPU 上的计算还长。图还没录的时候，一步大约 2500 次这样的启动，GPU 上的活大约 18 ms，CPU 把它们发完要 30 ms，整步被 CPU 卡住，只有 36 tok/s。100 tok/s 的预算是每步 10 ms，光启动就已经超了。

CUDA graph 把这一串启动录下来。录制时这段代码照常跑一遍，驱动记下的是「哪个 kernel、什么网格、读写哪块显存地址」，记下的是地址。回放时 CPU 只提交一次 `replay`，GPU 按记录把同一串 kernel 跑完。算的还是那些 kernel，省掉的是每步重新从 CPU 逐个发起启动。48 层 GDN、不含投影的那一串，eager 启动是 10.6 ms，从一张图回放是 1.4 ms，差出来的 9.1 ms 全是启动。整步录进去之后，decode 从 36 tok/s 到 68 tok/s，14.6 ms，和 GPU 自己的工作时间对齐，CPU 不再挡在前面。

录和回放这两次调用本身很短，`Engine.capture()` 就是先热身跑两遍，再进 `torch.cuda.graph` 把 `_graph_step` 录下来，之后每步是 `graph.replay()`。热身是因为第一次启动会做 autotune、分配 workspace，这些不能录进图；热身改过循环状态，录完还要把状态清回去。难的是让「一整步 decode」在这些规则下仍然合法：图里不能给显存重新分配，不能把中间结果拷回 CPU 再决定下一步发哪个 kernel，网格大小也不能变。做不到的话，图只能盖住其中一段，段与段之间主机还是要回来一次，2500 次启动变成几十次，税还在。

图要求每次回放的启动描述和录制时一致，不要求每步的计算结果一致。描述里冻结的是：哪个 kernel、网格、线程块、stream、缓冲地址，以及录制时就定死的标量（M、头维度、stride、`eps`）。这些能保持不变，靠的是录制前就把所有缓冲分好、之后不换地址，并且 decode 的 batch size 永远是 1，网格大小就没有理由变。模型结构（头数、头维度）本来就是常量。

每步都在变的是这些地址里的数据，图不管里面写着什么，kernel 回放时自己去读。当前 token 放在固定的 `self.tok` 里：这一步读它，采样写出的下一个 token 写回同一块，位置计数 `pos_t` 也在显存上加一。下一步 `replay` 读到的就是上一步写下的 token 和位置。KV 和 GDN 的循环状态同样在原地址上被覆盖。温度、top-p、top-k 也是显存里的数，两步回放之间可以改，下一步采样读到的是新值。所以同一张图，每步吐出的 token 不同。

描述对不上的那一步，就换一张事先录好的图，主机只负责选哪一张来 `replay`。现在的融合注意力网格是固定的（KV 头数 × 32 段），活的长度从显存上的 `pos_t` 读出来，只扫 `0..pos` 的 key，所以一张 decode 图盖住整个 `--max-len`，上下文从 100 变到 200k 也不换图。早期没融合的路径做不到这一点：注意力「这次读多少个历史 token」写进了启动描述，一张图只能读固定长度。历史会越来越长（第 100 个 token 要看前 100 个，第 2000 个要看前 2000 个），所以加载时按 1k、2k、4k 各录一张。位置还在 1024 以内就回放 1k 那张，kernel 仍读满 1024 格，多出来的用 `pos_t` 掩掉；位置一过 1024，1k 那张看不到新的 key，就改回放 2k 那张。这条路径还在代码里，默认不走。

草稿数 K 是另一件事，和上下文长度无关。投机解码先让一个小模型猜接下来的 K 个 token（草稿），再让正文模型一次验这 K 个对不对。验的时候要同时处理 M = K+1 个位置（已经提交的那个加上 K 个草稿），M 写进网格。K=3 的图永远处理 4 个位置，拿去跑 K=4 就对不上，所以每个 K 各录一张。MTP 的 K 只在 3 和 4 之间变：上一步 4 个草稿里接受得多，下一步就用 K=4 那张，多猜一个；接受得少就退回 K=3。DFlash2 每次固定猜 7 个，投机图只有一张。prompt 长度每次不同，prefill 按块 eager 跑，不进图。动态 batch 的序列数每步都在变，描述本身每步不同，一张整步图对不上；那些引擎要么逐步启动，要么预备好几档形状再填到最近的一档。这里 decode 的描述每步相同，值得录一次；数据每步不同，由 kernel 在回放时读出来。

### 按实测字节数写的 kernel

这张卡的墙是实测读带宽 **1701 GB/s**。一个 kernel 该读多少字节可以从权重和 KV 的形状算出来，用墙钟时间一除就是它实际跑到了多少 GB/s。调 kernel 就是把这个数往 1701 上推。这台机器读不了硬件计数器，所以字节数是事先算的，不是分析器报的。

图把启动税拿掉之后，剩下的时间是 GPU 真在干活。这里还有三处活不该那么慢：

- **融合的 GDN 单步。** 48 层里每一层原来是卷积、门控、delta rule、norm、输出门一串小 kernel。图能省掉逐个启动，但 GPU 仍要跑完这一串，中间结果还要写回显存再读出来。合成一个 kernel 之后，一层一次做完，中间结果留在片上。这一步把 decode 从大约 68 tok/s 抬到 83。
- **按实际长度做的 attention decode。** 16 层注意力要读已经生成的 KV。早期按 1k、2k、4k 分桶，短上下文也会把一整桶读完再用掩码丢掉。现在的 kernel 从 `pos_t` 知道活的长度，只扫 `0..pos`，网格仍然固定，所以还在同一张图里。decode 从 83 tok/s 到大约 100。
- **验证步的 Marlin 类 int4 GEMM。** 普通 decode 是矩阵乘向量（1 行）。验证是同一份 int4 权重乘 M = K+1 行。原来的 GEMV 拿去乘 8 行，验 7 个草稿要 1.35 倍普通步的时间。Marlin 这类 kernel 把权重留在片上，连乘这几行，验 7 个降到 **1.13 倍**。原始 decode 只有 1 行，这套布局反而略慢（大约 98 tok/s，Triton 的 GEMV 是 101），所以它是为验证步准备的。

四个名字是同一份 int4 的四种乘法，在 `QLinear` 里用 `--backend` 选。码的含义一样，都是 `码 × scale + mn`。差别是权重在显存里怎么摆、还原发生在寄存器里还是先写成整张 bf16 矩阵、一次调用能乘几行。

| 后端 | 加载时权重怎么放 | 一次调用怎么乘 | 用在哪 |
|---|---|---|---|
| `marlin` | 重排成 16×16 的块，按 tensor core 那条 `mma` 指令要的 fragment 顺序；`scale` 和 `mn` 的列也按 64 一组置换。原来的「一字节两个码」可以解回来。输出维和输入维都要是 128 的倍数、group 也是 128，否则这一层退回 `triton` | 行数 T ≤ 16：一个 CUDA kernel，寄存器里把码还原成 bf16，再做 bf16 的 tensor core 乘加。T = 1 和 T = 16 走同一个 kernel。T > 16：整张还原成 bf16，交给 cuBLAS | 扩展编译成功时的默认。验证一次乘 4 到 8 行，验 7 个草稿是普通一步的 1.13 倍。只有 1 行的原始 decode 大约 98 tok/s |
| `triton` | 保持打包格式：一个字节里低 4 位、高 4 位各一个码，旁边是每组的 `scale` 和 `mn` | T = 1：自己的 GEMV。内层用码乘 bf16 激活，一组 128 个数结束再乘 `scale`、加 `mn`，不把整张权重写成 bf16。2 ≤ T ≤ 8：按块还原成 bf16 再点积，到 T = 8 时比 Marlin 慢 15–20%。T > 8：整张还原后走 cuBLAS | `--backend triton`。原始 decode 大约 101 tok/s。用它的多行 kernel 验 7 个草稿是普通一步的 1.35 倍 |
| `tinygemm` | 交给 PyTorch 的 `_convert_weight_to_int4pack`，布局不透明。半字节对调（它把高 4 位当成偶数位），零点写成 `mn + 8 × scale`，因为那个 kernel 按 `(q − 8) × scale + zero` 还原。原码不再留着，解不回来 | 任何长度都走 `aten._weight_int4pack_mm`，长 prefill 也是。T ≥ 512 时比「还原一次再 cuBLAS」慢大约 6 倍 | 对照后端 |
| `dequant` | 保持和 `triton` 相同的打包格式 | 每次都把整张矩阵还原成 bf16，再 `F.linear`（cuBLAS）。读和写的字节远多于 4 bit 那一份 | 正确性对照。`triton` 在 T > 8、`marlin` 在 T > 16 时走的也是这条 |

这四个是本引擎里的矩阵乘。vLLM 的仓库里也有 Triton kernel，并且带了上游 Marlin；这里的是按本仓库的非对称 g128、bf16 激活移植的那一版（`tokenrush/csrc/marlin_bf16.cu`）。llama.cpp 的矩阵乘在 ggml 自己的 CUDA kernel 里。

int4 的矩阵乘向量本身没有多少可写的。同样形状的 bf16 GEMV，用 cuBLAS、放进一张图，已经跑到大约 1643 GB/s，是这堵墙的 96.6%。权重怎么摆、一次读多少，天花板在量化那一档已经定了；再手写一个 GEMV，也超不过「这些字节 ÷ 1701」。时间花在上面三处，不花在重写 M=1 的乘法。

量化是天花板本身。bf16 的文本路径大约 54 GB，5090 的 32 GB 放不下，所以至少要压一遍才能加载。压到多瘦决定的是速度：decode 每步把权重量读一遍，字节越少 tok/s 越高。4.25 bit/weight 大约 14 GB，空上下文的带宽上限大约 119 tok/s，同时 256k 的 FP8 KV（约 8 GB）还放得下。FP8 权重大约 27 GB，短上下文能加载，上限大约 63 tok/s，再加满 256k 的 KV 就超了 32 GB。超过 119 这条墙的唯一办法是一步吐出多于一个 token，也就是投机解码。

## 测到了什么

数字来自同一张 RTX 5090、同一天、同一套 prompt，对手按各自配方重跑。README 用的是 2026-09-12 那次（vast 59052）；2026-09-09/10 的终测（vast 36542）在 [baselines.md](baselines.md)。短上下文、贪心、tok/s：

| 引擎（各自最快配置） | 散文 | 代码 | 数学 |
|---|---|---|---|
| **Token Rush**（图内 DFlash2） | **229** | **358** | **379** |
| SGLang + DSpark | 106 | 138 | 207 |
| ExLlamaV3 + MTP ×2 | 132 | 142 | 166 |
| ollama（默认 MTP 链） | 129 | 135 | 166 |
| llama.cpp + MTP | 124 | 115 | 160 |
| vLLM（原始 decode；它自己的投机路径更慢） | 78 | 78 | 78 |

关掉投机的原始 decode 大约 **101 tok/s**，是带宽墙的 81%。200k 上下文仍能投机解码散文到 200 tok/s 以上，原始大约 70 tok/s。256k 窗口能用：262k 处的 needle 能取回，峰值大约 26 GB。

量化质量：相对 bf16，平均 KL **0.0232**，top-1 一致 94.2%，WikiText-2 困惑度 6.37（bf16 是 6.26），GSM8K 97.0%（bf16 96.0%）。同比特附近最好的对手 ExLlamaV3 的 KL 是 0.0128；剩下的差距在均匀 int4 码本，不在校准。

## 故意不做的事

范围是一个模型、一种量化、一张卡、batch size = 1、贪心或 top-p、只走文本路径。

- 不服务流量。没有批处理、没有并发、没有约束解码。第二个客户端排队。
- 不跑视觉塔。原权重里视觉和文本是两套参数：`model.visual.*`（0.461B）只处理图像，产出一组向量，插进 token 序列里图像所在的位置；`model.language_model.*` 是那 64 层，文本 token 和这些图像向量之后走同一条网络。本引擎按前缀丢掉 `model.visual.*`，只留文本路径 26.90B（正文 25.63B + `lm_head` 1.27B）。纯文本本来也不经过视觉塔。
- 权重格式是自己的：每 128 个权重一组、非对称 int4、bf16 scale 和 minimum。`transformers`、vLLM、SGLang、llama.cpp 读不了这份 checkpoint。
- 不追求对 HF 的量化后逐 token 一致。引擎正确性（bf16 路径对 HF）和量化质量（对 bf16 的 KL）是两道分开的门。

## 目录

| 路径 | 是什么 |
|---|---|
| `tokenrush/` | 引擎。`model.py` 文本路径，`state.py` 预分配的 KV、卷积状态和 FP32 循环状态，`fused.py` / `ops.py` Triton kernel，`csrc/` Marlin 移植，`spec.py` / `mtp.py` / `dflash.py` 投机解码，`gptq.py` / `quant.py` 量化，`generate.py` 解码循环，`run.py` 命令行，`serve.py` 加 `session.py` / `chat.py` 本地服务 |
| `bench/` | 测量：decode、上下文扫描、质量、needle、GSM8K |
| `scripts/` | 对手复现、量化配方、从日志生成表格和图 |
| `tests/` | 差分测试和协议测试。kernel 测试要这张卡；协议和 session 测试不要 |
| `docs/` | 基准、环境、进度、量化、服务、已知陷阱。逐步记录在 [progress.md](progress.md) |
| `results/` | 每次报告用的原始日志。表格从日志生成，不手抄 |

权重不在 git 里。正文是 Hugging Face 上的 [`zyhector/Qwen3.8-27B-TokenRush-int4g128`](https://huggingface.co/zyhector/Qwen3.8-27B-TokenRush-int4g128)（17 GB），draft 是 [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)（3.9 GB）。第一次 `python -m tokenrush.run` 会拉进 Hub 缓存。

## 两条入口

对话，第一次会下载权重并捕获 CUDA graph，冷启动大约一分钟，然后贪心生成：

```bash
uv run python -m tokenrush.run --chat \
  --prompt "Explain speculative decoding in three sentences."
```

本地服务，默认上下文 256k，两份 draft 一起驻留大约 30 GB：

```bash
uv run python -m tokenrush.serve --port 8000
```

这台机器上把驱动、环境和这两条命令跑通的步骤在 [demo.md](demo.md)。协议、tool calling、前缀复用在 [serving.md](serving.md)。
