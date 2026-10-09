# 本机写代码：方案

要的是在本机 vibe coding：说一句话，模型自己读仓库、改文档或代码、跑命令，把 diff 交回来。编辑器或终端把这些当成工具，工具在本机执行，结果塞回对话，模型再写。本仓库的引擎已经是那个模型：`python -m tokenrush.serve` 说 OpenAI Chat Completions 和 Anthropic Messages，Claude Code 已经在这上面读过文件、改过文件、跑过命令（[serving.md](serving.md)）。缺的是一个开源客户端。

不做 Tab 补全，也不在本仓库里写编辑器。推理仍是这一条流。

## 现在各家怎么做

| | 客户端开源 | 接到本服务 | 人用的表面 | 和 Cursor / Codex 的距离 |
|---|---|---|---|---|
| Cursor | 否 | 否，走它自己的路由 | Agent 改文件、终端；另外还有 Tab 补全和代码库索引 | 对齐的是改文件这一段，不是 Tab |
| Codex CLI | 是（OpenAI） | 对不上 | 终端，沙箱，批准后再写 | 只认 Responses API。本服务没有 `POST /v1/responses` |
| Claude Code | 否 | **已经接上** | 终端里读、改、跑 | 体验最近。客户端不是开源的 |
| [OpenCode](https://opencode.ai) | 是 | Chat Completions，自定义 `baseURL` | **浏览器**（`opencode web`），以及终端、桌面、编辑器插件；语言服务器；Plan / Build | **第一选择** |
| [Cline](https://github.com/cline/cline) | 是 | 设置里选 OpenAI Compatible | VS Code 侧栏，逐步批准，检查点可回退 | 想留在编辑器里时用它，同一套地址 |
| [Aider](https://github.com/Aider-AI/aider) | 是 | OpenAI 兼容 | 终端，每次改动落成一个 git commit | 模型写 diff 不稳时更合适。界面不像 Cursor |
| Continue | 是，主仓库已只读 | 能接 | 侧栏，另做 Tab | 不作为默认 |

OpenCode 的自定义供应商用 `@ai-sdk/openai-compatible`，请求打到 `/v1/chat/completions`。换成 `@ai-sdk/openai` 会去打 `/v1/responses`，本服务没有这条路由。Codex CLI 的 `wire_api` 目前只接受 `responses`，所以它不能靠改一个 base URL 接上来。要 Codex 那个客户端，得先在服务里做一整套 Responses 协议，收益只是换皮。

## 这张卡决定了体验的形状

- **一次一个请求。** 写文档和改代码走同一条生成。27B 加两份草稿已经占满这张 5090；旁边还有别的程序时，窗口是 `--max-len 32768`。
- **不做第二份权重，不做向量索引。** 仓库上下文用客户端自己的检索：OpenCode 加载对应语言的 LSP，再加文本搜索。嵌入模型要另占显存，和这一条流抢卡。
- **整段重发才快。** 服务比对新 prompt 和显存里的前缀，只 prefill 多出来的 token（[serving.md](serving.md)）。客户端每轮改写系统提示，前缀对不上，下一次就整段重算。代码任务把整个仓库塞进上下文，慢的是这次 prefill，不是页面。
- **工具参数没有约束解码。** 模型按 Qwen 的模板写下 `<tool_call>`，`chat.OutputParser` 再解析。写坏的 JSON 会失败；客户端把错误当工具结果塞回去，模型可以重试。Claude Code 那条路径已经这样跑通过。
- **质量上限是 Qwen3.8-27B。** 方案不换模型。界面换得再像，写错补丁的次数仍由这个权重决定。代码上的出字速度是现成的：投机解码在代码上测到过每秒几百 token（仓库开头的目标表）。交互慢，多半是上下文太长或工具回合太多，不是 decode。

## 第一版：OpenCode 指到现有服务

配置放在用户目录（`~/.config/opencode/opencode.json`），不进本仓库。地址和本机窗口跟这次启动走，和 SearXNG 的地址一样留在机器上。

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "tokenrush/token-rush",
  "provider": {
    "tokenrush": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Token Rush",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "local"
      },
      "models": {
        "token-rush": {
          "name": "Qwen3.8-27B",
          "tool_call": true,
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  }
}
```

`token-rush` 是服务的 `--served-name`，请求里的模型名服务不校验，只回显。`limit.context` 写成这次的 `--max-len`。卡空出来、服务去掉 `--max-len 32768`（默认 262144）时，把这里改成同一个数。`baseURL` 停在 `/v1`，不要写成 `/v1/chat/completions`。没设 `--api-key` 时 `apiKey` 随便写一串非空；设了就写那一串。

服务要先就绪。冷启动是加载权重和捕获 CUDA graph，终端打印地址之后再开 OpenCode。

```text
人                    OpenCode                         serve / GPU
 |  一句任务            |                                  |
 |                     |-- 读文件、LSP、搜索，都在本机 ---->|  （这一步不占卡）
 |                     |  POST /v1/chat/completions       |
 |                     |  messages + tools, stream        |
 |                     |                                  |-- 前缀对上就只算新增的
 |                     |<-- tool_calls 或正文 ------------|
 |                     |-- 本机改文件或跑命令 ------------>|  （命令不进 CUDA graph）
 |                     |  工具结果接回 messages，再请求    |
 |<-- diff、终端输出 ---|                                  |
```

文件和命令留在 OpenCode 进程里。GPU 只看见渲染后的文本。网页聊天那套 `server_tools`（搜索在服务里执行）不要打开：写代码的工具必须在 OpenCode 里跑，服务只解析 `tool_call` 并交还。

网页的单人座位不拦这条路径。座位只包住页面和带 `server_tools` 的生成。OpenCode 和页面同时生成时，后来的请求在服务里排队，不会并行占卡。

## 浏览器里切换

OpenCode 有浏览器界面。`opencode web` 在本机再起一个 HTTP 服务。打开的页面是 Cursor 里用来 vibe coding 的那一块：文件树、打开文件、这一轮的 diff、终端、对话。插件市场、调试器、Tab 补全不在这个页面里。终端界面是连到同一个服务的另一个客户端。文档和代码写在**启动时所在的目录**，也就是这台机器上的仓库，不进聊天用的 `chats.json`。

浏览器只开聊天这一页，地址仍是 `:8000`。顶部「对话 / 写代码」在同一页里切换。写代码时收起「新对话」，左边是目录，中间是文件，右边是对话。工具栏的「打开目录」选择服务器上的一个目录，模型只能改这个目录里的文件。

中间是 Monaco，侧栏是活动栏和就地展开的文件树。不另起一个编辑器进程。

## 编辑器长得像 VS Code

中间那一块用 [Monaco Editor](https://github.com/microsoft/monaco-editor)（MIT）。它就是 VS Code 里的编辑器，从 VS Code 拆出来单独发布。换上之后有行号、按文件名上色、折叠、括号匹配、小地图、Ctrl+F 查找。页面仍是这一个 HTML，不引入构建。脚本从 jsDelivr 拉取固定版本，和页面上的 KaTeX 一样；拉不到时留着现在的文本框。

侧栏和工具栏没有对应的「整块 VS Code 外壳」可以嵌。VS Code 的活动栏、文件树、标签页没有单独发布成组件。这一步只换编辑区。新建、删除、重命名和 Git 在下一节，用浏览器里的整份 VS Code。

外壳仍是我们自己的，只借图标和编辑器：

| 位置 | 之前 | 现在 |
|---|---|---|
| 中间 | 文本框 | Monaco，主题 `vs-dark`，按扩展名选语言 |
| 最左一条 | 没有 | 活动栏，图标用 [VS Code 的 Codicons](https://github.com/microsoft/vscode-codicons)（MIT）：资源管理器、搜索 |
| 文件树 | 点进文件夹会整棵换成那一层 | 就地展开、缩进，文件夹前有三角，文件按扩展名上色 |
| 标签 | 一行文件名 | 打开过的文件各一个标签，未保存有圆点 |
| 工具栏 | 「打开目录」「保存」文字按钮 | 图标按钮，路径做成可点的面包屑 |
| 右侧对话 | 留着 | 不改成编辑器的一部分 |

文件仍走现有的 `/fs/list` 和 `/fs/file`。Monaco 只负责显示和编辑，保存还是 Ctrl+S 写回服务器。语言服务、调试器、插件市场不做。没有新建、删除、重命名，也没有 Git。

## 下一版：用浏览器里的 VS Code

Monaco 补不出这些。新建文件、删除、重命名、Git 状态和提交都在 VS Code 的工作台里，没有跟编辑区一起发布。继续给这一页的侧栏加按钮，做出来仍是一套更小的外壳。下一版直接用浏览器里的 VS Code。

用 [code-server](https://github.com/coder/code-server)（MIT）。打开之后就是 VS Code：资源管理器里可以新建、删除、重命名，源代码管理里是 Git，还有终端和按内容搜索。扩展从 Open VSX 装。[OpenVSCode Server](https://github.com/gitpod-io/openvscode-server) 是同一类程序，先用 code-server。

公网仍然只有 `:8000`。code-server 只听 `127.0.0.1`，由现在的服务反代到 `/vscode/`。顶部「对话 / 写代码」留着。写代码时，这一页剩下的区域是它，自带的文件树和 Monaco 先不显示。

右侧的对话用 Cline，装在 code-server 里面。供应商选 OpenAI Compatible，Base URL `http://127.0.0.1:8000/v1`，模型 `token-rush`，密钥写一串非空。读文件、改文件、跑命令发生在这台机器上，生成仍进现在这一条流。不做 Tab 补全。

等 tokenrush 打印就绪后再起。这一版 code-server 没有 `--base-path`，它自己把 VS Code 挂在 `/vscode`。8080 上已经有别的程序，所以听 8088。认证用 none：它的 `/login` 会和聊天页的座位登录撞车，座位已经挡住 `/vscode`。从 Cursor 的终端里启动时要去掉 `VSCODE_IPC_HOOK_CLI`，否则它以为该把目录交给已经开着的编辑器，然后自己退出。

```bash
env -u VSCODE_IPC_HOOK_CLI code-server --bind-addr 127.0.0.1:8088 --auth none --disable-telemetry --disable-update-check --disable-workspace-trust /home/jesse/workspace/token-rush
```

它能改服务器上的文件、能跑命令，所以不要改成 `0.0.0.0`。服务里把 `/vscode` 和 `/_static` 转到 `127.0.0.1:8088`，这两条写在 OpenCode 的兜底反代前面，避免被转到 4096。

自带的 `/fs` 和 Monaco 先留在仓库里。这一页切过去、Cline 能改一个文件之后再收。

OpenCode 仍要单独启动，先等 tokenrush 打印就绪：

```bash
cd /home/jesse/workspace/token-rush
OPENCODE_SERVER_PASSWORD=bestcalib opencode web --hostname 127.0.0.1 --port 4096
```

密码和聊天页的座位密码是同一串，只放在这台机器的环境变量里，不写进仓库。它能改服务器上的文件、能跑命令，所以不要改成 `0.0.0.0`。

## 这一版做什么

| | 用什么 | 引擎 |
|---|---|---|
| 聊天页上切到写代码 | 同一页里嵌 code-server（`/vscode/`）。新建、删除、Git、终端是 VS Code 的。右侧对话用里面的 Cline | 仍是这一条流。自带的 Monaco 先留着，不显示 |
| 先计划再改 | OpenCode 自带的 plan / build | 不改 |
| 批准、回退 | 客户端的权限和 git | 不改 |
| 找该改的文件 | LSP 加搜索 | 不改。不做向量 |
| 想留在编辑器侧栏里、逐步点批准 | 再换 Cline：供应商选 OpenAI Compatible，Base URL `http://127.0.0.1:8000/v1`，模型 `token-rush`，密钥写一串非空 | 不改，同一进程 |

Cline 是编辑器里的另一种客户端，不是另一套推理。Aider 留作退路：若 27B 经常把整文件改坏、但 diff 格式稳定，用它的编辑格式，仍指同一个 `/v1`。Codex 那个客户端要对上，得先实现 `/v1/responses`，这一版不改协议。

## 接上之后若工具调用对不上

先用真实的一轮看服务收到的 JSON，再改 `tokenrush/chat.py` 或 `tokenrush/serve.py` 里差的那一个字段。不预先移植 Responses。

已经为 Claude Code 处理过、OpenAI 这条也可能再碰到的两类情况：客户端把一条系统消息放在 `messages` 末尾（Qwen 模板不收，要收成 user）；工具结果和调用对不上 id（模板没有 id，按顺序配对）。OpenCode 若依赖流式 `tool_calls` 的 `index`，或把 `tool_call_id` 原样传回，就补这一处。约束解码仍然不做：采样在 CUDA graph 里，语法要的是 kernel，不是多一个 HTTP 字段。

## 不做什么

- 不做 Tab 补全，不加载第二份权重，不在这张卡上做嵌入索引。
- 不再给自带的 Monaco 外壳加新建、删除和 Git。这些用 code-server。
- 不实现 `/v1/responses`。
- 不把 OpenCode / Cline 的配置提交进仓库。
- 不改 bs=1。

## 怎么算做完

1. 服务已在跑。聊天页上的「写代码」打开 OpenCode 的网页，工作目录是本仓库。
2. 在这个网页里一句任务改文档：给 `docs/coding.md` 补一小节，它读文件、改文件，diff 能看，文件落在服务器的仓库里。
3. 一句任务改代码：给 `tests/test_serve.py` 里一个现有断言旁边再加一条检查，它改完并跑 pytest。
4. 工具结果回到对话之后，下一轮只 prefill 新增的 token，而不是把整段仓库重算一遍。
5. 页面正在生成时，OpenCode 的请求排队，显存里仍然只有一条流。
