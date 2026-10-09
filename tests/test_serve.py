"""The HTTP layer without a GPU: a stub Session that "generates" a canned reply
token by token, the real tokenizer (skipped when it is not on disk), FastAPI's
test client. Checks the shapes of both protocols, streaming and not, tool
calls, stop sequences, usage and the Anthropic event sequence."""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, __file__.rsplit("/", 2)[0])


def _tok():
    from tokenrush.weights import DEFAULT_REPO, resolve_model
    for p in ("/workspace/models/Qwen3.8-27B-int4g128-gptq-mse", DEFAULT_REPO):
        try:
            d = resolve_model(p, download=False)
        except SystemExit:
            continue
        if os.path.exists(os.path.join(d, "tokenizer.json")):
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(d)
    pytest.skip("checkpoint tokenizer not available")


class StubSession:
    """Replies with `reply` (a string) for every prompt, 3 tokens per step, then
    <|im_end|>. Records the prompt it got."""

    def __init__(self, tok, reply):
        self.tok = tok
        self.reply = reply
        self.stop_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
        self.max_len = 4096
        self.prompts = []
        self.stats = types.SimpleNamespace(as_dict=lambda: {"reused_tokens": 7})

    def pick_mode(self, text, mode):
        return "dflash"

    def forget(self):
        pass

    def generate(self, ids, max_new, mode, temperature, top_p, top_k, seed):
        self.prompts.append((list(ids), max_new, temperature, top_p))
        toks = self.tok.encode(self.reply, add_special_tokens=False) + [self.tok.convert_tokens_to_ids("<|im_end|>")]
        for i in range(0, len(toks), 3):
            yield toks[i:i + 3]


TOOLS_A = [{"name": "Bash", "description": "run", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]
TOOLS_O = [{"type": "function", "function": {"name": "Bash", "description": "run", "parameters": TOOLS_A[0]["input_schema"]}}]
REPLY_TOOL = "I'll check.\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nwc -l docs/traps.md\n</parameter>\n</function>\n</tool_call>"


def _client(reply):
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    tok = _tok()
    sess = StubSession(tok, reply)
    args = types.SimpleNamespace(api_key=None, think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[])
    cfg = types.SimpleNamespace(eos_ids=(tok.eos_token_id,))
    return TestClient(build_app(sess, tok, cfg, args)), sess


def _sse(text):
    return [json.loads(l[5:]) for l in text.splitlines() if l.startswith("data:") and "[DONE]" not in l]


def test_openai_chat_tool_call_and_usage():
    c, sess = _client(REPLY_TOOL)
    r = c.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "how many lines?"}], "tools": TOOLS_O}).json()
    msg = r["choices"][0]["message"]
    assert msg["content"] == "I'll check."
    assert msg["tool_calls"][0]["function"] == {"name": "Bash", "arguments": json.dumps({"command": "wc -l docs/traps.md"})}
    assert r["choices"][0]["finish_reason"] == "tool_calls"
    assert r["usage"]["prompt_tokens"] == len(sess.prompts[0][0]) and r["usage"]["completion_tokens"] > 0
    assert sess.prompts[0][2] == 0.7                       # the server default when the request does not say
    # tools reached the template
    assert "<tools>" in sess.tok.decode(sess.prompts[0][0])


def test_openai_chat_stream_shapes():
    c, _ = _client("Hello there.")
    ev = _sse(c.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True,
                                                    "stream_options": {"include_usage": True}, "temperature": 0}).text)
    assert ev[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in ev) == "Hello there."
    assert ev[-1]["choices"][0]["finish_reason"] == "stop" and ev[-1]["usage"]["completion_tokens"] > 0
    assert all(e["object"] == "chat.completion.chunk" for e in ev)


def test_completions_stop_sequence():
    c, sess = _client("Paris.\nThe capital of Germany is Berlin.")
    r = c.post("/v1/completions", json={"model": "m", "prompt": "The capital of France is", "max_tokens": 50, "stop": ["\n"]}).json()
    assert r["choices"][0]["text"] == "Paris."
    assert r["choices"][0]["finish_reason"] == "stop"
    assert sess.prompts[0][0] == sess.tok.encode("The capital of France is", add_special_tokens=False)


def test_anthropic_messages_tool_use_and_round_trip():
    c, sess = _client(REPLY_TOOL)
    body = {"model": "claude-x", "max_tokens": 100, "system": [{"type": "text", "text": "sys"}], "tools": TOOLS_A,
            "messages": [{"role": "user", "content": "how many lines?"}]}
    r = c.post("/v1/messages", json=body).json()
    assert [b["type"] for b in r["content"]] == ["text", "tool_use"]
    assert r["content"][1]["name"] == "Bash" and r["content"][1]["input"] == {"command": "wc -l docs/traps.md"}
    assert r["content"][1]["id"].startswith("toolu_")
    assert r["stop_reason"] == "tool_use" and r["usage"]["cache_read_input_tokens"] == 7
    # the round trip renders tool_use / tool_result through the template
    body["messages"] += [{"role": "assistant", "content": r["content"]},
                         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": r["content"][1]["id"], "content": "145 docs/traps.md"}]}]
    c.post("/v1/messages", json=body)
    prompt = sess.tok.decode(sess.prompts[1][0])
    assert "<tool_call>\n<function=Bash>\n<parameter=command>\nwc -l docs/traps.md\n</parameter>\n</function>\n</tool_call>" in prompt
    assert "<tool_response>\n145 docs/traps.md\n</tool_response>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # count_tokens agrees with what the job saw
    assert c.post("/v1/messages/count_tokens", json=body).json()["input_tokens"] == len(sess.prompts[1][0])


def test_anthropic_stream_event_sequence():
    c, _ = _client(REPLY_TOOL)
    text = c.post("/v1/messages", json={"model": "claude-x", "max_tokens": 100, "tools": TOOLS_A, "stream": True,
                                        "messages": [{"role": "user", "content": "how many lines?"}]}).text
    ev = _sse(text)
    types_ = [e["type"] for e in ev]
    assert types_[0] == "message_start" and types_[-2:] == ["message_delta", "message_stop"]
    starts = [e for e in ev if e["type"] == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["text", "tool_use"]
    assert [s["index"] for s in starts] == [0, 1]
    deltas = [e for e in ev if e["type"] == "content_block_delta"]
    assert "".join(d["delta"]["text"] for d in deltas if d["delta"]["type"] == "text_delta") == "I'll check."
    js = "".join(d["delta"]["partial_json"] for d in deltas if d["delta"]["type"] == "input_json_delta")
    assert json.loads(js) == {"command": "wc -l docs/traps.md"}
    assert types_.count("content_block_stop") == 2
    md = next(e for e in ev if e["type"] == "message_delta")
    assert md["delta"]["stop_reason"] == "tool_use" and md["usage"]["output_tokens"] > 0
    assert "event: message_start" in text          # the SSE event: lines Anthropic clients dispatch on


def test_anthropic_rejects_without_max_tokens_and_lists_models():
    c, _ = _client("x")
    r = c.post("/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400 and r.json()["type"] == "error"
    m = c.get("/v1/models").json()
    assert m["data"][0]["id"] == "token-rush" and m["data"][0]["object"] == "model"
    assert c.get("/health").json()["status"] == "ok"


def test_chat_page_and_saved_transcripts(tmp_path):
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app

    class Unused:
        max_len = 32768

    path = tmp_path / "chats.json"
    args = types.SimpleNamespace(api_key=None, think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[], chats=str(path))
    c = TestClient(build_app(Unused(), None, None, args))
    page = c.get("/")
    assert page.status_code == 200 and "新对话" in page.text and "写代码" in page.text and 'id="ide"' in page.text and 'id="monaco"' in page.text and "打开目录" in page.text
    assert c.get("/chats").json() == []
    saved = c.put("/chats/abc", json={"title": "草稿", "messages": [{"role": "user", "content": "你好"}]}).json()
    assert saved["title"] == "草稿" and saved["messages"] == [{"role": "user", "content": "你好"}]
    assert c.get("/chats/abc").json()["messages"][0]["content"] == "你好"
    assert c.get("/chats").json()[0]["id"] == "abc"
    # a title is taken from the first user line when the client does not send one
    auto = c.put("/chats/def", json={"messages": [{"role": "user", "content": "第二段对话的开头"}]}).json()
    assert auto["title"] == "第二段对话的开头"
    tools = c.put("/chats/src", json={"messages": [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\": \"q\"}"}}]},
        {"role": "tool", "name": "web_search", "tool_call_id": "call_1", "content": "https://example.com/a"},
    ]}).json()
    assert tools["messages"][0]["tool_calls"][0]["function"]["name"] == "web_search"
    assert tools["messages"][1]["tool_call_id"] == "call_1"
    assert c.put("/chats/abc", json={"messages": [{"role": "nope", "content": "x"}]}).status_code == 400
    assert c.put("/chats/bad!id", json={"messages": []}).status_code == 400
    assert c.get("/chats/abc").json()["messages"][0]["role"] == "user"   # the rejected put did not replace it
    assert c.delete("/chats/abc").status_code == 200
    assert c.get("/chats/abc").status_code == 404
    assert c.delete("/chats/abc").status_code == 404
    assert json.loads(path.read_text())["chats"][0]["id"] == "def"
    folded = c.put("/chats/fold1", json={"messages": [{"role": "user", "content": "原文还在"}],
                                         "fold": {"summary": "一段摘要", "before": 1}}).json()
    assert folded["fold"] == {"summary": "一段摘要", "before": 1}
    assert folded["messages"][0]["content"] == "原文还在"


def test_server_tools_search_then_cite(monkeypatch):
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    tok = _tok()
    seen = []

    def fake_search(query):
        seen.append(query)
        return "1. Example\nhttps://example.com/a\none", [{"title": "Example", "url": "https://example.com/a"}]

    monkeypatch.setattr("tokenrush.serve.web_search", fake_search)
    tool = ("<tool_call>\n<function=web_search>\n<parameter=query>\nnvidia price\n</parameter>\n"
            "</function>\n</tool_call>")
    sess = StubSession(tok, tool)
    sess.replies = [tool, "See https://example.com/a for the price."]
    orig = sess.generate

    def generate(ids, max_new, mode, temperature, top_p, top_k, seed):
        sess.reply = sess.replies.pop(0)
        yield from orig(ids, max_new, mode, temperature, top_p, top_k, seed)

    sess.generate = generate
    args = types.SimpleNamespace(api_key=None, think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[], chats=None)
    c = TestClient(build_app(sess, tok, types.SimpleNamespace(eos_ids=(tok.eos_token_id,)), args))
    r = c.post("/v1/chat/completions", json={"model": "m", "server_tools": True,
                                            "messages": [{"role": "user", "content": "price?"}]}).json()
    assert seen == ["nvidia price"]
    assert "https://example.com/a" in r["choices"][0]["message"]["content"]
    second = tok.decode(sess.prompts[1][0])
    assert "https://example.com/a" in second and "<tool_response>" in second


def test_without_server_tools_does_not_search(monkeypatch):
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    tok = _tok()

    def boom(query):
        raise AssertionError("search must not run")

    monkeypatch.setattr("tokenrush.serve.web_search", boom)
    tool = ("<tool_call>\n<function=web_search>\n<parameter=query>\nnvidia price\n</parameter>\n"
            "</function>\n</tool_call>")
    args = types.SimpleNamespace(api_key=None, think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[], chats=None)
    c = TestClient(build_app(StubSession(tok, tool), tok, types.SimpleNamespace(eos_ids=(tok.eos_token_id,)), args))
    r = c.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "price?"}]}).json()
    assert r["choices"][0]["finish_reason"] == "tool_calls"
    assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "web_search"


def _scripted(tok, replies):
    sess = StubSession(tok, replies[0])
    sess.replies = list(replies)

    def generate(ids, max_new, mode, temperature, top_p, top_k, seed):
        sess.reply = sess.replies.pop(0)
        yield from StubSession.generate(sess, ids, max_new, mode, temperature, top_p, top_k, seed)

    sess.generate = generate
    return sess


_RECALL = ("<tool_call>\n<function=recall>\n<parameter=query>\n学习率\n</parameter>\n"
           "</function>\n</tool_call>")


def test_recall_uses_this_chat_when_the_old_one_differs(tmp_path):
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    tok = _tok()
    path = tmp_path / "chats.json"
    args = types.SimpleNamespace(api_key=None, think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[], chats=str(path))
    sess = _scripted(tok, [_RECALL, "用 3e-4。旧记录是 1e-4。"])
    c = TestClient(build_app(sess, tok, types.SimpleNamespace(eos_ids=(tok.eos_token_id,)), args))
    c.put("/chats/old", json={"title": "旧的学习率笔记", "created": 1760000000,
                              "messages": [{"role": "user", "content": "把学习率定成 1e-4。"}]})
    r = c.post("/v1/chat/completions", json={
        "model": "m", "server_tools": True, "chat_id": "new",
        "messages": [
            {"role": "user", "content": "学习率用 3e-4"},
            {"role": "assistant", "content": "好"},
            {"role": "user", "content": "上次说的学习率是多少"},
        ],
    }).json()
    first = tok.decode(sess.prompts[0][0])
    assert "1e-4" not in first and "旧的学习率笔记" not in first
    second = tok.decode(sess.prompts[1][0])
    assert "1e-4" in second and "3e-4" in second and "以本段为准" in second
    assert r["choices"][0]["message"]["content"] == "用 3e-4。旧记录是 1e-4。"


def test_recall_does_not_open_the_file_unless_the_user_asked(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    tok = _tok()

    def boom(*args, **kwargs):
        raise AssertionError("chat file was searched")

    monkeypatch.setattr("tokenrush.recall.search_other", boom)
    args = types.SimpleNamespace(api_key=None, think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[], chats=str(tmp_path / "chats.json"))
    sess = _scripted(tok, [_RECALL, "继续写。"])
    c = TestClient(build_app(sess, tok, types.SimpleNamespace(eos_ids=(tok.eos_token_id,)), args))
    r = c.post("/v1/chat/completions", json={"model": "m", "server_tools": True,
                                            "messages": [{"role": "user", "content": "继续写代码"}]}).json()
    assert r["choices"][0]["message"]["content"] == "继续写。"
    sess2 = StubSession(tok, _RECALL)
    c2 = TestClient(build_app(sess2, tok, types.SimpleNamespace(eos_ids=(tok.eos_token_id,)), args))
    bare = c2.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "继续写代码"}]}).json()
    assert bare["choices"][0]["finish_reason"] == "tool_calls"
    assert bare["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "recall"


def test_guide_is_unchanged_on_the_next_turn():
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    tok = _tok()
    args = types.SimpleNamespace(api_key=None, think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[], chats=None)
    sess = _scripted(tok, ["你好呀", "再看一眼。"])
    c = TestClient(build_app(sess, tok, types.SimpleNamespace(eos_ids=(tok.eos_token_id,)), args))
    c.post("/v1/chat/completions", json={"model": "m", "server_tools": True,
                                        "messages": [{"role": "user", "content": "你好"}]})
    c.post("/v1/chat/completions", json={"model": "m", "server_tools": True, "messages": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
        {"role": "user", "content": "再来一句"},
    ]})
    a, b = sess.prompts[0][0], sess.prompts[1][0]
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    shared = tok.decode(a[:n])
    assert n > 30 and "recall" in shared and "web_search" in shared


def test_fold_shortens_the_prompt_and_leaves_the_file(tmp_path):
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    tok = _tok()
    filler = "填充。" * 300
    original = "学习率用 3e-4。" + filler
    sess = StubSession(tok, "好的，按摘要继续。")
    sess.max_len = 1400

    def generate(ids, max_new, mode, temperature, top_p, top_k, seed):
        sess.prompts.append((list(ids), max_new, temperature, top_p))
        text = tok.decode(ids)
        reply = "摘要：学习率用 3e-4。" if "收成一段摘要" in text else "好的，按摘要继续。"
        toks = tok.encode(reply, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
        for i in range(0, len(toks), 3):
            yield toks[i:i + 3]

    sess.generate = generate
    path = tmp_path / "chats.json"
    args = types.SimpleNamespace(api_key=None, think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[], chats=str(path))
    c = TestClient(build_app(sess, tok, types.SimpleNamespace(eos_ids=(tok.eos_token_id,)), args))
    messages = [
        {"role": "user", "content": original},
        {"role": "assistant", "content": "记下了"},
        {"role": "user", "content": "请继续"},
    ]
    c.put("/chats/long", json={"messages": messages})
    r = c.post("/v1/chat/completions", json={"model": "m", "server_tools": True, "chat_id": "long",
                                            "messages": messages}).json()
    assert r["fold"]["before"] == 1 and "3e-4" in r["fold"]["summary"]
    answer = tok.decode(sess.prompts[-1][0])
    assert "已收成摘要" in answer and "请继续" in answer
    assert len(sess.prompts[-1][0]) < len(sess.prompts[0][0])
    assert c.get("/chats/long").json()["messages"][0]["content"] == original
    sess.prompts.clear()
    again = messages + [{"role": "assistant", "content": "好的，按摘要继续。"}, {"role": "user", "content": "还有呢"}]
    c.post("/v1/chat/completions", json={"model": "m", "server_tools": True, "chat_id": "long",
                                        "messages": again, "fold": r["fold"]})
    assert sess.prompts and "收成一段摘要" not in tok.decode(sess.prompts[0][0])


def test_search_cap_still_answers_the_question(monkeypatch):
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    tok = _tok()
    tool = ("<tool_call>\n<function=web_search>\n<parameter=query>\naria fisheye62\n</parameter>\n"
            "</function>\n</tool_call>")
    sess = StubSession(tok, tool)

    def generate(ids, max_new, mode, temperature, top_p, top_k, seed):
        sess.prompts.append((list(ids),))
        text = tok.decode(ids)
        reply = "Aria 的文档没有要求用 fisheye62。" if "不要再调用工具" in text else tool
        toks = tok.encode(reply, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
        for i in range(0, len(toks), 3):
            yield toks[i:i + 3]

    sess.generate = generate
    monkeypatch.setattr("tokenrush.serve.web_search",
                        lambda q: ("1. Aria\nhttps://example.com/aria\ndocs", [{"title": "Aria", "url": "https://example.com/aria"}]))
    args = types.SimpleNamespace(api_key=None, think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[], chats=None)
    c = TestClient(build_app(sess, tok, types.SimpleNamespace(eos_ids=(tok.eos_token_id,)), args))
    r = c.post("/v1/chat/completions", json={"model": "m", "server_tools": True,
                                            "messages": [{"role": "user", "content": "那不应该用fisheye62吗"}]}).json()
    assert r["choices"][0]["message"]["content"] == "Aria 的文档没有要求用 fisheye62。"
    assert "已达到本轮搜索次数上限" not in r["choices"][0]["message"]["content"]


def test_second_browser_takes_the_seat_with_the_password(tmp_path):
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    args = types.SimpleNamespace(api_key=None, think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[], chats=str(tmp_path / "chats.json"))
    app = build_app(types.SimpleNamespace(max_len=32768), None, None, args)
    first, second = TestClient(app), TestClient(app)
    sat = first.get("/seat")
    assert sat.status_code == 200 and sat.json()["ok"] is True
    ip = sat.json()["ip"]
    blocked = second.get("/seat")
    assert blocked.status_code == 401 and blocked.json()["holder"] == ip
    assert second.post("/login", json={"password": "nope"}).status_code == 401
    assert second.get("/chats").status_code == 401
    taken = second.post("/login", json={"password": "bestcalib"})
    assert taken.status_code == 200 and taken.json()["ip"] == ip
    assert first.get("/seat").status_code == 401
    assert second.get("/chats").status_code == 200
    assert first.get("/chats").status_code == 401


def test_loopback_login_shows_the_lan_address(tmp_path):
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    args = types.SimpleNamespace(api_key=None, think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[], chats=str(tmp_path / "chats.json"))
    app = build_app(types.SimpleNamespace(max_len=32768), None, None, args)
    local = TestClient(app, client=("127.0.0.1", 50000))
    ip = local.get("/seat").json()["ip"]
    assert ip and not ip.startswith("127.")


def test_api_key_required_when_set():
    from fastapi.testclient import TestClient
    from tokenrush.serve import build_app
    tok = _tok()
    args = types.SimpleNamespace(api_key="secret", think="auto", temperature=0.7, top_p=0.9, max_new=512, draft="auto",
                                 served_name="token-rush", alias=[])
    c = TestClient(build_app(StubSession(tok, "ok"), tok, types.SimpleNamespace(eos_ids=(tok.eos_token_id,)), args))
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"x-api-key": "secret"}).status_code == 200
    assert c.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
