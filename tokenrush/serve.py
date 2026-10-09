"""One-stream HTTP server: OpenAI Chat Completions / Completions and the
Anthropic Messages API over the resident engine.

    python -m tokenrush.serve [--port 8000] [--max-len 262144] [--api-key ...]

One request runs at a time (the engine is bs=1); others queue. Conversations
are re-sent whole by every client, so the Session keeps the context between
requests and prefills only the new turn (tokenrush/session.py). Tool calling
goes through the model's own format (tokenrush/chat.py); nothing here is
constrained decoding.

Claude Code:   ANTHROPIC_BASE_URL=http://host:8000 ANTHROPIC_AUTH_TOKEN=x claude
               (set CLAUDE_CODE_MAX_CONTEXT_TOKENS to --max-len; docs/serving.md)
OpenAI clients: base_url http://host:8000/v1, any model name.
"""
import argparse
import asyncio
import base64
import json
import os
import queue
import secrets
import socket
import threading
import time
import uuid

import torch

from .chat import (OutputParser, StopFilter, from_anthropic, from_openai, render, tools_from_anthropic,
                   tools_from_openai)
from .chats import ChatError, ChatStore, default_chats_path
from .recall import asks_about_past, recall, shorten_tool
from .fsview import FsError, inside, list_dir, read_text, write_text
from .search import backend_name, web_fetch, web_search

_SEAT_PASSWORD = "bestcalib"


def lan_ip() -> str:
    """This machine's intranet address. A browser on 127.0.0.1 is still this host."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 1))
        ip = sock.getsockname()[0]
    except OSError:
        ip = ""
    finally:
        sock.close()
    if not ip or ip.startswith("127."):
        return ""
    return ip
from .session import Session

_MAX_TOOL_CALLS = 8
_MAX_FETCHES = 4
_SEARCH_GUIDE = (
    "需要时效、或你没有把握的事实时，调用 web_search。需要某条结果的正文时，调用 web_fetch，参数是搜索结果里的 https URL。"
    "只引用工具结果里出现过的 URL。搜索没有结果、超时或抓取失败时，直接说没查到，不要编造数字或链接。"
    "只在用户询问以前某段对话的内容时调用 recall。摘录与本段已经说过的话冲突时，按本段回答，并写出旧记录里的不同说法，不要用旧记录改正本段。"
)
_SERVER_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the public web. Use for news, prices, and facts you are not sure about.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "web_fetch",
        "description": "Fetch the readable text of one https page returned by web_search.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "recall",
        "description": "Look up another saved chat. Use only when the user asks what was said in an earlier chat. If the excerpt conflicts with this chat, follow this chat and mention the difference.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]
_CODE_TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List one directory inside the opened project. path is absolute.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file inside the opened project. path is absolute.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write a text file inside the opened project, replacing its contents. path is absolute.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
]
_FOLD_NOTE = "更早的内容已收成摘要（模型看不到被收起的原文）：\n"
_SUMMARY_SYSTEM = "把更早的对话收成一段摘要，保留用户定下的数字和决定。只输出摘要，不要调用工具。"

# ------------------------------------------------------------ the worker


class Job:
    def __init__(self, ids, mode, max_new, temperature, top_p, top_k, seed, stops, schemas, thinking, loop):
        self.ids, self.mode, self.max_new = ids, mode, max_new
        self.temperature, self.top_p, self.top_k, self.seed = temperature, top_p, top_k, seed
        self.stops, self.schemas, self.thinking = stops, schemas, thinking
        self.loop = loop
        self.q = asyncio.Queue()
        self.cancel = threading.Event()

    def push(self, ev):
        self.loop.call_soon_threadsafe(self.q.put_nowait, ev)


class Worker:
    """The one thread that touches the GPU. Events pushed per job:
    ("thinking", s) / ("text", s) / ("tool_call", ToolCall) / ("done", info) / ("error", msg)."""

    def __init__(self, session: Session, tok):
        self.session, self.tok = session, tok
        self.jobs = queue.Queue()
        self.last_stats = {}
        self.busy = False
        threading.Thread(target=self._loop, daemon=True, name="tokenrush-worker").start()

    def submit(self, job: Job):
        self.jobs.put(job)

    def _loop(self):
        while True:
            job = self.jobs.get()
            if job.cancel.is_set():
                continue
            self.busy = True
            try:
                self._run(job)
            except Exception as e:      # noqa: BLE001 — report to the client, keep serving
                import traceback
                traceback.print_exc()
                job.push(("error", f"{type(e).__name__}: {e}"))
                self.session.forget()
            finally:
                self.busy = False

    def _run(self, job: Job):
        sess, tok = self.session, self.tok
        parser = OutputParser(job.schemas, thinking=job.thinking)
        stopf = StopFilter(job.stops)
        out, printed = [], 0
        stop_reason, stop_seq = None, None
        gen = sess.generate(job.ids, max_new=job.max_new, mode=job.mode, temperature=job.temperature,
                            top_p=job.top_p, top_k=job.top_k, seed=job.seed)
        try:
            for toks in gen:
                if job.cancel.is_set():
                    stop_reason = "cancelled"
                    break
                out.extend(toks)
                if len(out) > job.max_new:
                    out = out[:job.max_new]
                ended = bool(out) and out[-1] in sess.stop_ids
                full = len(out) >= job.max_new
                text = tok.decode(out[printed:], skip_special_tokens=True)
                if text.endswith("�") and not (ended or full):
                    continue                                   # half a multibyte character: wait for the rest
                printed = len(out)
                piece, hit = stopf.feed(text)
                for ev in parser.feed(piece):
                    job.push(ev)
                if hit:
                    stop_reason, stop_seq = "stop_sequence", stopf.hit
                    break
                if ended:
                    stop_reason = "end_turn"
                    break
                if full:
                    stop_reason = "max_tokens"
                    break
            else:
                stop_reason = "max_tokens"                     # the context is full
        finally:
            gen.close()
        for ev in parser.feed(stopf.finish()) + parser.finish():
            job.push(ev)
        if stop_reason == "end_turn" and parser.calls:
            stop_reason = "tool_use"
        n_out = len(out) - (1 if out and out[-1] in sess.stop_ids else 0)
        st = sess.stats.as_dict()
        self.last_stats = st
        job.push(("done", {"stop_reason": stop_reason, "stop_sequence": stop_seq,
                           "input_tokens": len(job.ids), "output_tokens": n_out, "stats": st}))


# ------------------------------------------------------------ the app


def build_app(session: Session, tok, cfg, args):
    from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

    app = FastAPI(title="token-rush")
    worker = Worker(session, tok)
    store = ChatStore(getattr(args, "chats", None) or default_chats_path())
    seat = {"token": None, "ip": ""}
    seat_lock = threading.Lock()
    web = os.path.join(os.path.dirname(__file__), "web", "index.html")
    served = args.served_name
    max_len = session.max_len
    reserve = 8 + 2                                            # a spec step may process K+1 tokens past the prompt

    def auth(req: Request):
        if not args.api_key:
            return
        key = req.headers.get("x-api-key") or req.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if key != args.api_key:
            raise HTTPException(401, "invalid api key")

    def client_ip(req: Request) -> str:
        host = req.client.host if req.client else ""
        if host.startswith("::ffff:"):
            host = host.removeprefix("::ffff:")
        if host in ("", "127.0.0.1", "::1", "localhost") or host.startswith("127."):
            return lan_ip() or host
        return host

    def seat_view(req: Request):
        token = req.cookies.get("tr_seat")
        with seat_lock:
            if not seat["token"]:
                return "empty", ""
            if token == seat["token"]:
                return "in", seat["ip"]
            return "out", seat["ip"]

    def take_seat(response, ip: str) -> str:
        token = secrets.token_urlsafe(24)
        with seat_lock:
            seat["token"] = token
            seat["ip"] = ip
        response.set_cookie("tr_seat", token, httponly=True, samesite="lax", max_age=14 * 24 * 3600, path="/")
        return token

    def hold_page(req: Request, response) -> str:
        """One browser holds the page. An empty seat is taken; a second browser is refused."""
        state, ip = seat_view(req)
        if state == "in":
            return req.cookies.get("tr_seat")
        if state == "empty":
            return take_seat(response, client_ip(req))
        raise HTTPException(401, {"holder": ip})

    def want_thinking(body: dict, proto: str) -> bool:
        if args.think == "on":
            return True
        if args.think == "off":
            return False
        if proto == "anthropic":
            t = body.get("thinking")
            return isinstance(t, dict) and t.get("type") == "enabled"
        kw = body.get("chat_template_kwargs") or {}
        if "enable_thinking" in kw:
            return bool(kw["enable_thinking"])
        return body.get("reasoning_effort") is not None or isinstance(body.get("reasoning"), dict)

    def make_job(ids, body: dict, max_new, stops, schemas, thinking, text: str):
        if len(ids) + reserve >= max_len:
            raise HTTPException(400, f"prompt of {len(ids)} tokens does not fit the {max_len}-token context")
        max_new = max(1, min(int(max_new), max_len - len(ids) - reserve))
        temperature = body.get("temperature", args.temperature)
        top_p = body.get("top_p", args.top_p)
        top_k = body.get("top_k", 64)
        top_k = 64 if top_k is None or top_k <= 0 else min(int(top_k), 64)
        temperature = float(temperature) if temperature is not None else args.temperature
        top_p = float(top_p) if top_p is not None else args.top_p
        mode = session.pick_mode(text, args.draft)
        return Job(ids, mode, max_new, temperature, top_p, top_k, body.get("seed"), stops, schemas, thinking,
                   asyncio.get_running_loop())

    async def events(job: Job, req: Request):
        """Async iterator over the job's events; cancels the job when the client leaves."""
        worker.submit(job)
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(job.q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if await req.is_disconnected():
                        job.cancel.set()
                        return
                    continue
                yield ev
                if ev[0] in ("done", "error"):
                    return
        finally:
            job.cancel.set()

    def sse(obj, event=None):
        head = f"event: {event}\n" if event else ""
        return f"{head}data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def encode(text: str):
        return tok.encode(text, add_special_tokens=False)

    def schemas_of(tools):
        return {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}

    # ---------------------------------------------------------------- models

    def model_entries():
        names = [served] + list(args.alias)
        return [{"id": n, "object": "model", "type": "model", "created": int(app.state.t0), "owned_by": "token-rush",
                 "display_name": n, "max_model_len": max_len} for n in names]

    @app.get("/v1/models")
    async def models(req: Request):
        auth(req)
        return {"object": "list", "data": model_entries(), "has_more": False}

    @app.get("/v1/models/{name:path}")
    async def model_one(name: str, req: Request):
        auth(req)
        return {**model_entries()[0], "id": name, "display_name": name}

    @app.get("/")
    def home():
        return FileResponse(web)

    @app.get("/seat")
    def session_get(req: Request, response: Response):
        state, ip = seat_view(req)
        if state == "in":
            return {"ok": True, "ip": ip}
        if state == "empty":
            take_seat(response, client_ip(req))
            return {"ok": True, "ip": client_ip(req)}
        return JSONResponse({"ok": False, "holder": ip}, status_code=401)

    @app.post("/login")
    async def session_login(req: Request, response: Response):
        body = await req.json()
        state, ip = seat_view(req)
        if state == "in":
            return {"ok": True, "ip": ip}
        if state == "empty" or str(body.get("password") or "") == _SEAT_PASSWORD:
            take_seat(response, client_ip(req))
            return {"ok": True, "ip": client_ip(req)}
        return JSONResponse({"ok": False, "holder": ip}, status_code=401)

    def chat_call(fn, *a):
        try:
            return fn(*a)
        except ChatError as e:
            raise HTTPException(400, str(e))

    @app.get("/chats")
    def chats_list(req: Request, response: Response):
        auth(req)
        hold_page(req, response)
        return store.list()

    @app.get("/chats/{chat_id}")
    def chats_get(chat_id: str, req: Request, response: Response):
        auth(req)
        hold_page(req, response)
        found = chat_call(store.get, chat_id)
        if found is None:
            raise HTTPException(404, "no such chat")
        return found

    @app.put("/chats/{chat_id}")
    async def chats_put(chat_id: str, req: Request, response: Response):
        auth(req)
        hold_page(req, response)
        return chat_call(store.put, chat_id, await req.json())

    @app.delete("/chats/{chat_id}")
    def chats_delete(chat_id: str, req: Request, response: Response):
        auth(req)
        hold_page(req, response)
        if not chat_call(store.delete, chat_id):
            raise HTTPException(404, "no such chat")
        return {"ok": True}

    def fs_call(fn, *a):
        try:
            return fn(*a)
        except FsError as e:
            raise HTTPException(400, str(e))

    @app.get("/fs/list")
    def fs_list(req: Request, response: Response, path: str = ""):
        auth(req)
        hold_page(req, response)
        return fs_call(list_dir, path)

    @app.get("/fs/file")
    def fs_read(req: Request, response: Response, path: str):
        auth(req)
        hold_page(req, response)
        return {"path": path, "content": fs_call(read_text, path)}

    @app.put("/fs/file")
    async def fs_write(req: Request, response: Response):
        auth(req)
        hold_page(req, response)
        body = await req.json()
        root = str(body.get("root") or "")
        path = fs_call(write_text, str(body.get("path") or ""), body.get("content"), root)
        return {"ok": True, "path": path}

    @app.get("/health")
    async def health():
        return {"status": "ok", "busy": worker.busy, "queued": worker.jobs.qsize(), "context": max_len,
                "temperature": args.temperature, "top_p": args.top_p}

    @app.get("/stats")
    async def stats():
        return worker.last_stats

    # ---------------------------------------------------------------- OpenAI

    def with_search_guide(msgs):
        msgs = [dict(m) for m in msgs]
        if msgs and msgs[0].get("role") == "system":
            if _SEARCH_GUIDE not in (msgs[0].get("content") or ""):
                msgs[0]["content"] = ((msgs[0].get("content") or "") + "\n\n" + _SEARCH_GUIDE).strip()
        else:
            msgs.insert(0, {"role": "system", "content": _SEARCH_GUIDE})
        return msgs

    def _split_guide(msgs):
        if msgs and msgs[0].get("role") == "system":
            return msgs[0], list(msgs[1:])
        return None, list(msgs)

    def _apply_fold(system, rest, summary, before):
        head = [system] if system else []
        note = {"role": "user", "content": _FOLD_NOTE + summary}
        return head + [note] + [shorten_tool(dict(m)) for m in rest[before:]]

    def _prompt_len(messages, body, thinking):
        text = render(tok, messages, list(_SERVER_TOOLS), think=thinking, reasoning_effort=body.get("reasoning_effort"))
        return len(encode(text))

    async def _summarize(older, body, req: Request) -> str:
        plain = "\n".join(f"{shorten_tool(m).get('role')}: {shorten_tool(m).get('content') or ''}" for m in older)
        ids, text = [], ""
        for _ in range(6):
            text = render(tok, [{"role": "system", "content": _SUMMARY_SYSTEM}, {"role": "user", "content": plain}], think=False)
            ids = encode(text)
            if len(ids) + 16 < max_len or len(plain) < 200:
                break
            plain = plain[:len(plain) // 2]
        try:
            job = make_job(ids, body, 256, [], {}, False, text)
        except HTTPException:
            return "（更早的对话已省略）"
        parts = []
        async for kind, v in events(job, req):
            if kind == "text":
                parts.append(v)
            elif kind == "error":
                return "（更早的对话已省略）"
        summary = "".join(parts).strip()
        return summary or "（更早的对话已省略）"

    async def prepare_context(msgs, body, thinking, req: Request):
        """Drop older turns from the prompt once the chat passes half the window.

        The chat file keeps every sentence. A fold already in the request is reused
        so the prefix stays put; it is rewritten only when that shorter prompt
        itself passes the halfway mark."""
        system, rest = _split_guide(msgs)
        raw = body.get("fold") if isinstance(body.get("fold"), dict) else None
        reusable = (isinstance(raw, dict) and isinstance(raw.get("summary"), str) and isinstance(raw.get("before"), int)
                    and not isinstance(raw.get("before"), bool) and 0 < raw["before"] < len(rest))
        if reusable:
            folded = _apply_fold(system, rest, raw["summary"], raw["before"])
            if _prompt_len(folded, body, thinking) <= max_len // 2:
                return folded, None, rest
        elif _prompt_len(msgs, body, thinking) <= max_len // 2:
            return msgs, None, rest
        keep = 4 if len(rest) > 4 else (2 if len(rest) > 2 else len(rest))
        if keep >= len(rest):
            return msgs, None, rest
        before = len(rest) - keep
        summary = await _summarize(rest[:before], body, req)
        return _apply_fold(system, rest, summary, before), {"summary": summary, "before": before}, rest

    async def tool_rounds(body, msgs, req: Request):
        """Yields the same worker events, plus ("search", info), ("tool_result", info)
        and ("fold", info). At most three tool executions, then one answer with the tools removed."""
        thinking = want_thinking(body, "openai")
        chat_id = body.get("chat_id") if isinstance(body.get("chat_id"), str) else ""
        page_token = getattr(req.state, "seat_token", None)
        msgs, fold, client_msgs = await prepare_context(msgs, body, thinking, req)
        if fold:
            yield ("fold", fold)
        tools = list(_SERVER_TOOLS)
        fetches = 0
        used = 0
        nudged = False
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        max_new = body.get("max_completion_tokens") or body.get("max_tokens") or args.max_new
        while True:
            if page_token is not None:
                with seat_lock:
                    if seat["token"] != page_token:
                        yield ("error", "你已被挤下线")
                        return
            if await req.is_disconnected():
                return
            text = render(tok, msgs, tools, think=thinking, reasoning_effort=body.get("reasoning_effort"))
            try:
                job = make_job(encode(text), body, max_new, stops, schemas_of(tools), thinking, text)
            except HTTPException as exc:
                yield ("error", exc.detail)
                return
            texts, thinks, calls, done = [], [], [], None
            async for kind, v in events(job, req):
                if kind == "text":
                    texts.append(v)
                    yield ("text", v)
                elif kind == "thinking":
                    thinks.append(v)
                    yield ("thinking", v)
                elif kind == "tool_call":
                    calls.append(v)
                elif kind == "error":
                    yield ("error", v)
                    return
                elif kind == "done":
                    done = v
            if done is None:
                return
            if not calls or not tools:
                # Tools are gone and the model still only emitted a call. One more
                # turn, with the results already in the prompt, has to answer the
                # question instead of stopping on the cap sentence.
                if calls and not ("".join(texts).strip()) and not nudged:
                    nudged = True
                    tools = []
                    msgs.append({"role": "assistant", "content": "".join(texts),
                                 "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in calls]})
                    msgs.append({"role": "user", "content": "不要再调用工具。根据已经拿到的结果直接回答上面的问题。不够就说明没查到，不要罗列链接。"})
                    continue
                yield ("done", done)
                return
            executed = []
            for call in calls:
                if used >= _MAX_TOOL_CALLS:
                    break
                if call.name not in ("web_search", "web_fetch", "recall"):
                    yield ("text", f"不会执行这个工具：{call.name}")
                    yield ("done", done)
                    return
                query = str((call.arguments or {}).get("query") or (call.arguments or {}).get("url") or "")
                if call.name == "recall":
                    if asks_about_past(client_msgs):
                        yield ("search", {"tool": "recall", "query": query})
                    result, sources = recall(store, query, chat_id, client_msgs), []
                elif call.name == "web_fetch" and fetches >= _MAX_FETCHES:
                    yield ("search", {"tool": call.name, "query": query})
                    result, sources = f"本轮最多打开 {_MAX_FETCHES} 个页面。", []
                elif call.name == "web_search":
                    yield ("search", {"tool": call.name, "query": query})
                    result, sources = await asyncio.to_thread(web_search, query)
                else:
                    yield ("search", {"tool": call.name, "query": query})
                    result, sources = await asyncio.to_thread(web_fetch, query)
                    if not result.startswith("拒绝"):
                        fetches += 1
                used += 1
                executed.append((call, result, sources))
                if await req.is_disconnected():
                    return
            content = "".join(texts)
            msgs.append({"role": "assistant", "content": content, "reasoning_content": "".join(thinks),
                         "tool_calls": [{"name": c.name, "arguments": c.arguments} for c, _, _ in executed]})
            for call, result, sources in executed:
                msgs.append({"role": "tool", "content": result})
                yield ("tool_result", {"name": call.name, "tool_call_id": call.id,
                                       "arguments": json.dumps(call.arguments, ensure_ascii=False),
                                       "content": result, "sources": sources})
            if used >= _MAX_TOOL_CALLS:
                tools = []

    async def code_rounds(body, msgs, req: Request, root: str):
        """File tools for the code page. The opened directory is the only place they can write."""
        thinking = want_thinking(body, "openai")
        page_token = getattr(req.state, "seat_token", None)
        tools = list(_CODE_TOOLS)
        used = 0
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        max_new = body.get("max_completion_tokens") or body.get("max_tokens") or args.max_new
        guide = ("你在编辑服务器目录 " + root + "。用 list_dir、read_file、write_file 查看和修改其中的文件，"
                 "path 用这个目录下的绝对路径。改完用一两句话说明改了什么。")
        if msgs and msgs[0].get("role") == "system":
            msgs = [dict(msgs[0], content=((msgs[0].get("content") or "") + "\n\n" + guide).strip())] + [dict(m) for m in msgs[1:]]
        else:
            msgs = [{"role": "system", "content": guide}] + [dict(m) for m in msgs]
        while True:
            if page_token is not None:
                with seat_lock:
                    if seat["token"] != page_token:
                        yield ("error", "你已被挤下线")
                        return
            if await req.is_disconnected():
                return
            text = render(tok, msgs, tools, think=thinking, reasoning_effort=body.get("reasoning_effort"))
            try:
                job = make_job(encode(text), body, max_new, stops, schemas_of(tools), thinking, text)
            except HTTPException as exc:
                yield ("error", exc.detail)
                return
            texts, thinks, calls, done = [], [], [], None
            async for kind, v in events(job, req):
                if kind == "text":
                    texts.append(v)
                    yield ("text", v)
                elif kind == "thinking":
                    thinks.append(v)
                elif kind == "tool_call":
                    calls.append(v)
                elif kind == "error":
                    yield ("error", v)
                    return
                elif kind == "done":
                    done = v
            if done is None:
                return
            if not calls or not tools:
                yield ("done", done)
                return
            executed = []
            for call in calls:
                if used >= _MAX_TOOL_CALLS:
                    break
                if call.name not in ("list_dir", "read_file", "write_file"):
                    result = f"不会执行这个工具：{call.name}"
                else:
                    args_ = call.arguments or {}
                    path = str(args_.get("path") or "")
                    try:
                        if call.name == "list_dir":
                            listed = list_dir(path)
                            if not (listed["path"] == root or listed["path"].startswith(root + os.sep)):
                                raise FsError("只能看当前打开的目录")
                            lines = [listed["path"]] + [("/" if e["dir"] else " ") + e["name"] for e in listed["entries"]]
                            result = "\n".join(lines)
                        elif call.name == "read_file":
                            result = read_text(inside(path, root), limit=12_000)
                        else:
                            written = write_text(path, str(args_.get("content") or ""), root)
                            result = "已写入 " + written
                    except FsError as exc:
                        result = str(exc)
                used += 1
                executed.append((call, result))
                if await req.is_disconnected():
                    return
            msgs.append({"role": "assistant", "content": "".join(texts), "reasoning_content": "".join(thinks),
                         "tool_calls": [{"name": c.name, "arguments": c.arguments} for c, _ in executed]})
            for call, result in executed:
                msgs.append({"role": "tool", "content": result})
                yield ("tool_result", {"name": call.name, "path": str((call.arguments or {}).get("path") or ""),
                                       "content": result})
            if used >= _MAX_TOOL_CALLS:
                tools = []

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request, response: Response):
        auth(req)
        body = await req.json()
        msgs = from_openai(body.get("messages") or [])
        tools = tools_from_openai(body.get("tools")) if body.get("tool_choice") != "none" else []
        thinking = want_thinking(body, "openai")
        if body.get("code_tools"):
            req.state.seat_token = hold_page(req, response)
            try:
                root = inside(str(body.get("workspace") or ""), str(body.get("workspace") or ""))
            except FsError as exc:
                raise HTTPException(400, str(exc))
            rid, created, model = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), body.get("model") or served

            def cchunk(delta, finish=None):
                return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            async def cgen():
                yield sse(cchunk({"role": "assistant", "content": ""}))
                async for kind, v in code_rounds(body, msgs, req, root):
                    if kind == "text":
                        yield sse(cchunk({"content": v}))
                    elif kind == "tool_result":
                        yield sse({"tool_result": v})
                    elif kind == "error":
                        yield sse({"error": {"message": v if isinstance(v, str) else str(v), "type": "server_error"}})
                    elif kind == "done":
                        yield sse(cchunk({}, "stop"))
                yield "data: [DONE]\n\n"
            return StreamingResponse(cgen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

        if body.get("server_tools"):
            req.state.seat_token = hold_page(req, response)
            msgs = with_search_guide(msgs)
            rid, created, model = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), body.get("model") or served

            def schunk(delta, finish=None):
                return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            if body.get("stream"):
                async def sgen():
                    yield sse(schunk({"role": "assistant", "content": ""}))
                    async for kind, v in tool_rounds(body, msgs, req):
                        if kind == "text":
                            yield sse(schunk({"content": v}))
                        elif kind == "thinking":
                            yield sse(schunk({"reasoning_content": v}))
                        elif kind == "search":
                            yield sse({"search": v})
                        elif kind == "tool_result":
                            yield sse({"tool_result": v})
                        elif kind == "fold":
                            yield sse({"fold": v})
                        elif kind == "error":
                            yield sse({"error": {"message": v, "type": "server_error"}})
                        elif kind == "done":
                            yield sse(schunk({}, "stop"))
                    yield "data: [DONE]\n\n"
                return StreamingResponse(sgen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

            text_out, think_out, err, done, fold_out = [], [], None, None, None
            async for kind, v in tool_rounds(body, msgs, req):
                if kind == "text":
                    text_out.append(v)
                elif kind == "thinking":
                    think_out.append(v)
                elif kind == "error":
                    err = v
                elif kind == "done":
                    done = v
                elif kind == "fold":
                    fold_out = v
            if err:
                raise HTTPException(500, err)
            if done is None:
                raise HTTPException(499, "client disconnected")
            msg = {"role": "assistant", "content": "".join(text_out)}
            if think_out:
                msg["reasoning_content"] = "".join(think_out)
            out = {"id": rid, "object": "chat.completion", "created": created, "model": model,
                   "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": done["input_tokens"], "completion_tokens": done["output_tokens"],
                             "total_tokens": done["input_tokens"] + done["output_tokens"]}}
            if fold_out:
                out["fold"] = fold_out
            return out

        text = render(tok, msgs, tools, think=thinking, reasoning_effort=body.get("reasoning_effort"))
        ids = encode(text)
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        max_new = body.get("max_completion_tokens") or body.get("max_tokens") or args.max_new
        job = make_job(ids, body, max_new, stops, schemas_of(tools), thinking, text)
        rid, created, model = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), body.get("model") or served
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def chunk(delta, finish=None, usage=None):
            c = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage is not None:
                c["usage"] = usage
            return c

        def usage_of(d):
            return {"prompt_tokens": d["input_tokens"], "completion_tokens": d["output_tokens"],
                    "total_tokens": d["input_tokens"] + d["output_tokens"]}

        def finish_of(d):
            return {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls",
                    "cancelled": "stop"}[d["stop_reason"]]

        if body.get("stream"):
            async def gen():
                yield sse(chunk({"role": "assistant", "content": ""}))
                n_calls = 0
                async for kind, v in events(job, req):
                    if kind == "text":
                        yield sse(chunk({"content": v}))
                    elif kind == "thinking":
                        yield sse(chunk({"reasoning_content": v}))
                    elif kind == "tool_call":
                        yield sse(chunk({"tool_calls": [{"index": n_calls, "id": v.id, "type": "function",
                                                         "function": {"name": v.name, "arguments": json.dumps(v.arguments, ensure_ascii=False)}}]}))
                        n_calls += 1
                    elif kind == "error":
                        yield sse({"error": {"message": v, "type": "server_error"}})
                    else:
                        yield sse(chunk({}, finish_of(v), usage_of(v) if include_usage else None))
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

        text_out, think_out, calls, done, err = [], [], [], None, None
        async for kind, v in events(job, req):
            if kind == "text":
                text_out.append(v)
            elif kind == "thinking":
                think_out.append(v)
            elif kind == "tool_call":
                calls.append(v)
            elif kind == "error":
                err = v
            else:
                done = v
        if err:
            raise HTTPException(500, err)
        msg = {"role": "assistant", "content": "".join(text_out) or (None if calls else "")}
        if think_out:
            msg["reasoning_content"] = "".join(think_out)
        if calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                                 for c in calls]
        return {"id": rid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish_of(done)}], "usage": usage_of(done)}

    @app.post("/v1/completions")
    async def completions(req: Request):
        auth(req)
        body = await req.json()
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], int):
                ids, prompt = list(prompt), ""
            else:
                prompt = "".join(prompt)
                ids = encode(prompt)
        else:
            ids = encode(prompt)
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        job = make_job(ids, body, body.get("max_tokens") or args.max_new, stops, {}, False, prompt)
        job.thinking = False
        rid, created, model = "cmpl-" + uuid.uuid4().hex[:24], int(time.time()), body.get("model") or served
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        def chunk(text, finish=None, usage=None):
            c = {"id": rid, "object": "text_completion", "created": created, "model": model,
                 "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}]}
            if usage is not None:
                c["usage"] = usage
            return c

        def usage_of(d):
            return {"prompt_tokens": d["input_tokens"], "completion_tokens": d["output_tokens"],
                    "total_tokens": d["input_tokens"] + d["output_tokens"]}

        finish_of = lambda d: "length" if d["stop_reason"] == "max_tokens" else "stop"
        if body.get("stream"):
            async def gen():
                async for kind, v in events(job, req):
                    if kind in ("text", "thinking"):
                        yield sse(chunk(v))
                    elif kind == "tool_call":
                        yield sse(chunk("<tool_call>" + json.dumps({"name": v.name, "arguments": v.arguments}) + "</tool_call>"))
                    elif kind == "error":
                        yield sse({"error": {"message": v, "type": "server_error"}})
                    else:
                        yield sse(chunk("", finish_of(v), usage_of(v) if include_usage else None))
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
        parts, done, err = [], None, None
        async for kind, v in events(job, req):
            if kind in ("text", "thinking"):
                parts.append(v)
            elif kind == "tool_call":
                parts.append("<tool_call>" + json.dumps({"name": v.name, "arguments": v.arguments}) + "</tool_call>")
            elif kind == "error":
                err = v
            else:
                done = v
        if err:
            raise HTTPException(500, err)
        return {"id": rid, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": "".join(parts), "finish_reason": finish_of(done), "logprobs": None}],
                "usage": usage_of(done)}

    # ---------------------------------------------------------------- Anthropic

    def anthropic_error(status, etype, message):
        return JSONResponse({"type": "error", "error": {"type": etype, "message": message}}, status_code=status)

    def prepare_messages(body: dict):
        msgs = from_anthropic(body.get("system"), body.get("messages") or [])
        tc = body.get("tool_choice") or {}
        tools = [] if tc.get("type") == "none" else tools_from_anthropic(body.get("tools"))
        thinking = want_thinking(body, "anthropic")
        text = render(tok, msgs, tools, think=thinking)
        return text, encode(text), tools, thinking

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(req: Request):
        auth(req)
        body = await req.json()
        _, ids, _, _ = prepare_messages(body)
        return {"input_tokens": len(ids)}

    @app.post("/v1/messages")
    async def messages(req: Request):
        auth(req)
        body = await req.json()
        if "max_tokens" not in body:
            return anthropic_error(400, "invalid_request_error", "max_tokens is required")
        text, ids, tools, thinking = prepare_messages(body)
        try:
            job = make_job(ids, body, body["max_tokens"], body.get("stop_sequences") or [], schemas_of(tools), thinking, text)
        except HTTPException as e:
            return anthropic_error(e.status_code, "invalid_request_error", str(e.detail))
        mid, model = "msg_" + uuid.uuid4().hex[:24], body.get("model") or served

        def usage_of(d):
            return {"input_tokens": d["input_tokens"], "output_tokens": d["output_tokens"],
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": d["stats"].get("reused_tokens", 0)}

        stop_of = lambda d: "end_turn" if d["stop_reason"] == "cancelled" else d["stop_reason"]

        if body.get("stream"):
            async def gen():
                yield sse({"type": "message_start", "message": {"id": mid, "type": "message", "role": "assistant", "model": model,
                                                                 "content": [], "stop_reason": None, "stop_sequence": None,
                                                                 "usage": {"input_tokens": len(ids), "output_tokens": 0}}}, "message_start")
                idx, open_kind = -1, None          # the content block currently open

                def close():
                    return sse({"type": "content_block_stop", "index": idx}, "content_block_stop")

                async for kind, v in events(job, req):
                    if kind in ("text", "thinking"):
                        if open_kind != kind:
                            if open_kind is not None:
                                yield close()
                            idx += 1
                            open_kind = kind
                            block = {"type": "text", "text": ""} if kind == "text" else {"type": "thinking", "thinking": "", "signature": ""}
                            yield sse({"type": "content_block_start", "index": idx, "content_block": block}, "content_block_start")
                        delta = {"type": "text_delta", "text": v} if kind == "text" else {"type": "thinking_delta", "thinking": v}
                        yield sse({"type": "content_block_delta", "index": idx, "delta": delta}, "content_block_delta")
                    elif kind == "tool_call":
                        if open_kind is not None:
                            yield close()
                        idx += 1
                        open_kind = None
                        tid = "toolu_" + v.id.removeprefix("call_")
                        yield sse({"type": "content_block_start", "index": idx,
                                   "content_block": {"type": "tool_use", "id": tid, "name": v.name, "input": {}}}, "content_block_start")
                        yield sse({"type": "content_block_delta", "index": idx,
                                   "delta": {"type": "input_json_delta", "partial_json": json.dumps(v.arguments, ensure_ascii=False)}}, "content_block_delta")
                        yield close()
                    elif kind == "error":
                        yield sse({"type": "error", "error": {"type": "api_error", "message": v}}, "error")
                    else:
                        if open_kind is not None:
                            yield close()
                        yield sse({"type": "message_delta", "delta": {"stop_reason": stop_of(v), "stop_sequence": v["stop_sequence"]},
                                   "usage": usage_of(v)}, "message_delta")
                        yield sse({"type": "message_stop"}, "message_stop")
            return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

        content, done, err = [], None, None
        async for kind, v in events(job, req):
            if kind == "text":
                if content and content[-1]["type"] == "text":
                    content[-1]["text"] += v
                else:
                    content.append({"type": "text", "text": v})
            elif kind == "thinking":
                if content and content[-1]["type"] == "thinking":
                    content[-1]["thinking"] += v
                else:
                    content.append({"type": "thinking", "thinking": v, "signature": ""})
            elif kind == "tool_call":
                content.append({"type": "tool_use", "id": "toolu_" + v.id.removeprefix("call_"), "name": v.name, "input": v.arguments})
            elif kind == "error":
                err = v
            else:
                done = v
        if err:
            return anthropic_error(500, "api_error", err)
        return {"id": mid, "type": "message", "role": "assistant", "model": model, "content": content,
                "stop_reason": stop_of(done), "stop_sequence": done["stop_sequence"], "usage": usage_of(done)}

    app.state.t0 = time.time()

    # The code page is OpenCode, on this same port. Its own server stays on
    # 127.0.0.1:4096; the browser only talks to us. /seat is ours, so it does
    # not collide with OpenCode's /session.
    oc_skip = {"host", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
               "trailers", "transfer-encoding", "upgrade", "content-length", "content-encoding", "authorization"}

    def oc_basic():
        user = os.environ.get("OPENCODE_SERVER_USERNAME") or "opencode"
        pw = os.environ.get("OPENCODE_SERVER_PASSWORD") or _SEAT_PASSWORD
        return user, pw, "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()

    @app.api_route("/{full:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    async def code_proxy(full: str, req: Request):
        import httpx
        box = Response()
        hold_page(req, box)
        user, pw, _ = oc_basic()
        url = "http://127.0.0.1:4096/" + full
        if req.url.query:
            url += "?" + req.url.query
        headers = {k: v for k, v in req.headers.items() if k.lower() not in oc_skip}
        client = httpx.AsyncClient(timeout=None, auth=(user, pw))
        try:
            upstream = await client.send(
                client.build_request(req.method, url, headers=headers, content=await req.body()), stream=True)
        except httpx.ConnectError:
            await client.aclose()
            raise HTTPException(502, "写代码没有启动")
        passed = {k: v for k, v in upstream.headers.items()
                  if k.lower() not in {"content-length", "content-encoding", "transfer-encoding", "connection", "set-cookie"}}

        async def chunks():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        out = StreamingResponse(chunks(), status_code=upstream.status_code, headers=passed)
        for key, value in box.raw_headers:
            if key.lower() == b"set-cookie":
                out.raw_headers.append((key, value))
        return out

    @app.websocket("/{full:path}")
    async def code_ws(ws: WebSocket, full: str):
        from websockets.asyncio.client import connect as ws_connect
        if seat_view(ws)[0] != "in":
            await ws.close(code=1008)
            return
        await ws.accept()
        _, _, basic = oc_basic()
        url = "ws://127.0.0.1:4096/" + full
        if ws.url.query:
            url += "?" + ws.url.query
        try:
            remote_cm = ws_connect(url, additional_headers={"Authorization": basic}, max_size=None, proxy=None)
            remote = await remote_cm.__aenter__()
        except Exception:
            await ws.close(code=1011)
            return

        async def from_browser():
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        return
                    if msg.get("bytes") is not None:
                        await remote.send(msg["bytes"])
                    elif msg.get("text") is not None:
                        await remote.send(msg["text"])
            finally:
                await remote.close()

        async def from_code():
            try:
                async for message in remote:
                    if isinstance(message, str):
                        await ws.send_text(message)
                    else:
                        await ws.send_bytes(message)
            finally:
                try:
                    await ws.close()
                except RuntimeError:
                    pass

        try:
            await asyncio.gather(from_browser(), from_code())
        finally:
            await remote_cm.__aexit__(None, None, None)

    return app


# ------------------------------------------------------------ startup


def load_session(args):
    """The same construction run.py does, kept resident."""
    from transformers import AutoTokenizer
    from .model import Engine
    from .quant import DEFAULT_BACKEND
    from .weights import is_packed, load_packed, not_packed_message, resolve_model
    args.model = resolve_model(args.model, download=not args.no_download)
    if not is_packed(args.model):
        raise SystemExit(not_packed_message(args.model))
    want_dflash = args.draft in ("auto", "dflash")
    want_mtp = args.draft in ("auto", "mtp")
    if want_dflash:
        try:
            args.dflash_path = resolve_model(args.dflash_path, download=not args.no_download)
        except SystemExit as e:
            print(f"[warn] {e}; serving with the MTP draft")
            want_dflash, want_mtp = False, True
    cfg, w, mtp_t = load_packed(args.model, backend=args.backend or DEFAULT_BACKEND, with_mtp=want_mtp)
    tok = AutoTokenizer.from_pretrained(args.model)
    kv = torch.float8_e4m3fn if args.kv == "fp8" else torch.bfloat16
    kmin, kmax = 3, 4
    engine = Engine(cfg, w, max_len=args.max_len, kv_dtype=kv, max_spec=7 if want_dflash else (kmax if want_mtp else 0))
    t0 = time.perf_counter()
    engine.capture()
    dflash = mtp = None
    dv = torch.arange(131072)
    if want_dflash:
        from .dflash import DFlashDraft, load_dflash
        dflash = DFlashDraft(load_dflash(args.dflash_path, int4=True), w.embed, w.lm_head, cfg.hidden, args.max_len)
        engine.attach_dflash(dflash, draft_vocab=dv)
        engine.capture_spec_dflash()
    if want_mtp:
        from .mtp import MTPHead, build_mtp
        mtp = MTPHead(cfg, build_mtp(cfg, mtp_t, "cuda", int4=True), w.embed, w.lm_head, args.max_len, kv_dtype=engine.state.kv_dtype)
        engine.attach_mtp(mtp, draft_vocab=dv)
        for k in range(kmin, kmax + 1):
            engine.capture_spec(k)
    print(f"captured the graphs in {time.perf_counter() - t0:.1f}s; weights {w.nbytes / 1e9:.2f} GB, state {engine.state.nbytes / 1e9:.2f} GB, "
          f"cuda allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB; context {args.max_len}, drafts: "
          f"{'DFlash2 ' if dflash else ''}{'MTP' if mtp else ''}{'none' if not (dflash or mtp) else ''}", flush=True)
    session = Session(engine, tok, cfg, dflash=dflash, mtp=mtp, chunk=args.chunk, kmin=kmin, kmax=kmax)
    # warm up every prefill shape class (Triton autotunes the first call of each) and the
    # spec loop, so the first request does not pay for it
    t0 = time.perf_counter()
    rows = engine.state.n_slots
    for mode in [m for m, d in (("dflash", dflash), ("mtp", mtp)) if d is not None] or ["raw"]:
        # one eager chunk, then every fused row count 1..rows (each M autotunes separately)
        for n in [args.chunk + 1] + [rows + m for m in range(1, rows + 1)]:
            g = session.generate(torch.randint(1000, 100000, (n,)).tolist(), max_new=4, mode=mode)
            for _ in g:
                pass
    session.forget()
    print(f"warmed up in {time.perf_counter() - t0:.1f}s", flush=True)
    return session, tok, cfg


def main():
    from .weights import DEFAULT_REPO, DFLASH_REPO
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=DEFAULT_REPO, help="packed checkpoint: a Hub repo id or a local directory")
    ap.add_argument("--dflash-path", default=DFLASH_REPO)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--draft", default="auto", choices=("auto", "dflash", "mtp", "raw"),
                    help="auto (default): both drafts resident, the MTP chain for CJK prompts, DFlash2 otherwise")
    ap.add_argument("--backend", default=None, choices=("marlin", "triton", "tinygemm", "dequant"))
    ap.add_argument("--kv", default="fp8", choices=("bf16", "fp8"))
    ap.add_argument("--max-len", type=int, default=262144, help="the context window (256k needs ~30 GB with both drafts)")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--think", default="auto", choices=("auto", "on", "off"),
                    help="thinking: auto follows the request (Anthropic `thinking`, OpenAI `reasoning_effort` / chat_template_kwargs)")
    ap.add_argument("--temperature", type=float, default=0.7, help="when the request does not say")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=8192, help="max_tokens when the request does not say")
    ap.add_argument("--chats", default=default_chats_path(), help="JSON file the web page saves conversations in")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=os.environ.get("TOKENRUSH_API_KEY"), help="require it as x-api-key / Bearer")
    ap.add_argument("--served-name", default="token-rush")
    ap.add_argument("--alias", action="append", default=[], help="extra model names to list")
    args = ap.parse_args()
    session, tok, cfg = load_session(args)
    print(f"web search: {backend_name()}", flush=True)
    print(f"chat page http://{args.host}:{args.port}/", flush=True)
    print(f"conversations {args.chats}", flush=True)
    import uvicorn
    uvicorn.run(build_app(session, tok, cfg, args), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
