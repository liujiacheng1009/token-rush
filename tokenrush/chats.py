"""One local JSON file of chat transcripts. The GPU session is not involved:
a turn is the whole message list, re-sent to /v1/chat/completions."""
import json
import os
import re
import threading
import time

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ROLES = {"system", "user", "assistant", "tool"}


class ChatError(ValueError):
    pass


def default_chats_path() -> str:
    override = os.environ.get("TOKENRUSH_CHATS")
    if override:
        return override
    base = os.environ.get("XDG_DATA_HOME", os.path.join(os.path.expanduser("~"), ".local", "share"))
    return os.path.join(base, "tokenrush", "chats.json")


class ChatStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def list(self):
        with self._lock:
            chats = self._read()
        chats.sort(key=lambda c: c.get("created", 0), reverse=True)
        return [{"id": c["id"], "title": c["title"], "created": c["created"]} for c in chats]

    def get(self, chat_id: str):
        self._check_id(chat_id)
        with self._lock:
            for c in self._read():
                if c["id"] == chat_id:
                    return c
        return None

    def put(self, chat_id: str, body: dict) -> dict:
        self._check_id(chat_id)
        if not isinstance(body, dict):
            raise ChatError("body must be an object")
        messages = self._messages(body.get("messages", []))
        created = body.get("created")
        created = int(created) if isinstance(created, (int, float)) else int(time.time())
        chat = {"id": chat_id, "title": self._title(body.get("title"), messages), "created": created, "messages": messages}
        if "fold" in body and body.get("fold") is not None:
            chat["fold"] = self._fold(body.get("fold"))
        with self._lock:
            chats = self._read()
            for i, c in enumerate(chats):
                if c["id"] == chat_id:
                    chats[i] = chat
                    break
            else:
                chats.append(chat)
            self._write(chats)
        return chat

    def delete(self, chat_id: str) -> bool:
        self._check_id(chat_id)
        with self._lock:
            chats = self._read()
            kept = [c for c in chats if c["id"] != chat_id]
            if len(kept) == len(chats):
                return False
            self._write(kept)
        return True

    def _check_id(self, chat_id: str):
        if not isinstance(chat_id, str) or not _ID.match(chat_id):
            raise ChatError("id must be 1–64 characters of letters, digits, _ or -")

    def _messages(self, raw):
        if not isinstance(raw, list):
            raise ChatError("messages must be a list")
        out = []
        for m in raw:
            if not isinstance(m, dict) or m.get("role") not in ROLES:
                raise ChatError("each message needs role system|user|assistant|tool and a string content")
            content = m.get("content")
            if content is None and m.get("role") == "assistant":
                content = ""
            if not isinstance(content, str):
                raise ChatError("each message needs role system|user|assistant|tool and a string content")
            item = {"role": m["role"], "content": content}
            if m["role"] == "assistant" and m.get("tool_calls") is not None:
                item["tool_calls"] = self._tool_calls(m["tool_calls"])
            if m["role"] == "tool":
                if isinstance(m.get("name"), str):
                    item["name"] = m["name"]
                if isinstance(m.get("tool_call_id"), str):
                    item["tool_call_id"] = m["tool_call_id"]
            out.append(item)
        return out

    def _tool_calls(self, raw):
        if not isinstance(raw, list):
            raise ChatError("tool_calls must be a list")
        out = []
        for c in raw:
            if not isinstance(c, dict):
                raise ChatError("tool call must be an object")
            fn = c.get("function") if isinstance(c.get("function"), dict) else c
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                raise ChatError("tool call needs a name")
            args = fn.get("arguments", c.get("arguments", {}))
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            elif not isinstance(args, str):
                raise ChatError("tool call arguments must be an object or a string")
            item = {"type": "function", "function": {"name": name, "arguments": args}}
            if isinstance(c.get("id"), str):
                item["id"] = c["id"]
            out.append(item)
        return out

    def _fold(self, raw):
        if not isinstance(raw, dict):
            raise ChatError("fold must be an object")
        summary = raw.get("summary")
        before = raw.get("before")
        if not isinstance(summary, str) or not summary.strip():
            raise ChatError("fold needs a summary")
        if type(before) is not int or before < 1:
            raise ChatError("fold needs a before index")
        return {"summary": summary.strip()[:4000], "before": before}

    def _title(self, title, messages):
        if isinstance(title, str) and title.strip():
            return title.strip()[:80]
        for m in messages:
            if m["role"] == "user" and m["content"].strip():
                return m["content"].strip().replace("\n", " ")[:24]
        return "新对话"

    def _read(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("chats", [])
        if not isinstance(data, list):
            return []
        return data

    def _write(self, chats):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"chats": chats}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)
