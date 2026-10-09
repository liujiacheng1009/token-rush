"""Look up another saved chat only when the user asked about one.

A new conversation is not seeded with older chats. `recall` reads
`chats.json` on the CPU and skips the chat this request belongs to.
If an excerpt and this chat name different values, the tool text says
this chat wins.
"""
import re
import time

_PAST = re.compile(r"上次|上一次|上一回|以前|之前聊|之前说|旧对话|另一段|上一段|那段对话|历史对话|你说过|记得你")
_NUM = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_URL = re.compile(r"https://[^\s<>)\]]+")
_MAX_HITS = 3


def asks_about_past(messages) -> bool:
    """True when the latest user line is asking about an earlier chat."""
    for m in reversed(messages or []):
        if m.get("role") == "user" and (m.get("content") or "").strip():
            return _PAST.search(m["content"]) is not None
    return False


def shorten_tool(message: dict) -> dict:
    """One line for a tool result that is being folded into a prompt."""
    if message.get("role") != "tool":
        return message
    urls = _URL.findall(message.get("content") or "")
    line = "已搜索或打开：" + " ".join(urls[:5]) if urls else "已搜索。"
    out = dict(message)
    out["content"] = line
    return out


def recall(store, query: str, chat_id: str, messages) -> str:
    """Tool text. Does not open the chat file unless the user asked about the past."""
    if not asks_about_past(messages):
        return "用户没有在问以前的对话，未查找。按本段已有内容回答。"
    query = (query or "").strip()
    if not query:
        return "没有查询词。未查找。"
    return search_other(store, query, chat_id, messages)


def search_other(store, query: str, chat_id: str, messages) -> str:
    hits = []
    needle = query.casefold()
    for row in store.list():
        if row["id"] == chat_id:
            continue
        chat = store.get(row["id"])
        if not chat:
            continue
        for msg in chat.get("messages") or []:
            content = msg.get("content") or ""
            if needle not in content.casefold():
                continue
            hits.append({"day": _day(chat.get("created")), "title": chat.get("title") or "新对话",
                         "snippet": _snippet(content, needle)})
            break
        if len(hits) >= _MAX_HITS:
            break
    if not hits:
        return "其它对话里没有找到这句。不要据此编造旧记录。"
    lines = []
    for i, hit in enumerate(hits, 1):
        lines.append(f"{i}. {hit['day']} 《{hit['title']}》\n{hit['snippet']}")
    text = "\n\n".join(lines)
    clash = _conflict(messages, text)
    if clash:
        text += "\n\n" + clash
    return text


def _day(created) -> str:
    try:
        return time.strftime("%Y-%m-%d", time.localtime(int(created)))
    except (TypeError, ValueError, OSError):
        return ""


def _snippet(content: str, needle: str) -> str:
    for line in content.splitlines():
        if needle in line.casefold() and line.strip():
            return line.strip()[:240]
    i = content.casefold().find(needle)
    return " ".join(content[max(0, i - 40):i + 160].split())


def _conflict(messages, excerpt: str) -> str:
    here = set()
    for m in messages or []:
        if m.get("role") == "user":
            here.update(_NUM.findall(m.get("content") or ""))
    there = set(_NUM.findall(excerpt))
    if not here or not there or here == there or here <= there:
        return ""
    old = "、".join(sorted(there - here))
    cur = "、".join(sorted(here - there))
    if not old or not cur:
        return ""
    return f"冲突：本段里有 {cur}，旧记录里有 {old}。以本段为准，不要用旧记录改正本段。"
