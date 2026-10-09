"""Web search and page fetch for the server-side tool loop.

The model never opens a socket. `web_search` calls SearXNG (`SEARXNG_URL`) or
Brave (`BRAVE_API_KEY`) with the query alone. `web_fetch` reads one https URL
and returns its readable text. Private addresses, plain http, and redirects
onto them are refused.
"""
import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

_MAX_HITS = 5
_FETCH_BYTES = 1_000_000
_FETCH_CHARS = 4000
_TIMEOUT = 8
_SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "header", "svg"}


def backend_name() -> str:
    if os.environ.get("SEARXNG_URL", "").strip():
        return "searxng " + os.environ["SEARXNG_URL"].strip()
    if os.environ.get("BRAVE_API_KEY", "").strip():
        return "brave"
    return "未配置（设置 SEARXNG_URL 或 BRAVE_API_KEY）"


def web_search(query: str):
    """query -> (text for the tool message, [{title, url}])."""
    query = (query or "").strip()
    if not query:
        return "没有查询词。", []
    try:
        if os.environ.get("SEARXNG_URL", "").strip():
            hits = _searxng(query)
        elif os.environ.get("BRAVE_API_KEY", "").strip():
            hits = _brave(query)
        else:
            return "没有配置搜索。设置环境变量 SEARXNG_URL 或 BRAVE_API_KEY。", []
    except Exception as exc:
        return f"搜索失败：{exc.__class__.__name__}: {exc}", []
    if not hits:
        return "没有结果。", []
    lines, sources = [], []
    for i, h in enumerate(hits[:_MAX_HITS], 1):
        lines.append(f"{i}. {h['title']}\n{h['url']}\n{h['snippet']}")
        sources.append({"title": h["title"], "url": h["url"]})
    return "\n\n".join(lines), sources


def web_fetch(url: str):
    """url -> (text for the tool message, [{title, url}] or [])."""
    url = (url or "").strip()
    err = public_https_error(url)
    if err:
        return "拒绝：" + err, []
    try:
        body, final, ctype = _get(url)
    except Exception as exc:
        return f"抓取失败：{exc.__class__.__name__}: {exc}", []
    if "html" in ctype:
        text = _html_text(body)
    else:
        text = body.decode("utf-8", "replace")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > _FETCH_CHARS:
        text = text[:_FETCH_CHARS] + "\n…（已截断）"
    if not text:
        return "页面没有可读正文。", [{"title": final, "url": final}]
    return text, [{"title": final, "url": final}]


def public_https_error(url: str):
    """None when url is https and every resolved address is a public IP."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return "URL 无法解析"
    if parts.scheme != "https":
        return "只允许 https"
    host = parts.hostname or ""
    if not host or host.lower() in ("localhost",) or host.lower().endswith((".local", ".internal")):
        return "拒绝内网地址"
    return _host_error(host)


def _host_error(host: str):
    try:
        ips = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        except socket.gaierror:
            return "无法解析主机名"
        ips = []
        for info in infos:
            try:
                ips.append(ipaddress.ip_address(info[4][0]))
            except ValueError:
                return "无法解析主机名"
        if not ips:
            return "无法解析主机名"
    for ip in ips:
        if not ip.is_global:
            return "拒绝内网地址"
    return None


def _searxng(query: str):
    base = os.environ["SEARXNG_URL"].strip().rstrip("/")
    url = base + "/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
    data = json.loads(_get(url, limit=2_000_000)[0].decode("utf-8", "replace"))
    hits = []
    for row in data.get("results") or []:
        if not row.get("url"):
            continue
        hits.append({"title": row.get("title") or row["url"], "url": row["url"], "snippet": row.get("content") or ""})
        if len(hits) >= _MAX_HITS:
            break
    return hits


def _brave(query: str):
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": _MAX_HITS})
    raw = _get(url, headers={"X-Subscription-Token": os.environ["BRAVE_API_KEY"].strip(), "Accept": "application/json"})[0]
    data = json.loads(raw.decode("utf-8", "replace"))
    hits = []
    for row in ((data.get("web") or {}).get("results") or []):
        if not row.get("url"):
            continue
        hits.append({"title": row.get("title") or row["url"], "url": row["url"], "snippet": row.get("description") or ""})
    return hits


class _NoRedirect(urllib.request.HTTPErrorProcessor):
    def http_response(self, request, response):
        return response

    https_response = http_response


def _get(url: str, headers=None, limit=_FETCH_BYTES):
    """GET, following at most two redirects that stay on public https. Returns (body, final_url, content_type)."""
    opener = urllib.request.build_opener(_NoRedirect)
    current = url
    for _ in range(3):
        if current != url:
            err = public_https_error(current)
            if err:
                raise RuntimeError(err)
        req = urllib.request.Request(current, headers={"User-Agent": "TokenRush/0.1", **(headers or {})})
        try:
            resp = opener.open(req, timeout=_TIMEOUT)
        except urllib.error.HTTPError as exc:
            resp = exc
        code = getattr(resp, "status", None) or resp.getcode()
        if code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            resp.close()
            if not loc:
                raise RuntimeError(f"HTTP {code} without Location")
            current = urllib.parse.urljoin(current, loc)
            continue
        if code >= 400:
            resp.close()
            raise RuntimeError(f"HTTP {code}")
        body = resp.read(limit + 1)
        ctype = resp.headers.get("Content-Type", "")
        resp.close()
        if len(body) > limit:
            body = body[:limit]
        return body, current, ctype.lower()
    raise RuntimeError("重定向次数过多")


class _VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _html_text(body: bytes) -> str:
    parser = _VisibleText()
    parser.feed(body.decode("utf-8", "replace"))
    return "".join(parser.parts)
