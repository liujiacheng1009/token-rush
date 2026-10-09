"""Search formatting and the web_fetch address check. No GPU, no outbound calls
except a local fake SearXNG."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from tokenrush.search import public_https_error, web_fetch, web_search


def test_public_https_rejects_plain_http_and_private_addresses():
    assert public_https_error("http://example.com/a") == "只允许 https"
    assert public_https_error("https://127.0.0.1/a") == "拒绝内网地址"
    assert public_https_error("https://10.1.1.1/a") == "拒绝内网地址"
    assert public_https_error("https://localhost/a") == "拒绝内网地址"


def test_searxng_returns_title_url_snippet(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"results": [
                {"title": "A", "url": "https://example.com/a", "content": "one"},
                {"title": "B", "url": "https://example.com/b", "content": "two"},
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("SEARXNG_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    try:
        text, sources = web_search("qwen")
    finally:
        server.shutdown()
    assert "https://example.com/a" in text and "one" in text
    assert [s["url"] for s in sources] == ["https://example.com/a", "https://example.com/b"]


def test_fetch_keeps_article_tex_and_drops_chrome(monkeypatch):
    html = b"""<html><body><nav>skip me</nav><main>
      <aside class="theme-doc-toc-desktop">On this page toc</aside>
      <article>
        <p>Hello <span class="katex"><span class="katex-mathml"><math><semantics>
          <annotation encoding="application/x-tex">k_1</annotation></semantics></math></span>
          <span class="katex-html">XXX</span></span> world</p>
        <p>Tail of the model.</p>
      </article></main><footer>foot</footer></body></html>"""
    monkeypatch.setattr("tokenrush.search.public_https_error", lambda url: None)
    monkeypatch.setattr("tokenrush.search._get", lambda url, **kw: (html, url, "text/html"))
    text, sources = web_fetch("https://example.com/aria")
    assert "Hello" in text and "$k_1$" in text and "Tail of the model." in text
    assert "XXX" not in text and "skip me" not in text and "toc" not in text and "foot" not in text
    assert sources[0]["url"] == "https://example.com/aria"


def test_fetch_truncates_on_a_paragraph(monkeypatch):
    paragraph = "段" * 1000
    html = ("<article>" + "".join(f"<p>{paragraph} {i}</p>" for i in range(20)) + "</article>").encode()
    monkeypatch.setattr("tokenrush.search.public_https_error", lambda url: None)
    monkeypatch.setattr("tokenrush.search._get", lambda url, **kw: (html, url, "text/html"))
    text, _ = web_fetch("https://example.com/long")
    assert text.startswith("段")
    assert "已截断，后面还有" in text
    assert len(text) < 16000 + 40


def test_unconfigured_search_is_an_error_string(monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    text, sources = web_search("qwen")
    assert "没有配置搜索" in text and sources == []
