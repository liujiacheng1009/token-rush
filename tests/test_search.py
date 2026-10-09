"""Search formatting and the web_fetch address check. No GPU, no outbound calls
except a local fake SearXNG."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from tokenrush.search import public_https_error, web_search


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


def test_unconfigured_search_is_an_error_string(monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    text, sources = web_search("qwen")
    assert "没有配置搜索" in text and sources == []
