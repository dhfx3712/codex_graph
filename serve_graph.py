#!/usr/bin/env python3
"""用 Python 标准库在云服务器上启动图页面静态服务。

用法：
    .venv/bin/python serve_graph.py --host 0.0.0.0 --port 6134

浏览器访问：
    http://<服务器公网IP>:6134/
"""

from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path


WEB_DIR = Path(__file__).resolve().parent / "web"


class GraphHandler(http.server.SimpleHTTPRequestHandler):
    public_host = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def parse_request(self):
        raw = getattr(self, "raw_requestline", b"")
        if raw.startswith(b"\x16\x03"):
            host = self.public_host or self.server.server_address[0]
            self.log_error("HTTPS/TLS request rejected; use http://%s:%s/",
                           host, self.server.server_address[1])
            self.close_connection = True
            return False
        return super().parse_request()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(f"[graph] {self.address_string()} - {fmt % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6134)
    parser.add_argument("--public-host", default="", help="用于日志提示的公网 IP 或域名，默认使用 --host")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not WEB_DIR.exists():
        print(f"找不到前端目录：{WEB_DIR}", file=__import__("sys").stderr)
        return 2

    GraphHandler.public_host = args.public_host
    display_host = args.public_host or args.host
    handler = functools.partial(GraphHandler)
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        print(f"图网络页面已启动：http://{display_host}:{args.port}/")
        if not args.public_host and args.host in {"0.0.0.0", "::"}:
            print("提示：如需让拒绝 HTTPS 的日志显示可访问地址，请加 --public-host <公网IP或域名>")
        print(f"按 Ctrl+C 停止服务")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止服务")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
