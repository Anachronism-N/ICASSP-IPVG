#!/usr/bin/env python3
"""
Serve the review demo (demo/index.html + demo/media) with HTTP Range support (needed for video
seeking; the stdlib http.server ignores Range) and rebuild the page every REBUILD_MIN minutes so
training progress / eval results show up without manual steps.

Usage: python3 serve_demo.py [PORT] [REBUILD_MIN]
View:  https://launcher-v10s5whg7i-codeserver.ide.taiji.woa.com/proxy/<PORT>/
"""
import http.server
import os
import re
import subprocess
import sys
import threading
import time

ROOT = "/srv/wm_teacher"
DEMO = f"{ROOT}/demo"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
REBUILD_MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 30
BUILDER = "/dockerdata/wm_tmp/build_demo.py"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=DEMO, **kw)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache" if self.path.endswith((".html", "/")) else "max-age=86400")
        super().end_headers()

    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        size = os.path.getsize(path)
        if not m:
            return super().send_head()
        start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2) or 0))
        end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, src, dst):
        n = getattr(self, "_remaining", None)
        if n is None:
            return super().copyfile(src, dst)
        while n > 0:
            buf = src.read(min(1 << 16, n))
            if not buf:
                break
            dst.write(buf)
            n -= len(buf)

    def log_message(self, *a):
        pass


def rebuild_loop():
    while True:
        time.sleep(REBUILD_MIN * 60)
        r = subprocess.run(["nice", "-n", "10", "/opt/conda/envs/torch-base/bin/python", BUILDER],
                           capture_output=True, text=True)
        print(time.strftime("%F %T"), (r.stdout or r.stderr).strip().splitlines()[-1:], flush=True)


threading.Thread(target=rebuild_loop, daemon=True).start()
print(f"serving {DEMO} on :{PORT}, rebuild every {REBUILD_MIN} min", flush=True)
http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
