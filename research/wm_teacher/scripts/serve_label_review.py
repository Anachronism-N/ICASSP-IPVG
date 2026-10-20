#!/usr/bin/env python3
"""Blind human caption-label review web service (stdlib only).

Serves only the whitelisted pilot cases (original MP4 + two contact sheets + original caption)
and records the human reviewer's verdicts privately. Never reads or shows assistant verdicts.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import secrets
import sys
import threading
import time
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'eval/diag/teacher_label_audit_L1_decode_20261019T152827Z_125305/fullframe_semantic_pilot'
DEFAULT_INDEX = PILOT / 'index.json'
DEFAULT_AUDIT = ROOT / 'eval/diag/teacher_label_audit_L1_20261019.json'
DEFAULT_OUT = ROOT / 'eval/diag/human_label_review_20261020'
CATEGORIES = [('objects', '物体/场景'), ('action', '动作'), ('camera', '镜头运动'), ('terminal_state', '结束状态')]
VERDICTS = [('supported', '相符'), ('contradicted', '矛盾'), ('unverifiable', '无法判断')]
OVERALL = [('usable', '可直接用于训练'), ('needs_fix', '需修改后使用'), ('reject', '不应使用')]
MAX_TEXT = 8000
MAX_BODY = 64 * 1024
COOKIE = 'label_review_token'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_cases(root, index_path, audit_path):
    index = json.loads(Path(index_path).read_text())
    audit = {c['audit_id']: c for c in json.loads(Path(audit_path).read_text())['cases']}
    cases, errors = [], []
    for entry in index['cases']:
        cid = entry['audit_id']
        if not re.fullmatch(r'[A-Za-z0-9_]+', cid) or cid not in audit:
            errors.append(cid + ': unknown or invalid id')
            continue
        row = audit[cid]
        prompt = row['prompt']
        if hashlib.sha256(prompt.encode()).hexdigest() != row['prompt_sha256']:
            errors.append(cid + ': caption hash mismatch')
        video = (Path(root) / row['video']).resolve()
        if not video.is_file() or sha256(video) != entry['video_sha256']:
            errors.append(cid + ': video missing or hash mismatch')
        pages = []
        for page in entry['pages']:
            p = Path(page['path'])
            if not p.is_file() or sha256(p) != page['sha256']:
                errors.append(cid + ': contact sheet missing or hash mismatch')
            pages.append({'path': p, 'start': int(page['start_frame']), 'end': int(page['end_frame_inclusive'])})
        cases.append({'audit_id': cid, 'dataset': entry.get('dataset', row.get('dataset', '')), 'prompt': prompt,
                      'prompt_sha256': row['prompt_sha256'], 'video': video, 'video_sha256': entry['video_sha256'],
                      'pages': pages})
    if errors:
        raise ValueError('Input verification failed: ' + '; '.join(errors))
    if not cases:
        raise ValueError('No review cases')
    return cases


class Store:
    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        self.lock = threading.Lock()
        self.latest_path = self.dir / 'latest.json'
        self.history_path = self.dir / 'history.jsonl'
        self.latest = json.loads(self.latest_path.read_text()) if self.latest_path.exists() else {}

    def _atomic(self, path, data):
        tmp = path.with_name(path.name + '.tmp%d' % os.getpid())
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def save(self, record):
        with self.lock:
            fd = os.open(self.history_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, 'a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            self.latest[record['audit_id']] = record
            self._atomic(self.latest_path, json.dumps(self.latest, ensure_ascii=False, indent=2) + '\n')

    def get(self, cid):
        with self.lock:
            return self.latest.get(cid)


def validate_form(form, case):
    def one(key, default=''):
        values = form.get(key, [default])
        if len(values) != 1:
            raise ValueError('duplicate field ' + key)
        value = values[0]
        if len(value) > MAX_TEXT:
            raise ValueError('field too long ' + key)
        return value
    reviewer = one('reviewer').strip()
    if not reviewer or len(reviewer) > 200:
        raise ValueError('reviewer name required (<=200 characters)')
    record = {'audit_id': case['audit_id'], 'prompt_sha256': case['prompt_sha256'],
              'video_sha256': case['video_sha256'], 'reviewer': reviewer, 'reviewer_type': 'human',
              'submitted_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    allowed = {v for v, _ in VERDICTS}
    for key, _ in CATEGORIES:
        verdict = one(key)
        if verdict not in allowed:
            raise ValueError('invalid verdict for ' + key)
        record[key] = {'verdict': verdict, 'note': one(key + '_note').strip()}
    overall = one('overall')
    if overall not in {v for v, _ in OVERALL}:
        raise ValueError('invalid overall decision')
    record['overall'] = overall
    record['corrected_caption'] = one('corrected_caption').strip()
    record['free_notes'] = one('free_notes').strip()
    unknown = set(form) - {'reviewer', 'overall', 'corrected_caption', 'free_notes'} - {k for k, _ in CATEGORIES} \
        - {k + '_note' for k, _ in CATEGORIES}
    if unknown:
        raise ValueError('unknown fields')
    return record


STYLE = ('body{font-family:system-ui,"PingFang SC","Microsoft YaHei",sans-serif;margin:0 auto;max-width:1400px;padding:16px;'
         'color:#1d232a;background:#f6f7f9}a{color:#1559b3}.card{background:#fff;border:1px solid #dde1e6;border-radius:8px;'
         'padding:14px;margin:12px 0}.caption{white-space:pre-wrap;line-height:1.6;font-size:15px}table{border-collapse:collapse;'
         'width:100%}td,th{border-bottom:1px solid #e3e6ea;padding:6px 8px;text-align:left}.done{color:#18794e;font-weight:600}'
         '.todo{color:#b54708}img.sheet{width:100%;border:1px solid #ccd;cursor:zoom-in}video{width:100%;max-height:70vh;'
         'background:#000}fieldset{border:1px solid #dde1e6;border-radius:6px;margin:8px 0}textarea,input[type=text]{width:100%;'
         'box-sizing:border-box;font:inherit}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.nav{display:flex;'
         'justify-content:space-between}.err{background:#fde8e8;color:#9b1c1c;padding:8px;border-radius:6px}.ok{background:#e6f6ec;'
         'color:#18794e;padding:8px;border-radius:6px}button{font-size:16px;padding:8px 22px}')


def page(title, body):
    return ('<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>%s</title><style>%s</style></head><body>%s</body></html>' % (html.escape(title), STYLE, body)).encode()


def index_html(cases, store):
    done = sum(1 for c in cases if store.get(c['audit_id']))
    rows = []
    for i, c in enumerate(cases, 1):
        rec = store.get(c['audit_id'])
        state = ('<span class="done">已提交 %s</span>' % html.escape(rec['submitted_at'])) if rec else '<span class="todo">待审</span>'
        rows.append('<tr><td>%d</td><td><a href="/case/%s">%s</a></td><td>%s</td><td>%s</td></tr>' % (
            i, html.escape(c['audit_id']), html.escape(c['audit_id']), html.escape(c['dataset']), state))
    body = ('<h1>训练caption人工审阅</h1><div class="card">请观看每个5秒原视频（121帧），判断训练caption的描述是否与画面一致。'
            '页面不展示任何助手结论；你的提交为独立人工审阅，可随时修改，历史均保留。已完成 %d / %d。</div>'
            '<div class="card"><table><tr><th>#</th><th>案例</th><th>数据集</th><th>状态</th></tr>%s</table></div>'
            % (done, len(cases), ''.join(rows)))
    return page('caption人工审阅', body)


def case_html(cases, store, pos, message='', error=''):
    c = cases[pos]
    cid = html.escape(c['audit_id'])
    rec = store.get(c['audit_id']) or {}
    prev_link = '<a href="/case/%s">← 上一例</a>' % html.escape(cases[pos - 1]['audit_id']) if pos > 0 else '<span></span>'
    next_link = '<a href="/case/%s">下一例 →</a>' % html.escape(cases[pos + 1]['audit_id']) if pos + 1 < len(cases) else '<span></span>'
    sheets = ''.join('<div><div>帧 %d–%d</div><a href="/media/%s/page/%d" target="_blank"><img class="sheet" loading="lazy" '
                     'src="/media/%s/page/%d" alt="contact sheet"></a></div>' % (p['start'], p['end'], cid, i, cid, i)
                     for i, p in enumerate(c['pages']))
    fields = []
    for key, label in CATEGORIES:
        old = rec.get(key, {})
        radios = ''.join('<label><input type="radio" name="%s" value="%s" required%s> %s</label> ' % (
            key, v, ' checked' if old.get('verdict') == v else '', t) for v, t in VERDICTS)
        fields.append('<fieldset><legend><b>%s</b></legend>%s<input type="text" name="%s_note" maxlength="%d" '
                      'placeholder="可选：具体说明（帧范围/差异）" value="%s"></fieldset>' % (
                          label, radios, key, MAX_TEXT, html.escape(old.get('note', ''))))
    overall = ''.join('<label><input type="radio" name="overall" value="%s" required%s> %s</label> ' % (
        v, ' checked' if rec.get('overall') == v else '', t) for v, t in OVERALL)
    notice = ('<div class="ok">%s</div>' % html.escape(message) if message else '') + \
             ('<div class="err">%s</div>' % html.escape(error) if error else '')
    body = ('<div class="nav"><a href="/">返回列表</a>%s%s</div><h2>%d / %d · %s <small>(%s)</small></h2>%s'
            '<div class="grid"><div class="card"><video controls preload="metadata" src="/media/%s/video"></video></div>'
            '<div class="card"><b>训练caption（原文）</b><div class="caption">%s</div></div></div>'
            '<div class="card grid">%s</div>'
            '<form class="card" method="post" action="/case/%s">%s<fieldset><legend><b>总体结论</b></legend>%s</fieldset>'
            '<label>修正后的caption（可选）<textarea name="corrected_caption" rows="5" maxlength="%d">%s</textarea></label>'
            '<label>其他备注（可选）<textarea name="free_notes" rows="3" maxlength="%d">%s</textarea></label>'
            '<label>审阅人<input type="text" name="reviewer" required maxlength="200" value="%s"></label>'
            '<p><button type="submit">提交 / 更新</button></p></form>' % (
                prev_link, next_link, pos + 1, len(cases), cid, html.escape(c['dataset']), notice, cid,
                html.escape(c['prompt']), sheets, cid, ''.join(fields), overall, MAX_TEXT,
                html.escape(rec.get('corrected_caption', '')), MAX_TEXT, html.escape(rec.get('free_notes', '')),
                html.escape(rec.get('reviewer', ''))))
    return page(c['audit_id'] + ' 审阅', body)


def make_server(cases, store, token, host='127.0.0.1', port=8791):
    positions = {c['audit_id']: i for i, c in enumerate(cases)}

    class Handler(BaseHTTPRequestHandler):
        server_version = 'LabelReview/1'
        sys_version = ''

        def log_message(self, fmt, *args):
            line = re.sub(r'token=[^&\s"]*', 'token=REDACTED', fmt % args)
            sys.stderr.write('%s %s\n' % (self.log_date_time_string(), line))

        def _authorized(self):
            query = parse_qs(urlsplit(self.path).query)
            supplied = query.get('token', [None])[0]
            if supplied is not None and secrets.compare_digest(supplied, token):
                return 'query'
            jar = cookies.SimpleCookie(self.headers.get('Cookie', ''))
            if COOKIE in jar and secrets.compare_digest(jar[COOKIE].value, token):
                return 'cookie'
            return None

        def _send(self, status, body=b'', ctype='text/html; charset=utf-8', extra=None):
            self.send_response(status)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

        def _deny(self):
            self._send(403, page('拒绝访问', '<p>需要有效的访问令牌。</p>'))

        def _file(self, path, ctype):
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            rng = self.headers.get('Range')
            if rng:
                m = re.fullmatch(r'bytes=(\d*)-(\d*)', rng.strip())
                if not m or (m.group(1) == '' and m.group(2) == ''):
                    return self._send(416, b'', extra={'Content-Range': 'bytes */%d' % size})
                if m.group(1) == '':
                    start = max(0, size - int(m.group(2)))
                else:
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                if start > end or start >= size:
                    return self._send(416, b'', extra={'Content-Range': 'bytes */%d' % size})
                status = 206
            length = end - start + 1
            self.send_response(status)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(length))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if status == 206:
                self.send_header('Content-Range', 'bytes %d-%d/%d' % (start, end, size))
            self.end_headers()
            if self.command == 'HEAD':
                return
            with open(path, 'rb') as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            auth = self._authorized()
            if not auth:
                return self._deny()
            route = urlsplit(self.path).path
            if auth == 'query':
                cookie = '%s=%s; HttpOnly; SameSite=Strict; Path=/' % (COOKIE, token)
                return self._send(303, b'', extra={'Location': route or '/', 'Set-Cookie': cookie})
            if route == '/':
                return self._send(200, index_html(cases, store))
            m = re.fullmatch(r'/case/([A-Za-z0-9_]+)', route)
            if m and m.group(1) in positions:
                saved = 'saved' in parse_qs(urlsplit(self.path).query)
                return self._send(200, case_html(cases, store, positions[m.group(1)], '已保存。' if saved else ''))
            m = re.fullmatch(r'/media/([A-Za-z0-9_]+)/(video|page/([01]))', route)
            if m and m.group(1) in positions:
                case = cases[positions[m.group(1)]]
                if m.group(2) == 'video':
                    return self._file(case['video'], 'video/mp4')
                index = int(m.group(3))
                if index < len(case['pages']):
                    return self._file(case['pages'][index]['path'], 'image/jpeg')
            return self._send(404, page('未找到', '<p>未找到。</p>'))

        def do_POST(self):
            if not self._authorized():
                return self._deny()
            route = urlsplit(self.path).path
            m = re.fullmatch(r'/case/([A-Za-z0-9_]+)', route)
            if not m or m.group(1) not in positions:
                return self._send(404, page('未找到', '<p>未找到。</p>'))
            try:
                length = int(self.headers.get('Content-Length', ''))
            except ValueError:
                return self._send(411, page('错误', '<p>需要Content-Length。</p>'))
            if length < 0 or length > MAX_BODY:
                return self._send(413, page('错误', '<p>提交内容过大。</p>'))
            if not self.headers.get('Content-Type', '').startswith('application/x-www-form-urlencoded'):
                return self._send(415, page('错误', '<p>不支持的提交格式。</p>'))
            pos = positions[m.group(1)]
            try:
                form = parse_qs(self.rfile.read(length).decode('utf-8'), keep_blank_values=True, strict_parsing=False)
                record = validate_form(form, cases[pos])
            except (ValueError, UnicodeDecodeError) as exc:
                return self._send(400, case_html(cases, store, pos, error='提交未保存：' + str(exc)))
            store.save(record)
            self._send(303, b'', extra={'Location': '/case/%s?saved=1' % cases[pos]['audit_id']})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8791)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    cases = load_cases(args.root, args.index, args.audit)
    print('INPUT_VERIFIED', len(cases), 'cases (video/caption/contact-sheet SHA256)', flush=True)
    if args.check_only:
        return 0
    store = Store(args.out_dir)
    token = secrets.token_urlsafe(24)
    token_path = args.out_dir / 'access_token'
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(token + '\n')
    server = make_server(cases, store, token, args.host, args.port)
    print('SERVING http://%s:%d/?token=%s' % (args.host, server.server_address[1], token), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
