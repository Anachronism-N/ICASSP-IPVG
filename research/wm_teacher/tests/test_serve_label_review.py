import hashlib
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import serve_label_review as srv

TOKEN = 'fixture' + '-access-value'  # fake test value, not a credential


def sha(b):
    return hashlib.sha256(b).hexdigest()


class Fixture:
    def __init__(self, tmp):
        self.root = Path(tmp)
        (self.root / 'videos').mkdir()
        (self.root / 'pilot').mkdir()
        self.video = bytes(range(256)) * 40
        (self.root / 'videos/a.mp4').write_bytes(self.video)
        self.prompt = 'A <b>person</b> & "cup"'
        pages = []
        for i, (s, e) in enumerate([(0, 59), (60, 120)]):
            p = self.root / 'pilot' / ('case_one_%d.jpg' % i)
            p.write_bytes(b'jpeg%d' % i)
            pages.append({'path': str(p), 'start_frame': s, 'end_frame_inclusive': e, 'sha256': sha(p.read_bytes())})
        self.index = self.root / 'pilot/index.json'
        self.index.write_text(json.dumps({'cases': [{'audit_id': 'case_one', 'dataset': 'demo',
                                                      'video_sha256': sha(self.video), 'pages': pages}]}))
        (self.root / 'pilot/lead_full17_review_v1.json').write_text(json.dumps({'secret': 'ASSISTANT_VERDICT_XYZ'}))
        self.audit = self.root / 'audit.json'
        self.audit.write_text(json.dumps({'cases': [{'audit_id': 'case_one', 'dataset': 'demo', 'video': 'videos/a.mp4',
                                                     'prompt': self.prompt, 'prompt_sha256': sha(self.prompt.encode())}]}))


class ReviewServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fx = Fixture(self.tmp.name)
        self.cases = srv.load_cases(self.fx.root, self.fx.index, self.fx.audit)
        self.out = self.fx.root / 'out'
        self.store = srv.Store(self.out)
        self.server = srv.make_server(self.cases, self.store, TOKEN, '127.0.0.1', 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]

    def request(self, method, path, body=None, headers=None, cookie=True):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        h = dict(headers or {})
        if cookie:
            h['Cookie'] = srv.COOKIE + '=' + TOKEN
        conn.request(method, path, body=body, headers=h)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r, data

    def form(self, **override):
        values = {'reviewer': 'tester', 'objects': 'supported', 'action': 'contradicted', 'camera': 'supported',
                  'terminal_state': 'unverifiable', 'overall': 'needs_fix', 'corrected_caption': 'fixed', 'free_notes': ''}
        values.update(override)
        return urlencode(values).encode()

    def post(self, body):
        return self.request('POST', '/case/case_one', body, {'Content-Type': 'application/x-www-form-urlencoded'})

    def test_token_required_everywhere(self):
        for method, path in [('GET', '/'), ('GET', '/case/case_one'), ('GET', '/media/case_one/video'), ('POST', '/case/case_one')]:
            r, _ = self.request(method, path, b'' if method == 'POST' else None, cookie=False)
            self.assertEqual(r.status, 403, path)
        r, _ = self.request('GET', '/?token=wrong', cookie=False)
        self.assertEqual(r.status, 403)
        r, _ = self.request('GET', '/case/case_one?token=' + TOKEN, cookie=False)
        self.assertEqual(r.status, 303)
        self.assertIn('HttpOnly', r.getheader('Set-Cookie'))
        self.assertEqual(r.getheader('Location'), '/case/case_one')

    def test_token_redacted_from_access_log(self):
        import io
        from unittest import mock
        buf = io.StringIO()
        with mock.patch.object(srv.sys, 'stderr', buf):
            self.request('GET', '/?token=' + TOKEN, cookie=False)
        self.assertNotIn(TOKEN, buf.getvalue())
        self.assertIn('token=REDACTED', buf.getvalue())

    def test_blind_escaped_and_whitelisted(self):
        r, body = self.request('GET', '/case/case_one')
        self.assertEqual(r.status, 200)
        text = body.decode()
        self.assertNotIn('ASSISTANT_VERDICT_XYZ', text)
        self.assertNotIn('<b>person</b>', text)
        self.assertIn('&lt;b&gt;person&lt;/b&gt; &amp; &quot;cup&quot;', text)
        self.assertEqual(r.getheader('Cache-Control'), 'no-store')
        for path in ['/media/../audit.json', '/media/case_one/page/2', '/media/other/video', '/case/other',
                     '/pilot/lead_full17_review_v1.json', '/media/case_one/%2e%2e', '/index.json']:
            r, data = self.request('GET', path)
            self.assertEqual(r.status, 404, path)
            self.assertNotIn(b'ASSISTANT_VERDICT_XYZ', data)

    def test_video_range_and_pages(self):
        r, data = self.request('GET', '/media/case_one/video')
        self.assertEqual((r.status, data), (200, self.fx.video))
        r, data = self.request('GET', '/media/case_one/video', headers={'Range': 'bytes=100-199'})
        self.assertEqual(r.status, 206)
        self.assertEqual(data, self.fx.video[100:200])
        self.assertEqual(r.getheader('Content-Range'), 'bytes 100-199/%d' % len(self.fx.video))
        r, data = self.request('GET', '/media/case_one/video', headers={'Range': 'bytes=-10'})
        self.assertEqual((r.status, data), (206, self.fx.video[-10:]))
        r, _ = self.request('GET', '/media/case_one/video', headers={'Range': 'bytes=999999-'})
        self.assertEqual(r.status, 416)
        r, data = self.request('GET', '/media/case_one/page/1')
        self.assertEqual((r.status, data, r.getheader('Content-Type')), (200, b'jpeg1', 'image/jpeg'))

    def test_submit_update_keeps_history(self):
        r, _ = self.post(self.form())
        self.assertEqual(r.status, 303)
        r, _ = self.post(self.form(overall='reject', free_notes='second'))
        self.assertEqual(r.status, 303)
        latest = json.loads((self.out / 'latest.json').read_text())['case_one']
        self.assertEqual(latest['overall'], 'reject')
        self.assertEqual(latest['action'], {'verdict': 'contradicted', 'note': ''})
        self.assertEqual(latest['prompt_sha256'], sha(self.fx.prompt.encode()))
        self.assertEqual(latest['reviewer_type'], 'human')
        history = [json.loads(x) for x in (self.out / 'history.jsonl').read_text().splitlines()]
        self.assertEqual([h['overall'] for h in history], ['needs_fix', 'reject'])
        for f in ['latest.json', 'history.jsonl']:
            self.assertEqual((self.out / f).stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.out.stat().st_mode & 0o777, 0o700)
        reopened = srv.Store(self.out)
        self.assertEqual(reopened.get('case_one')['overall'], 'reject')
        r, body = self.request('GET', '/case/case_one?saved=1')
        self.assertIn('已保存', body.decode())
        r, body = self.request('GET', '/')
        self.assertIn('已完成 1 / 1', body.decode())

    def test_invalid_submissions_rejected_and_not_saved(self):
        bad = [self.form(objects='maybe'), self.form(overall='great'), self.form(reviewer=''),
               self.form(free_notes='x' * (srv.MAX_TEXT + 1)), self.form(extra='1'),
               self.form() + b'&objects=supported']
        for body in bad:
            r, _ = self.post(body)
            self.assertEqual(r.status, 400)
        r, _ = self.request('POST', '/case/case_one', b'x' * (srv.MAX_BODY + 1), {'Content-Type': 'application/x-www-form-urlencoded'})
        self.assertEqual(r.status, 413)
        r, _ = self.request('POST', '/case/case_one', self.form(), {'Content-Type': 'application/json'})
        self.assertEqual(r.status, 415)
        r, _ = self.request('POST', '/case/other', self.form(), {'Content-Type': 'application/x-www-form-urlencoded'})
        self.assertEqual(r.status, 404)
        self.assertFalse((self.out / 'history.jsonl').exists())


class InputVerificationTests(unittest.TestCase):
    def test_hash_mismatch_refuses_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            (fx.root / 'videos/a.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'video'):
                srv.load_cases(fx.root, fx.index, fx.audit)
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            audit = json.loads(fx.audit.read_text())
            audit['cases'][0]['prompt'] = 'edited caption'
            fx.audit.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, 'caption'):
                srv.load_cases(fx.root, fx.index, fx.audit)
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            Path(json.loads(fx.index.read_text())['cases'][0]['pages'][1]['path']).write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'contact sheet'):
                srv.load_cases(fx.root, fx.index, fx.audit)

    def test_check_only_does_not_listen_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            out = fx.root / 'never'
            self.assertEqual(srv.main(['--root', str(fx.root), '--index', str(fx.index), '--audit', str(fx.audit),
                                       '--out-dir', str(out), '--check-only']), 0)
            self.assertFalse(out.exists())


class BasePathTests(ReviewServerTests.__mro__[1]):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fx = Fixture(self.tmp.name)
        cases = srv.load_cases(self.fx.root, self.fx.index, self.fx.audit)
        self.out = self.fx.root / 'out'
        self.server = srv.make_server(cases, srv.Store(self.out), TOKEN, '127.0.0.1', 0, '/proxy/8791/')
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]

    request = ReviewServerTests.request

    def test_prefixed_links_redirects_cookie_and_stripped_proxy(self):
        r, _ = self.request('GET', '/proxy/8791/?token=' + TOKEN, cookie=False)
        self.assertEqual(r.status, 303)
        self.assertEqual(r.getheader('Location'), '/proxy/8791/')
        self.assertIn('Path=/proxy/8791/', r.getheader('Set-Cookie'))
        for prefix in ['/proxy/8791', '']:  # absproxy keeps the prefix, proxy strips it
            r, body = self.request('GET', prefix + '/case/case_one')
            self.assertEqual(r.status, 200, prefix)
            text = body.decode()
            for needle in ['href="/proxy/8791/"', 'src="/proxy/8791/media/case_one/video"',
                           'action="/proxy/8791/case/case_one"', 'src="/proxy/8791/media/case_one/page/0"']:
                self.assertIn(needle, text)
            self.assertNotIn('href="/case/', text)
            r, data = self.request('GET', prefix + '/media/case_one/video', headers={'Range': 'bytes=0-9'})
            self.assertEqual((r.status, data), (206, self.fx.video[:10]))
        r, body = self.request('GET', '/proxy/8791/')
        self.assertIn('href="/proxy/8791/case/case_one"', body.decode())
        values = {'reviewer': 'tester', 'objects': 'supported', 'action': 'supported', 'camera': 'supported',
                  'terminal_state': 'supported', 'overall': 'usable'}
        for prefix, location in [('', '/case/case_one?saved=1'), ('/proxy/8791', '/proxy/8791/case/case_one?saved=1')]:
            r, _ = self.request('POST', prefix + '/case/case_one', urlencode(values).encode(),
                                {'Content-Type': 'application/x-www-form-urlencoded'})
            self.assertEqual((r.status, r.getheader('Location')), (303, location))
        r, _ = self.request('GET', '/?token=' + TOKEN, cookie=False)
        self.assertEqual(r.getheader('Location'), '/')  # stripped proxy re-prefixes Location itself
        r, _ = self.request('GET', '/proxy/8791/case/case_one', cookie=False)
        self.assertEqual(r.status, 403)

    def test_invalid_base_rejected(self):
        with self.assertRaises(ValueError):
            srv.make_server([], None, TOKEN, '127.0.0.1', 0, '/bad path"')


if __name__ == '__main__':
    unittest.main()
