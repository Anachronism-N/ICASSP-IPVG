import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

spec = importlib.util.spec_from_file_location(
    "upload_asset_backup", Path(__file__).resolve().parents[1] / "scripts/upload_asset_backup.py")
uploader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uploader)


class FakeHub:
    def __init__(self, private=True):
        self.private = private
        self.files = {}
        self.sha = "initial"
        self.commit_count = 0
        self.committed_paths = []
        self.fail_commit = False
        self.corrupt_index = False

    def info(self, repo):
        return SimpleNamespace(private=self.private, sha=self.sha)

    def paths(self, repo, names, revision):
        rows = []
        for name in names:
            if name not in self.files:
                continue
            data = self.files[name]
            digest = hashlib.sha256(data).hexdigest()
            lfs = SimpleNamespace(sha256=digest) if name.endswith((".tar", ".tar.gz")) else None
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + bytes([0]) + data).hexdigest()
            rows.append(SimpleNamespace(path=name, size=len(data), lfs=lfs, blob_id=blob))
        return rows

    def commit(self, repo, paths, parent, workers):
        self.commit_count += 1
        if self.fail_commit:
            e = RuntimeError("rate limited")
            e.response = SimpleNamespace(status_code=429)
            raise e
        if parent != self.sha:
            raise RuntimeError("parent conflict")
        for name, p in paths:
            self.files[name] = p.read_bytes()
            self.committed_paths.append(name)
        self.sha = "revision-" + str(self.commit_count)
        return self.sha

    def read_index(self, repo, name, revision, work):
        return b"corrupt" if self.corrupt_index else self.files[name]


class AssetBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        self.root.mkdir()
        self.work = self.base / "stage"
        self.config = self.base / "config.json"
        self.plan_path = self.base / "plan.json"
        self.grant = self.base / "authorization.json"
        self.output = io.StringIO()
        self.silence = contextlib.redirect_stdout(self.output)
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)

    def file(self, name, data=b"video-bytes"):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def plan(self, groups=None, **kwargs):
        if groups is None:
            self.file("videos/a.MP4")
            groups = [{"id": "videos", "paths": ["videos"]}]
        config = dict(root=str(self.root), repo_id="owner/dataset", visibility="private", groups=groups)
        config.update(kwargs)
        self.config.write_text(json.dumps(config))
        return uploader.build_plan(self.config, self.plan_path)

    def approve(self, groups=None):
        plan = uploader.load_plan(self.plan_path)
        document = self.base / "rights.txt"
        document.write_text("Unit-test authorization fixture, not a real dataset grant")
        grant = dict(approved=True, plan_sha256=uploader.sha_file(self.plan_path), repo_id=plan["repo_id"],
                     visibility=plan["visibility"], evidence=[dict(group=g, document=document.name,
                     sha256=uploader.sha_file(document)) for g in (groups or plan["groups"])])
        self.grant.write_text(json.dumps(grant))
        return grant

    def upload(self, hub=None):
        return uploader.upload(self.plan_path, self.work, self.grant, True, hub=hub or FakeHub())

    def test_default_upload_is_offline_and_does_not_create_work(self):
        self.plan()
        with mock.patch.object(uploader, "Hub", side_effect=AssertionError("network forbidden")):
            self.assertEqual(uploader.main(["upload", "--plan", str(self.plan_path), "--work-dir", str(self.work)]), 0)
        self.assertFalse(self.work.exists())

    def test_authorization_required_before_hub_access(self):
        self.plan()
        with mock.patch.object(uploader, "Hub", side_effect=AssertionError("network forbidden")):
            self.assertEqual(uploader.main(["upload", "--plan", str(self.plan_path), "--work-dir", str(self.work), "--execute"]), 2)
        self.assertFalse(self.work.exists())

    def test_changed_grant_document_rejected(self):
        self.plan()
        self.approve()
        (self.base / "rights.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "authorization document"):
            self.upload()

    def test_changed_plan_target_rejected(self):
        self.plan()
        self.approve()
        plan = uploader.read_json(self.plan_path)
        plan["repo_id"] = "different/repo"
        self.plan_path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "Authorization does not match"):
            self.upload()

    def test_missing_group_authorization_rejected(self):
        self.file("a")
        self.file("b")
        self.plan([{ "id": "a", "paths": ["a"]}, {"id": "b", "paths": ["b"]}])
        self.approve(["a"])
        with self.assertRaisesRegex(ValueError, "missing"):
            self.upload()

    def test_visibility_conflict_does_not_write(self):
        self.plan()
        self.approve()
        hub = FakeHub(private=False)
        with self.assertRaisesRegex(ValueError, "visibility differs"):
            self.upload(hub)
        self.assertEqual(hub.commit_count, 0)

    def test_uppercase_suffix_preserved(self):
        plan = self.plan()
        self.assertEqual(plan["shards"][0]["entries"][0]["path"], "videos/a.MP4")

    def test_source_paths_file_and_exclusion(self):
        self.file("a")
        self.file("b")
        (self.base / "paths.txt").write_text("a\nb\na\n")
        (self.base / "done.txt").write_text("b\n")
        plan = self.plan([dict(id="part", paths_files=["paths.txt"], exclude_paths_files=["done.txt"])])
        self.assertEqual(plan["selected_files"], 1)
        self.assertEqual(plan["excluded"][0]["path"], "b")

    def test_overlapping_groups_rejected(self):
        self.file("a")
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            self.plan([dict(id="one", paths=["a"]), dict(id="two", paths=["a"])])

    def test_missing_file_never_silently_skipped(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.plan([dict(id="one", paths=["absent.MP4"])])

    def test_traversal_and_symlink_rejected(self):
        for p in ("../outside", "/tmp/file", "a//b", "a\\b"):
            with self.subTest(p=p), self.assertRaises(ValueError):
                uploader.safe_name(p)
        target = self.file("real")
        (self.root / "link").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.plan([dict(id="one", paths=["link"])])

    def test_private_names_excluded_not_cookie_prompt(self):
        self.file("data/.env", b"secret")
        self.file("data/cookies_netscape.txt", b"secret")
        self.file("data/a chocolate cookies.MP4")
        plan = self.plan([dict(id="one", paths=["data"])])
        self.assertEqual(plan["selected_files"], 1)
        self.assertEqual(len(plan["excluded"]), 2)

    def test_changed_or_deleted_source_rejected(self):
        self.plan()
        self.approve()
        self.file("videos/a.MP4", b"different-video")
        hub = FakeHub()
        with self.assertRaisesRegex(ValueError, "changed"):
            self.upload(hub)
        self.assertEqual(hub.commit_count, 0)

    def test_secret_across_chunks_blocked(self):
        s = uploader.SecretScanner()
        s.check(b"h" + b"f_" + b"a" * 10)
        with self.assertRaisesRegex(ValueError, "Credential"):
            s.check(b"a" * 25)

    def test_credentials_block_commit_and_source_untouched(self):
        raw = b"h" + b"f_" + b"a" * 35
        p = self.file("data.txt", raw)
        self.plan([dict(id="one", paths=["data.txt"])])
        self.approve()
        hub = FakeHub()
        with self.assertRaisesRegex(ValueError, "Credential"):
            self.upload(hub)
        self.assertEqual(hub.commit_count, 0)
        self.assertEqual(p.read_bytes(), raw)

    def test_large_file_split_restores_exact_original(self):
        raw = bytes(range(256)) * 400
        self.file("big.MP4", raw)
        plan = self.plan([dict(id="one", paths=["big.MP4"])], shard_bytes=32768)
        parts = sorted([e for s in plan["shards"] for e in s["entries"]], key=lambda e: e["offset"])
        self.assertGreater(len(parts), 1)
        self.approve()
        hub = FakeHub()
        self.upload(hub)
        restored = bytearray()
        indices = []
        for s in plan["shards"]:
            archive, index, base = uploader.shard_files(plan, self.work, s)
            members = uploader.read_json(index)["members"]
            with tarfile.open(archive, "r:gz") as tf:
                for r in members:
                    data = tf.extractfile(r["member"]).read()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), r["sha256"])
                    indices.append((r["offset"], data))
        for offset, data in sorted(indices):
            self.assertEqual(offset, len(restored))
            restored.extend(data)
        self.assertEqual(restored, raw)

    def test_gzip_is_deterministic_between_work_directories(self):
        plan = self.plan()
        digest = uploader.sha_file(self.plan_path)
        self.work.mkdir()
        other = self.base / "other"
        other.mkdir()
        first = uploader.pack_shard(plan, digest, self.work, plan["shards"][0])
        second = uploader.pack_shard(plan, digest, other, plan["shards"][0])
        self.assertEqual(first["archive"], second["archive"])

    def test_success_verifies_lfs_and_git_index_and_resume(self):
        self.plan()
        self.approve()
        hub = FakeHub()
        self.upload(hub)
        self.assertEqual(hub.commit_count, 1)
        self.assertTrue(uploader.read_json(self.work / "result.json")["scope_complete"])
        self.assertFalse(uploader.read_json(self.work / "result.json")["whole_project_complete"])
        self.upload(hub)
        self.assertEqual(hub.commit_count, 1)
        self.assertEqual(len((self.work / "state.jsonl").read_text().splitlines()), 1)

    def test_conflicting_remote_never_overwritten(self):
        plan = self.plan()
        self.approve()
        hub = FakeHub()
        archive, _, _ = uploader.shard_files(plan, self.work, plan["shards"][0])
        name = plan["prefix"] + "/" + archive.name
        hub.files[name] = b"user-existing-content"
        with self.assertRaisesRegex(ValueError, "no overwrite"):
            self.upload(hub)
        self.assertEqual(hub.files[name], b"user-existing-content")
        self.assertEqual(hub.commit_count, 0)

    def test_missing_index_recovered_without_archive_reupload(self):
        plan = self.plan()
        self.approve()
        self.work.mkdir()
        uploader.pack_shard(plan, uploader.sha_file(self.plan_path), self.work, plan["shards"][0])
        archive, _, _ = uploader.shard_files(plan, self.work, plan["shards"][0])
        hub = FakeHub()
        name = plan["prefix"] + "/" + archive.name
        hub.files[name] = archive.read_bytes()
        self.upload(hub)
        self.assertNotIn(name, hub.committed_paths)
        self.assertEqual(len(hub.committed_paths), 1)

    def test_previously_verified_missing_remote_rejected(self):
        self.plan()
        self.approve()
        hub = FakeHub()
        self.upload(hub)
        del hub.files[next(n for n in hub.files if n.endswith(".tar.gz"))]
        with self.assertRaisesRegex(ValueError, "disappeared"):
            self.upload(hub)
        self.assertEqual(hub.commit_count, 1)

    def test_cached_artifact_corruption_rejected(self):
        plan = self.plan()
        self.approve()
        hub = FakeHub()
        self.upload(hub)
        archive, _, _ = uploader.shard_files(plan, self.work, plan["shards"][0])
        archive.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            self.upload(hub)

    def test_rate_error_propagates_without_retry_or_receipt(self):
        self.plan()
        self.approve()
        hub = FakeHub()
        hub.fail_commit = True
        with self.assertRaisesRegex(RuntimeError, "rate limited"):
            self.upload(hub)
        self.assertEqual(hub.commit_count, 1)
        self.assertFalse((self.work / "state.jsonl").exists())
        self.assertFalse((self.work / "result.json").exists())

    def test_bad_downloaded_index_never_records_success(self):
        self.plan()
        self.approve()
        hub = FakeHub()
        hub.corrupt_index = True
        with self.assertRaisesRegex(ValueError, "Downloaded"):
            self.upload(hub)
        self.assertFalse((self.work / "result.json").exists())

    def test_work_binding_prevents_another_plan(self):
        self.plan()
        self.approve()
        self.upload()
        modified = uploader.read_json(self.plan_path)
        modified["prefix"] = "different-prefix"
        self.plan_path.write_text(json.dumps(modified))
        self.approve()
        with self.assertRaisesRegex(ValueError, "another plan"):
            self.upload()

    def test_lock_blocks_duplicate_process(self):
        with uploader.work_lock(self.work), self.assertRaisesRegex(ValueError, "Another"):
            with uploader.work_lock(self.work):
                pass

    def test_plan_is_exclusive_not_overwritten(self):
        self.plan()
        with self.assertRaises(FileExistsError):
            self.plan()

    def archive(self, member, data=b"data", member_type=None):
        p = self.root / "bundle.tar.gz"
        with tarfile.open(p, "w:gz") as tf:
            item = tarfile.TarInfo(member)
            if member_type:
                item.type = member_type
                item.linkname = "target"
                tf.addfile(item)
            else:
                item.size = len(data)
                tf.addfile(item, io.BytesIO(data))
        return p

    def test_safe_existing_archive_scanned(self):
        p = self.archive("data/a.txt")
        uploader.scan_archive(p)

    def test_tar_traversal_symlink_and_nested_archive_rejected(self):
        for name, typ in (("../escape", None), ("link", tarfile.SYMTYPE), ("data/secrets.zip", None)):
            with self.subTest(name=name):
                p = self.archive(name, member_type=typ)
                with self.assertRaises(ValueError):
                    uploader.scan_archive(p)

    def test_secrets_in_preexisting_tar_blocked(self):
        p = self.archive("data/a.txt", b"h" + b"f_" + b"a" * 35)
        with self.assertRaisesRegex(ValueError, "Credential"):
            uploader.scan_archive(p)

    def test_zip_secret_and_traversal_blocked(self):
        p = self.root / "bundle.zip"
        for member, raw in (("../escape", b"data"), ("a", b"h" + b"f_" + b"a" * 35)):
            with zipfile.ZipFile(p, "w") as zf:
                zf.writestr(member, raw)
            with self.assertRaises(ValueError):
                uploader.scan_archive(p)

    def test_other_plan_cannot_overwrite_earlier_success_status(self):
        self.plan()
        self.approve()
        self.upload()
        before = (self.work / "status.json").read_bytes()
        plan = uploader.read_json(self.plan_path)
        plan["prefix"] = "new-prefix"
        self.plan_path.write_text(json.dumps(plan))
        self.approve()
        with self.assertRaisesRegex(ValueError, "another plan"):
            self.upload()
        self.assertEqual((self.work / "status.json").read_bytes(), before)

    def test_failed_resume_status_does_not_use_old_success(self):
        self.plan()
        self.approve()
        hub = FakeHub()
        self.upload(hub)
        del hub.files[next(n for n in hub.files if n.endswith(".tar.gz"))]
        with self.assertRaises(ValueError):
            self.upload(hub)
        self.assertEqual(uploader.main(["status", "--work-dir", str(self.work)]), 0)
        last = json.loads(self.output.getvalue().splitlines()[-1])
        self.assertFalse(last["scope_complete"])
        self.assertEqual(last["current"]["phase"], "blocked")

    def test_truncated_or_missing_split_ranges_rejected(self):
        self.file("large", b"a" * 100000)
        plan = self.plan([dict(id="one", paths=["large"])], shard_bytes=32768)
        plan["shards"].pop()
        self.plan_path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            uploader.load_plan(self.plan_path)

    def test_work_directory_inside_source_is_refused(self):
        self.plan()
        self.approve()
        self.work = self.root / "stage"
        hub = FakeHub()
        with self.assertRaisesRegex(ValueError, "outside source"):
            self.upload(hub)
        self.assertEqual(hub.commit_count, 0)

    def test_low_disk_space_refuses_packing_before_write(self):
        self.plan()
        self.approve()
        with mock.patch.object(uploader.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                self.upload()
        self.assertFalse(list(self.work.glob("*.partial")))

    def test_token_file_must_be_private_before_hub_access(self):
        self.plan()
        self.approve()
        p = self.base / "token"
        p.write_text("test fixture not a live token")
        p.chmod(0o644)
        with mock.patch.object(uploader, "Hub", side_effect=AssertionError("network forbidden")):
            self.assertEqual(uploader.main(["upload", "--plan", str(self.plan_path), "--work-dir", str(self.work),
                "--execute", "--authorization", str(self.grant), "--token-file", str(p)]), 2)

    def test_boolean_shard_size_rejected(self):
        with self.assertRaisesRegex(ValueError, "shard byte"):
            self.plan(shard_bytes=True)

    def test_uncompressed_archive_mode(self):
        plan = self.plan(compression="none")
        self.approve()
        hub = FakeHub()
        self.upload(hub)
        self.assertTrue(any(name.endswith(".tar") for name in hub.files))
        archive, _, _ = uploader.shard_files(plan, self.work, plan["shards"][0])
        with tarfile.open(archive) as tf:
            self.assertEqual(tf.extractfile(tf.getmembers()[0]).read(), b"video-bytes")

    def test_renamed_cookie_content_blocked(self):
        p = self.file("random.txt", b"# Netscape HTTP Cookie File\n")
        self.plan([dict(id="one", paths=["random.txt"])])
        self.approve()
        hub = FakeHub()
        with self.assertRaisesRegex(ValueError, "Credential"):
            self.upload(hub)
        self.assertEqual(hub.commit_count, 0)

    def test_deleted_source_not_successful(self):
        self.plan()
        self.approve()
        (self.root / "videos/a.MP4").unlink()
        hub = FakeHub()
        with self.assertRaises(FileNotFoundError):
            self.upload(hub)
        self.assertFalse((self.work / "result.json").exists())

    def test_error_sanitization(self):
        token = "h" + "f_" + "a" * 35
        value = uploader.safe_error(RuntimeError("https://example.invalid/?token=" + token + " " + token))
        self.assertNotIn(token, value)
        self.assertNotIn("https://", value)


if __name__ == "__main__":
    unittest.main()
