"""CPU-only mocks and real GNU timeout/gpu_hold.exec; no CUDA or model execution."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_teacher_devset_node as node


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "tests").mkdir()
        self.folder = self.root / "eval/teacher_alg_devset_20261026"
        self.folder.mkdir(parents=True, mode=0o700)
        self.path = self.folder / "plan.json"
        self.state = self.root / "state"
        self.state.mkdir()
        self.digest = "a" * 64
        for name in ("scripts/run_teacher_devset_node.py", "tests/test_teacher_devset_node.py"):
            (self.root / name).write_bytes((ROOT / name).read_bytes())
        self.pf = self.root / "pf.json"
        self.pf.write_bytes(node.encoded(dict(rank=1, hostname=os.uname().nodename)))
        self.pf_sha = node.sha(self.pf.read_bytes())
        self.plan = dict(schema="teacher_devset_plan_v1", method="alg", root=str(self.root),
                         output_root=str(self.folder), nshards=16, shard_cap_seconds=[5400] * 16, source_files=[],
                         admission=dict(charged_or_reserved_seconds=100), runtime=dict(
                             python=sys.executable, tools=dict(timeout="/usr/bin/timeout")),
                         jobs=[dict(index=i, output=str(self.folder / f"video{i}.mp4")) for i in range(188)])
        self.path.write_bytes(node.encoded(self.plan))
        self.digest = node.sha(self.path.read_bytes())
        self.patches = []
        self.patch(node.dev, "verify_plan", return_value=(self.plan, [{}] * 9))
        self.patch(node, "holder_pid", return_value=987654)
        self.patch(node.dev.smoke, "supervisor_scope", side_effect=lambda state, pid, gpu: f"GPU-{gpu}")
        self.patch(node.dev, "strict_reuse", return_value=None)
        self.source_sha = node.source_digest(node.source_records(self.root))

    def patch(self, obj, name, **kwargs):
        p = mock.patch.object(obj, name, **kwargs)
        value = p.start()
        self.addCleanup(p.stop)
        return value

    def reserve(self, rank=1):
        self.pf.write_bytes(node.encoded(dict(rank=rank, hostname=os.uname().nodename)))
        return node.reserve("alg", self.path, self.digest, self.pf, node.sha(self.pf.read_bytes()), rank,
                            node.RANKS["alg"][rank], self.state, self.source_sha)

    def ready(self):
        self.reserve(2)
        self.reserve(1)
        self.pf_sha = node.sha(self.pf.read_bytes())

    def ledger(self):
        return json.loads(node.ledger_file(self.path).read_bytes())

    def save(self, ledger):
        node.dev.checkpoint(node.ledger_file(self.path), ledger)

    def preflight(self):
        return node.launch_preflight("alg", self.path, self.digest, self.pf, self.pf_sha, 1, 0,
                                     self.state, self.source_sha)

    def start_item(self, shard=0, elapsed=2):
        ledger = self.ledger()
        item = ledger["allocations"][shard]
        item.update(status="running", start=time.time() - elapsed, allocation_start_monotonic=time.monotonic() - elapsed,
                    node_supervisor_pid=os.getpid(), gpu_uuid=f"GPU-{shard % 8}")
        self.save(ledger)
        return item

    def worker(self, shard=0, success=True):
        receipt = dict(schema="teacher_devset_shard_receipt_v1", method="alg", plan_sha256=self.digest,
                       shard=shard, nshards=16, success=success)
        path = self.folder / f"shard{shard}.finished.json"
        node.dev.exclusive(path, receipt)
        ledger = self.ledger()
        ledger["allocations"][shard].update(worker_complete=True, worker_success=success,
                                             receipt_sha256=node.sha(path.read_bytes()))
        self.save(ledger)

    def finish(self, shard=0, **kwargs):
        return node.finish_allocation(self.path, self.digest, shard, os.getpid(), self.state,
                                      kwargs.get("rc", 0), kwargs.get("timeout", False),
                                      kwargs.get("interrupted"), kwargs.get("error"))

    def args(self):
        return argparse.Namespace(method="alg", plan=self.path, approved_plan_sha256=self.digest,
                                  node_preflight=self.pf, approved_node_preflight_sha256=self.pf_sha,
                                  rank=1, shard_offset=0, state_dir=self.state, approved_new_source_sha256=self.source_sha,
                                  gpu_authorized=True)


class ReservationTests(Fixture):
    def test_first_reservation_has_all_sixteen_caps(self):
        result = self.reserve()
        ledger = self.ledger()
        self.assertEqual(len(ledger["allocations"]), 16)
        self.assertEqual(sum(a["cap_seconds"] for a in ledger["allocations"]), 86400)
        self.assertEqual(sum(a["reservation_ready"] for a in ledger["allocations"]), 8)
        self.assertTrue(all(a["status"] == "reserved" for a in ledger["allocations"]))
        self.assertEqual(result["reserved_generation_GPUh"], 24)
        self.assertFalse(result["GPU_used"])

    def test_missing_peer_blocks_without_gpu_pause_or_clock(self):
        self.reserve()
        with self.assertRaisesRegex(ValueError, "Both nodes"):
            self.preflight()
        self.assertTrue(all("start" not in a for a in self.ledger()["allocations"]))
        self.assertFalse(list(self.state.glob("pause.*")))

    def test_two_reservations_ready_eight_local(self):
        self.ready()
        _, own, uuids = self.preflight()
        self.assertEqual([a["shard"] for a in own], list(range(8)))
        self.assertEqual(set(uuids.values()), {f"GPU-{g}" for g in range(8)})
        self.assertEqual({a["supervisor_pid"] for a in own}, {987654})

    def test_old_ledger_not_reset(self):
        self.reserve()
        before = node.ledger_file(self.path).read_bytes()
        with self.assertRaisesRegex(ValueError, "already reserved"):
            self.reserve()
        self.assertEqual(before, node.ledger_file(self.path).read_bytes())

    def test_wrong_method_rank_offset(self):
        for method, rank, offset in (("alg", 0, 0), ("alg", 1, 8), ("dymos", 2, 0), ("dymos", 0, 8)):
            with self.subTest(method=method, rank=rank):
                with self.assertRaisesRegex(ValueError, "Wrong method"):
                    node.verify_inputs(method, self.path, self.digest, self.pf, self.pf_sha, rank, offset)

    def test_wrong_shards_caps_jobs(self):
        for field, value in (("nshards", 8), ("shard_cap_seconds", [5401] * 16), ("jobs", [])):
            old = self.plan[field]
            self.plan[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "fixed 188"):
                self.reserve()
            self.plan[field] = old

    def test_wrong_host_preflight(self):
        self.ready()
        ledger = self.ledger()
        ledger["allocations"][0]["hostname"] = "not-this-host"
        self.save(ledger)
        with self.assertRaises(ValueError):
            self.preflight()

    def test_duplicate_missing_shard_bad_cap_rejected(self):
        self.ready()
        good = self.ledger()
        for field, value in (("shard", 1), ("nshards", 8), ("cap_seconds", 5401), ("gpu_index", 7)):
            ledger = copy.deepcopy(good)
            ledger["allocations"][0][field] = value
            self.save(ledger)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.preflight()

    def test_finished_over_cap_not_hidden_by_validator_probe(self):
        self.ready()
        ledger = self.ledger()
        ledger["allocations"][0].update(status="finished", start=1, end=5402, elapsed_seconds=5401)
        self.save(ledger)
        with self.assertRaisesRegex(ValueError, "over-cap"):
            self.preflight()

    def test_method_and_plan_ledger_binding(self):
        self.ready()
        good = self.ledger()
        for field, value in (("method", "dymos"), ("plan_sha256", "b" * 64)):
            ledger = copy.deepcopy(good)
            ledger[field] = value
            self.save(ledger)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "independent"):
                self.preflight()

    def test_changed_new_source_refuses(self):
        self.ready()
        (self.root / "scripts/run_teacher_devset_node.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "source SHA drift"):
            self.preflight()

    def test_duplicate_worker_launch_refuses(self):
        self.ready()
        for suffix in ("launch", "claim", "finished", "supervised", "ready"):
            candidate = self.folder / f"shard0.{suffix}.json"
            candidate.write_text("{}")
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "Existing worker"):
                self.preflight()
            candidate.unlink()

    def test_failed_old_media_claim_not_removed_or_retried(self):
        self.ready()
        claim = Path(self.plan["jobs"][0]["output"]).with_suffix(".claim.json")
        claim.write_text("old claim")
        with self.assertRaisesRegex(ValueError, "Failed old claim"):
            self.preflight()
        self.assertEqual(claim.read_text(), "old claim")

    def test_external_gpu_blocks_rank0_style_scope(self):
        self.ready()
        with mock.patch.object(node.dev.smoke, "supervisor_scope", side_effect=ValueError("external workload")):
            with self.assertRaisesRegex(ValueError, "external workload"):
                self.preflight()
        self.assertTrue(all("start" not in a for a in self.ledger()["allocations"]))

    def test_exact_frozen_worker_timeout_argv(self):
        command = node.command_for(self.plan, self.path, self.digest, self.pf, self.pf_sha, 3, 11, self.state)
        self.assertEqual(command[:4], ["/usr/bin/timeout", "--signal=TERM", "--kill-after=30s", "5370s"])
        for option, value in (("--gpus", "3"), ("--shard", "11"), ("--nshards", "16"),
                              ("--gen-cap-seconds", "5400"), ("--ledger", str(node.ledger_file(self.path))),
                              ("--node-preflight", str(self.pf)), ("--approved-node-preflight-sha256", self.pf_sha)):
            self.assertEqual(command[command.index(option) + 1], value)
        self.assertIn("--gpu-authorized", command)

    def test_default_cli_cpu_and_explicit_gpu_auth(self):
        base = ["--method", "alg", "--plan", str(self.path), "--approved-plan-sha256", self.digest,
                "--node-preflight", str(self.pf), "--approved-node-preflight-sha256", self.pf_sha,
                "--rank", "1", "--shard-offset", "0"]
        with mock.patch.object(node, "reserve", return_value={"GPU_used": False}) as reserve, mock.patch("builtins.print"):
            node.main(base)
            reserve.assert_called_once()
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            node.main(base + ["--run", "--gpu-authorized"])


class ChargingTests(Fixture):
    def setUp(self):
        super().setUp()
        self.ready()
        self.patch(node, "session_processes", return_value={})
        self.released = self.patch(node, "gpu_released", return_value=True)

    def test_success_true_elapsed_including_cleanup(self):
        item = self.start_item(elapsed=3)
        self.worker()
        result = self.finish()
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["elapsed_seconds"], 3)
        self.assertAlmostEqual(result["elapsed_GPUh"], result["elapsed_seconds"] / 3600)
        self.assertTrue(self.ledger()["allocations"][0]["release_confirmed"])
        self.assertEqual(result["start"], item["start"])

    def test_failed_spawn_still_charged(self):
        self.start_item()
        result = self.finish(rc=None, error="OSError spawn")
        self.assertFalse(result["success"])
        self.assertGreater(result["elapsed_GPUh"], 0)
        self.assertEqual(self.ledger()["allocations"][0]["status"], "finished")

    def test_timeout_and_user_term_charged(self):
        for shard, kwargs in ((0, dict(timeout=True, rc=124)), (1, dict(interrupted=signal.SIGTERM, rc=130))):
            self.start_item(shard)
            self.worker(shard)
            result = self.finish(shard, **kwargs)
            self.assertFalse(result["success"])
            self.assertGreater(result["elapsed_GPUh"], 0)

    def test_worker_complete_without_release_not_finalized(self):
        self.start_item()
        self.worker()
        self.released.return_value = False
        self.assertIs(self.finish(), False)
        item = self.ledger()["allocations"][0]
        self.assertEqual(item["status"], "running")
        self.assertNotIn("end", item)
        self.assertFalse((self.folder / "shard0.supervised.json").exists())

    def test_descendant_alive_not_free_finalized(self):
        self.start_item()
        with mock.patch.object(node, "session_processes", return_value={123: 123}):
            self.assertIs(self.finish(), False)
        self.assertNotIn("elapsed_GPUh", self.ledger()["allocations"][0])

    def test_bad_receipt_not_success(self):
        self.start_item()
        self.worker()
        receipt = self.folder / "shard0.finished.json"
        receipt.write_text("{}")
        self.assertFalse(self.finish()["success"])

    def test_finalize_owner_and_duplicate_refuse(self):
        self.start_item()
        with self.assertRaisesRegex(ValueError, "unowned"):
            node.finish_allocation(self.path, self.digest, 0, 999, self.state, 0, False, None, None)
        self.finish()
        before = (self.folder / "shard0.supervised.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "unowned"):
            self.finish()
        self.assertEqual((self.folder / "shard0.supervised.json").read_bytes(), before)

    def test_supervise_spawn_exception_clock_is_charged(self):
        with mock.patch.object(node.os, "getsid", side_effect=lambda pid: os.getpid() if pid == 0 else 12345), \
             mock.patch.object(node, "ready_barrier"), mock.patch.object(node, "stop_session"), \
             mock.patch.object(node.subprocess, "Popen", side_effect=OSError("spawn rejected")):
            result = node.supervise("alg", self.path, self.digest, self.pf, self.pf_sha, 1, 0, 0, self.state, self.source_sha, True)
        self.assertFalse(result["success"])
        self.assertIn("spawn rejected", result["error"])
        self.assertGreater(result["elapsed_GPUh"], 0)

    def test_supervisor_timeout_cleans_owned_session_and_charges(self):
        child = mock.Mock(pid=123456, returncode=124)
        child.poll.return_value = 124
        child.wait.return_value = 124
        with mock.patch.object(node.os, "getsid", side_effect=lambda pid: os.getpid() if pid == 0 else 12345), \
             mock.patch.object(node, "ready_barrier"), mock.patch.object(node, "stop_session") as cleanup, \
             mock.patch.object(node.subprocess, "Popen", return_value=child) as spawn:
            result = node.supervise("alg", self.path, self.digest, self.pf, self.pf_sha, 1, 0, 0, self.state, self.source_sha, True)
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["success"])
        self.assertGreater(result["elapsed_GPUh"], 0)
        self.assertEqual(cleanup.call_args.args[0], os.getpid())
        self.assertEqual(cleanup.call_args.args[2], (os.getpid(),))
        self.assertEqual(spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-0")

    def test_supervisor_user_term_cleans_owned_session_and_charges(self):
        child = mock.Mock(pid=123456, returncode=-signal.SIGTERM)
        child.poll.return_value = None
        child.wait.return_value = -signal.SIGTERM
        def spawn(*args, **kwargs):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return child
        with mock.patch.object(node.os, "getsid", side_effect=lambda pid: os.getpid() if pid == 0 else 12345), \
             mock.patch.object(node, "ready_barrier"), mock.patch.object(node, "stop_session") as cleanup, \
             mock.patch.object(node.subprocess, "Popen", side_effect=spawn):
            result = node.supervise("alg", self.path, self.digest, self.pf, self.pf_sha, 1, 0, 0, self.state, self.source_sha, True)
        self.assertEqual(result["interrupted_signal"], signal.SIGTERM)
        self.assertFalse(result["success"])
        self.assertGreater(result["elapsed_GPUh"], 0)
        self.assertEqual(cleanup.call_args.args[0], os.getpid())

    def test_changed_plan_after_ready_no_allocation(self):
        def barrier(*args):
            self.path.write_text("changed")
        with mock.patch.object(node.os, "getsid", side_effect=lambda pid: os.getpid() if pid == 0 else 12345), \
             mock.patch.object(node, "ready_barrier", side_effect=barrier), \
             mock.patch.object(node.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "changed at ready"):
                node.supervise("alg", self.path, self.digest, self.pf, self.pf_sha, 1, 0, 0, self.state, self.source_sha, True)
        spawn.assert_not_called()
        self.assertNotIn("start", self.ledger()["allocations"][0])

    def test_real_holder_exec_cpu_pause_after_durable_clock(self):
        # Execute the real holder.main exec branch, replacing only hardware pause.
        holder = self.root / "scripts/gpu_hold.py"
        worker = self.root / "scripts/run_teacher_devset.py"
        holder.write_text("import sys\nsys.path.insert(0, " + repr(str(ROOT / "scripts")) + ")\n"
                          "import gpu_hold as real, json, os\nfrom pathlib import Path\n"
                          "def pause(folder, gpus):\n"
                          " ledger=json.loads(Path(" + repr(str(node.ledger_file(self.path))) + ").read_bytes())\n"
                          " assert ledger['allocations'][0]['allocation_start_monotonic'] > 0\n"
                          " assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-0'\n"
                          " Path(" + repr(str(self.root / "paused-after-clock")) + ").write_text('CPU mock')\n"
                          "real.pause=pause\nreal.main()\n")
        worker.write_text("import sys\nsys.path.insert(0, " + repr(str(ROOT / "scripts")) + ")\n"
                          "import run_teacher_devset as d, fcntl, json, os\nfrom pathlib import Path\n"
                          "assert 'torch' not in sys.modules\n"
                          "p=Path(" + repr(str(node.ledger_file(self.path))) + ")\n"
                          "r=Path(" + repr(str(self.folder / "shard0.finished.json")) + ")\n"
                          "d.exclusive(r,dict(schema='teacher_devset_shard_receipt_v1',method='alg',plan_sha256="
                          + repr(self.digest) + ",shard=0,nshards=16,success=True))\n"
                          "with p.with_suffix('.json.lock').open('a') as lock:\n"
                          " fcntl.flock(lock,fcntl.LOCK_EX)\n"
                          " ledger=json.loads(p.read_bytes())\n"
                          " ledger['allocations'][0].update(status='running',worker_complete=True,worker_success=True,receipt_sha256=d.sha(r.read_bytes()))\n"
                          " d.checkpoint(p,ledger)\n")
        with mock.patch.object(node.os, "getsid", side_effect=lambda pid: os.getpid() if pid == 0 else 12345), \
             mock.patch.object(node, "ready_barrier"), mock.patch.object(node, "stop_session"):
            result = node.supervise("alg", self.path, self.digest, self.pf, self.pf_sha, 1, 0, 0, self.state, self.source_sha, True)
        self.assertTrue(result["success"], result)
        self.assertTrue((self.root / "paused-after-clock").exists())
        self.assertGreater(result["elapsed_GPUh"], 0)


class BarrierTests(Fixture):
    def node_launch(self):
        node.dev.exclusive(node.node_file(self.path, 1, "launch"),
                           dict(node_pid=os.getppid(), plan_sha256=self.digest, new_source_sha256=self.source_sha))

    def peer_ready(self, shard, **changes):
        proof = dict(plan_sha256=self.digest, node_preflight_sha256=self.pf_sha,
                     new_source_sha256=self.source_sha, shard=shard, node_pid=os.getppid(), supervisor_pid=os.getpid())
        proof.update(changes)
        node.dev.exclusive(self.folder / f"shard{shard}.ready.json", proof)

    def barrier(self, interrupted=lambda: None):
        return node.ready_barrier(self.path, self.digest, self.pf_sha, 1, 0, 0, self.source_sha, interrupted)

    def test_all_eight_cpu_ready_barrier(self):
        self.node_launch()
        for shard in range(1, 8):
            self.peer_ready(shard)
        self.barrier()
        self.assertEqual(len(list(self.folder.glob("shard*.ready.json"))), 8)
        self.assertFalse(list(self.folder.glob("shard*.launch.json")))

    def test_one_missing_worker_never_crosses_barrier(self):
        self.node_launch()
        for shard in range(1, 7):
            self.peer_ready(shard)
        with mock.patch.object(node.time, "sleep", side_effect=TimeoutError("test stops missing ready wait")):
            with self.assertRaisesRegex(TimeoutError, "missing ready"):
                self.barrier()
        self.assertFalse(node.ledger_file(self.path).exists())

    def test_wrong_peer_ready_digest_refuses(self):
        self.node_launch()
        self.peer_ready(1, new_source_sha256="wrong")
        with self.assertRaisesRegex(ValueError, "ready proof"):
            self.barrier()

    def test_term_at_barrier_no_gpu_allocation(self):
        self.node_launch()
        with self.assertRaisesRegex(ValueError, "interrupted"):
            self.barrier(lambda: signal.SIGTERM)
        self.assertFalse(node.ledger_file(self.path).exists())


class NodeTests(Fixture):
    def setUp(self):
        super().setUp()
        self.ready()
        self.patch(node, "session_processes", return_value={})
        self.patch(node, "gpu_released", return_value=True)
        self.cleanup = self.patch(node, "stop_session")
        self.kills = self.patch(node.os, "kill")

    def fake_child(self, shard, rc=0, charged=True):
        pid = 200000 + shard
        if charged:
            self.start_item(shard)
            self.worker(shard)
            ledger = self.ledger()
            ledger["allocations"][shard]["node_supervisor_pid"] = pid
            self.save(ledger)
        child = mock.Mock(pid=pid, returncode=rc)
        child.poll.return_value = rc
        child.wait.return_value = rc
        return child

    def test_parent_only_owned_eight_children_success_true_cost(self):
        launches = []
        def popen(command, **kwargs):
            launches.append((command, kwargs))
            shard = int(command[command.index("--shard") + 1])
            child = self.fake_child(shard)
            node.finish_allocation(self.path, self.digest, shard, child.pid, self.state, 0, False, None, None)
            return child
        with mock.patch.object(node.subprocess, "Popen", side_effect=popen):
            result = node.run_node(self.args())
        self.assertTrue(result["success"], result)
        self.assertEqual(len(launches), 8)
        self.assertTrue(all(k["start_new_session"] for _, k in launches))
        self.assertEqual(result["owned_shards"], list(range(8)))
        self.assertGreater(result["true_elapsed_GPUh"], 0)
        self.assertTrue(all(a["status"] == "reserved" for a in self.ledger()["allocations"][8:]))

    def test_parent_spawn_failure_finalizes_started_not_unstarted(self):
        spawned = []
        def popen(*args, **kwargs):
            if spawned:
                raise OSError("node spawn failure")
            spawned.append(self.fake_child(0, rc=1))
            return spawned[0]
        with mock.patch.object(node.subprocess, "Popen", side_effect=popen):
            result = node.run_node(self.args())
        child = spawned[0]
        self.assertFalse(result["success"])
        self.assertGreater(result["true_elapsed_GPUh"], 0)
        self.assertEqual(self.ledger()["allocations"][0]["status"], "finished")
        self.assertEqual(self.ledger()["allocations"][1]["status"], "reserved")
        self.cleanup.assert_called_once()
        self.assertEqual(self.cleanup.call_args.args[0], child.pid)

    def test_parent_term_stops_only_owned_supervisor(self):
        spawned = []
        def popen(*args, **kwargs):
            spawned.append(self.fake_child(0, rc=-signal.SIGTERM))
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return spawned[0]
        with mock.patch.object(node.subprocess, "Popen", side_effect=popen):
            result = node.run_node(self.args())
        self.assertFalse(result["success"])
        self.assertEqual(result["interrupted_signal"], signal.SIGTERM)
        self.assertGreater(result["true_elapsed_GPUh"], 0)
        self.assertEqual(self.cleanup.call_args.args[0], spawned[0].pid)
        self.assertFalse(any(c.args[0] == 987654 for c in self.kills.call_args_list))

    def test_parent_worker_release_unproven_never_free_finished(self):
        spawned = []
        def popen(*args, **kwargs):
            if spawned:
                raise OSError("fail")
            spawned.append(self.fake_child(0, rc=1))
            return spawned[0]
        with mock.patch.object(node, "gpu_released", return_value=False), \
             mock.patch.object(node.subprocess, "Popen", side_effect=popen):
            result = node.run_node(self.args())
        self.assertFalse(result["success"])
        self.assertIn(0, result["unfinalized_shards"])
        self.assertEqual(self.ledger()["allocations"][0]["status"], "running")
        self.assertNotIn("end", self.ledger()["allocations"][0])

    def test_parent_duplicate_one_shot_launch_not_overwritten(self):
        old = node.node_file(self.path, 1, "launch")
        old.write_text("old launch")
        with self.assertRaises(FileExistsError):
            node.run_node(self.args())
        self.assertEqual(old.read_text(), "old launch")


class ProcessTests(unittest.TestCase):
    def test_owned_session_term_kill_descendants_never_external(self):
        external = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        owned = subprocess.Popen([sys.executable, "-c",
            "import subprocess,sys,time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']); time.sleep(60)"],
            start_new_session=True)
        try:
            until = time.monotonic() + 3
            while len(node.session_processes(owned.pid)) < 2 and time.monotonic() < until:
                time.sleep(0.02)
            self.assertGreaterEqual(len(node.session_processes(owned.pid)), 2)
            node.stop_session(owned.pid, time.monotonic() + 0.2)
            owned.wait(timeout=3)
            self.assertFalse(node.session_processes(owned.pid))
            self.assertIsNone(external.poll())
        finally:
            for p in (owned, external):
                if p.poll() is None:
                    p.kill()
                p.wait(timeout=3)

    def test_cannot_cleanup_parent_session(self):
        with self.assertRaisesRegex(ValueError, "unrelated"):
            node.stop_session(os.getsid(os.getppid()), time.monotonic() + 1)


if __name__ == "__main__":
    unittest.main()
