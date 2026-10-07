"""Resident-limit behaviour of the model manager: several chat/VLM models
loaded at once, least-recently-used eviction, and the runtime limit change."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_switching_suite import FakeServer, make_model_dir
from server.model_manager import (MAX_RESIDENT_LIMIT, ModelManager,
                                  claimed_from_maps, mla_pool_bytes,
                                  mla_pool_regions)
from shared.config import HubConfig

CHAT = ("model-a", "model-b", "model-c")
ASR = "whisper-small-a16w8"


class ModelDirsCase(unittest.TestCase):
    """A catalog of fake model directories and a manager over a fake server."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for name in CHAT:
            make_model_dir(self.tmp, name, "chat")
        make_model_dir(self.tmp, ASR, "asr")
        patcher = patch.object(ModelManager, "_stop_model_streams", lambda *a: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def manager(self, limit, loaded=(ASR,), warmup=False):
        server = FakeServer(loaded)
        manager = ModelManager(
            server,
            catalog_dir=self.tmp,
            max_resident_chat_models=limit,
            asr_name=ASR,
            hub=HubConfig(allow_download=False),
            openai_base_url="http://127.0.0.1:9998",
            warmup=warmup,
            asr_warmup=False,
            switch_settle_s=0.0,
        )
        return manager, server


class ResidentLimitTests(ModelDirsCase):
    def test_default_limit_of_one_still_replaces_the_loaded_model(self):
        manager, server = self.manager(1)
        manager.load("model-a")
        result = manager.load("model-b")

        self.assertEqual(result["evicted"], ["model-a"])
        self.assertEqual(manager.residency()["resident"], ["model-b"])
        self.assertNotIn("model-a", server.model_names())

    def test_models_load_side_by_side_up_to_the_limit(self):
        manager, server = self.manager(2)
        manager.load("model-a")
        result = manager.load("model-b")

        self.assertEqual(result["evicted"], [])
        self.assertEqual(manager.residency()["resident"], ["model-b", "model-a"])
        self.assertLessEqual({"model-a", "model-b"}, set(server.model_names()))

    def test_loading_past_the_limit_evicts_the_least_recently_used(self):
        manager, server = self.manager(2)
        manager.load("model-a")
        manager.load("model-b")
        result = manager.load("model-c")

        self.assertEqual(result["evicted"], ["model-a"])
        self.assertEqual(manager.residency()["resident"], ["model-c", "model-b"])

    def test_touch_protects_a_model_from_the_next_eviction(self):
        manager, _ = self.manager(2)
        manager.load("model-a")
        manager.load("model-b")
        manager.touch("model-a")
        result = manager.load("model-c")

        self.assertEqual(result["evicted"], ["model-b"])
        self.assertEqual(manager.residency()["resident"], ["model-c", "model-a"])

    def test_reloading_a_resident_model_evicts_nothing(self):
        manager, server = self.manager(2)
        manager.load("model-a")
        manager.load("model-b")
        server.removed.clear()
        result = manager.load("model-a")

        self.assertEqual(result["evicted"], [])
        self.assertEqual(server.removed, [])
        self.assertEqual(manager.residency()["resident"], ["model-a", "model-b"])

    def test_the_asr_model_never_counts_against_the_chat_limit(self):
        manager, server = self.manager(2)
        manager.load("model-a")
        manager.load("model-b")

        self.assertIn(ASR, server.model_names())
        self.assertEqual(manager.active_asr(), ASR)
        self.assertNotIn(ASR, manager.residency()["resident"])

    def test_raising_the_limit_at_runtime_allows_a_second_model(self):
        manager, _ = self.manager(1)
        manager.load("model-a")
        result = manager.set_max_resident(2)
        manager.load("model-b")

        self.assertEqual(result["maxResident"], 2)
        self.assertEqual(result["evicted"], [])
        self.assertEqual(manager.residency()["resident"], ["model-b", "model-a"])

    def test_lowering_the_limit_unloads_the_least_recently_used_now(self):
        manager, server = self.manager(3)
        for name in CHAT:
            manager.load(name)
        result = manager.set_max_resident(1)

        self.assertEqual(result["evicted"], ["model-b", "model-a"])
        self.assertEqual(result["resident"], ["model-c"])
        self.assertEqual(
            [n for n in server.model_names() if n != ASR], ["model-c"])

    def test_limit_outside_the_allowed_range_is_refused(self):
        manager, _ = self.manager(1)
        for bad in (0, -1, MAX_RESIDENT_LIMIT + 1, "two", None):
            with self.assertRaises(ValueError):
                manager.set_max_resident(bad)
        self.assertEqual(manager.residency()["maxResident"], 1)

    def test_unload_removes_only_the_named_model(self):
        manager, _ = self.manager(2)
        manager.load("model-a")
        manager.load("model-b")
        manager.unload("model-a")

        self.assertEqual(manager.residency()["resident"], ["model-b"])

    def test_unload_all_frees_every_chat_model_and_keeps_speech(self):
        manager, server = self.manager(3)
        for name in CHAT:
            manager.load(name)
        result = manager.unload_all()

        self.assertEqual(sorted(result["unloaded"]), sorted(CHAT))
        self.assertEqual(result["failed"], [])
        self.assertEqual(manager.residency()["resident"], [])
        self.assertEqual(server.model_names(), [ASR])
        self.assertEqual(manager.active_asr(), ASR)

    def test_unload_all_with_nothing_loaded_is_a_no_op(self):
        manager, server = self.manager(2)
        result = manager.unload_all()
        self.assertEqual((result["unloaded"], result["failed"]), ([], []))
        self.assertEqual(server.model_names(), [ASR])

    def test_unload_all_carries_on_past_a_model_that_will_not_go(self):
        manager, server = self.manager(2)
        manager.load("model-a")
        manager.load("model-b")
        real = server.remove_model
        with patch.object(server, "remove_model",
                          side_effect=lambda n: False if n == "model-a" else real(n)):
            result = manager.unload_all()
        self.assertEqual(result["unloaded"], ["model-b"])
        self.assertEqual(result["failed"], ["model-a"])
        self.assertEqual(manager.residency()["resident"], ["model-a"])   # still tracked

    def test_a_failed_load_leaves_the_other_resident_models_alone(self):
        manager, server = self.manager(2, warmup=True)
        with patch.object(ModelManager, "_warm_check", return_value=(True, "")):
            manager.load("model-a")
        with patch.object(ModelManager, "_warm_check",
                          return_value=(False, "HTTP 500: Cannot allocate memory")):
            with self.assertRaises(RuntimeError) as caught:
                manager.load("model-b")

        self.assertIn("model-a", str(caught.exception))
        self.assertEqual(manager.residency()["resident"], ["model-a"])
        self.assertIn("model-a", server.model_names())
        self.assertNotIn("model-b", server.model_names())

    def test_status_reports_the_resident_order_and_the_limit(self):
        manager, _ = self.manager(2)
        manager.load("model-a")
        manager.load("model-b")
        status = manager.status()

        self.assertEqual(status["resident"], ["model-b", "model-a"])
        self.assertEqual(status["maxResident"], 2)
        self.assertEqual(status["maxResidentLimit"], MAX_RESIDENT_LIMIT)


class LoadedStateIsCheckedTests(ModelDirsCase):
    """Models can go away without this manager unloading them: another client
    of the runtime, or an accelerator reset by another program."""

    class Inline:
        """Stands in for threading.Thread: runs the target at start()."""

        def __init__(self, target=None, args=(), **_kw):
            self._target, self._args = target, args

        def start(self):
            self._target(*self._args)

    def probes(self, chat=None, asr=(True, "")):
        """Patch the two probes. `chat` maps a model name to its (ok, detail)."""
        chat = chat or {}
        return (patch.object(ModelManager, "_warm_check",
                             side_effect=lambda name, timeout=600: chat.get(name, (True, ""))),
                patch.object(ModelManager, "_warm_check_asr",
                             side_effect=lambda name, timeout=300: asr))

    def test_a_model_removed_by_another_client_is_no_longer_reported(self):
        manager, server = self.manager(2)
        manager.load("model-a")
        manager.load("model-b")
        server.names.remove("model-a")                 # someone else unloaded it
        status = manager.residency_status()
        self.assertEqual(status["resident"], ["model-b"])
        self.assertNotIn("model-a", status["loaded"])
        self.assertEqual(status["lost"], [])            # an ordinary unload, not a loss

    def test_a_model_loaded_by_another_client_shows_up(self):
        manager, server = self.manager(2)
        manager.load("model-a")
        server.names.append("model-b")
        self.assertEqual(manager.residency_status()["resident"], ["model-a", "model-b"])

    def test_verify_drops_only_the_model_that_stopped_answering(self):
        manager, server = self.manager(2)
        manager.load("model-a")
        manager.load("model-b")
        chat, asr = self.probes({"model-a": (False, "HTTP 500: mlashm handle is invalid")})
        with chat, asr:
            result = manager.verify_loaded(reason="the accelerator was reset by another program")
        self.assertEqual(result["lost"], ["model-a"])
        self.assertEqual(sorted(result["checked"]), ["model-a", "model-b", ASR])
        status = manager.residency_status()
        self.assertEqual(status["resident"], ["model-b"])
        self.assertNotIn("model-a", server.model_names())
        self.assertEqual(manager.active_asr(), ASR)
        self.assertEqual([(l["name"], l["reason"]) for l in status["lost"]],
                         [("model-a", "the accelerator was reset by another program")])
        self.assertEqual(status["lost"][0]["seq"], 1)

    def test_verify_one_model_by_name(self):
        manager, _ = self.manager(2)
        manager.load("model-a")
        manager.load("model-b")
        chat, asr = self.probes({"model-b": (False, "timed out")})
        with chat, asr:
            self.assertEqual(manager.verify_loaded("model-a"), {"checked": ["model-a"], "lost": []})
            self.assertEqual(manager.verify_loaded("model-b")["lost"], ["model-b"])
            self.assertEqual(manager.verify_loaded("model-b"), {"checked": [], "lost": []})

    def test_a_lost_model_the_runtime_will_not_remove_is_still_reported_unloaded(self):
        manager, server = self.manager(2)
        manager.load("model-a")
        chat, asr = self.probes({"model-a": (False, "connection reset")})
        with chat, asr, patch.object(server, "remove_model", return_value=False):
            manager.verify_loaded()
        self.assertIn("model-a", server.model_names())             # still registered
        self.assertNotIn("model-a", manager.residency_status()["loaded"])
        self.assertFalse(next(e for e in manager.catalog() if e["name"] == "model-a")["loaded"])
        manager.load("model-a")                                    # loading it again recovers
        self.assertEqual(manager.residency()["resident"], ["model-a"])

    def test_a_speech_probe_rejecting_the_silent_clip_is_not_a_loss(self):
        manager, _ = self.manager(1)
        chat, asr = self.probes(asr=(False, "HTTP 400: bad audio"))
        with chat, asr:
            self.assertEqual(manager.verify_loaded()["lost"], [])
        self.assertEqual(manager.active_asr(), ASR)

    def test_a_lost_speech_model_is_no_longer_active(self):
        manager, _ = self.manager(1)
        chat, asr = self.probes(asr=(False, "HTTP 500: mlashm handle is invalid"))
        with chat, asr:
            self.assertEqual(manager.verify_loaded()["lost"], [ASR])
        self.assertIsNone(manager.active_asr())

    def test_an_outside_accelerator_reset_triggers_verification(self):
        with patch("server.model_manager.dispatcher_identity", return_value=(100, 5)):
            manager, _ = self.manager(2)
            manager.load("model-a")
            manager.load("model-b")
            self.assertEqual(manager.residency()["lost"], [])      # same dispatcher: nothing to do
        chat, asr = self.probes({"model-a": (False, "HTTP 500: mlashm"), "model-b": (False, "HTTP 500: mlashm")},
                                asr=(False, "HTTP 500: mlashm"))
        with patch("server.model_manager.dispatcher_identity", return_value=(222, 9)), \
                patch("server.model_manager.threading.Thread", self.Inline), chat, asr:
            status = manager.residency_status()
        self.assertEqual(status["resident"], [])
        self.assertEqual(status["loaded"], [])
        self.assertEqual(sorted(l["name"] for l in status["lost"]), ["model-a", "model-b", ASR])
        self.assertTrue(all("reset by another program" in l["reason"] for l in status["lost"]))
        self.assertFalse(status["verifying"])

    def test_models_that_survive_the_check_stay_loaded(self):
        with patch("server.model_manager.dispatcher_identity", return_value=(100, 5)):
            manager, _ = self.manager(1)
            manager.load("model-a")
        chat, asr = self.probes()
        with patch("server.model_manager.dispatcher_identity", return_value=(222, 9)), \
                patch("server.model_manager.threading.Thread", self.Inline), chat as probe, asr:
            first = manager.residency_status()
            manager.residency_status()                 # the new dispatcher is adopted: no re-check
        self.assertEqual(first["resident"], ["model-a"])
        self.assertEqual(first["lost"], [])
        self.assertEqual(probe.call_count, 1)

    def test_a_dispatcher_that_is_down_is_not_mistaken_for_a_reset(self):
        with patch("server.model_manager.dispatcher_identity", return_value=(100, 5)):
            manager, _ = self.manager(1)
            manager.load("model-a")
        chat, asr = self.probes()
        with patch("server.model_manager.dispatcher_identity", return_value=None), \
                patch("server.model_manager.threading.Thread", self.Inline), chat as probe, asr:
            self.assertEqual(manager.residency_status()["resident"], ["model-a"])
        self.assertEqual(probe.call_count, 0)


class UnloadTimingTests(ModelDirsCase):
    """Unloads are timed so clients can show progress for the next one."""

    def setUp(self):
        super().setUp()
        (self.tmp / "model-a" / "elf_files" / "stage0_mla.elf").write_bytes(b"x" * 2_000_000)
        (self.tmp / "model-b" / "elf_files" / "stage0_mla.elf").write_bytes(b"x" * 1_000_000)

    def entry(self, manager, name):
        return next(e for e in manager.catalog() if e["name"] == name)

    def clock(self, *ticks):
        """Patch the manager's clock to step through `ticks`, then hold."""
        values = list(ticks)
        return patch("server.model_manager.time.monotonic",
                     side_effect=lambda: values.pop(0) if len(values) > 1 else values[0])

    def test_there_is_no_estimate_before_anything_was_unloaded(self):
        manager, _ = self.manager(2)
        manager.load("model-a")
        self.assertIsNone(self.entry(manager, "model-a")["estimatedUnloadS"])

    def test_an_unload_reports_its_duration_and_becomes_the_estimate(self):
        manager, _ = self.manager(2)
        manager.load("model-a")
        with self.clock(100.0, 100.0, 104.0, 104.0):   # unload start, removal start/end, end
            result = manager.unload("model-a")
        self.assertEqual(result["state"], "unloaded")
        self.assertEqual(result["unload_seconds"], 4.0)
        self.assertEqual(self.entry(manager, "model-a")["estimatedUnloadS"], 4.0)

    def test_other_models_are_estimated_from_the_learned_rate(self):
        manager, _ = self.manager(2)
        manager.load("model-a")
        with self.clock(100.0, 100.0, 104.0, 104.0):   # 2 MB took 4 s
            manager.unload("model-a")
        # model-b is half the size and has never been unloaded itself.
        self.assertAlmostEqual(self.entry(manager, "model-b")["estimatedUnloadS"], 2.0)

    def test_evicting_during_a_load_is_timed_too(self):
        manager, _ = self.manager(1)
        manager.load("model-a")
        manager.load("model-b")                        # pushes model-a out
        self.assertIsNotNone(self.entry(manager, "model-a")["estimatedUnloadS"])

    def test_a_model_that_was_not_loaded_teaches_nothing(self):
        manager, _ = self.manager(2)
        result = manager.unload("model-a")
        self.assertEqual(result["state"], "absent")
        self.assertIsNone(self.entry(manager, "model-a")["estimatedUnloadS"])

    def test_a_load_reports_which_model_it_is_unloading_first(self):
        manager, server = self.manager(1)
        manager.load("model-a")
        seen = []
        real_remove = server.remove_model

        def remove(name):
            seen.append(manager.loading_status())
            return real_remove(name)

        real_add = server.add_model

        def add(path, name):
            seen.append(manager.loading_status())
            return real_add(path, name)

        with patch.object(server, "remove_model", side_effect=remove), \
                patch.object(server, "add_model", side_effect=add):
            manager.load("model-b")
        self.assertEqual([(s["phase"], s["victim"]) for s in seen],
                         [("unloading", "model-a"), ("loading", None)])
        self.assertIsNone(manager.loading_status())


class MlaMemoryTests(ModelDirsCase):
    """Accelerator memory: pool size from the device tree, usage estimated
    from the ELF stages of the models this server has loaded."""

    def reserved(self, nodes, cells=(2, 2)):
        base = self.tmp / "reserved-memory"
        base.mkdir()
        (base / "#address-cells").write_bytes(cells[0].to_bytes(4, "big"))
        (base / "#size-cells").write_bytes(cells[1].to_bytes(4, "big"))
        for name, addr, size in nodes:
            (base / name).mkdir()
            (base / name / "reg").write_bytes(
                addr.to_bytes(cells[0] * 4, "big") + size.to_bytes(cells[1] * 4, "big"))
        return base

    def test_pool_size_is_the_dms_region(self):
        base = self.reserved([
            ("dms@0x1400000000", 0x1300000000, 16 << 30),
            ("linux,cma", 0x1000000000, 0x6FC00000),
            ("evmem@0xF0000000", 0xF0000000, 0x4000000),
        ])
        self.assertEqual(mla_pool_bytes(base), 16 << 30)

    def test_pool_size_honours_the_cell_widths(self):
        base = self.reserved([("dms@40000000", 0x40000000, 0x20000000)], cells=(1, 1))
        self.assertEqual(mla_pool_bytes(base), 0x20000000)

    def test_pool_size_is_unknown_without_a_dms_region(self):
        self.assertIsNone(mla_pool_bytes(self.reserved([("linux,cma", 0, 1 << 20)])))
        self.assertIsNone(mla_pool_bytes(self.tmp / "missing"))

    POOL = [(0x1300000000, 16 << 30)]
    # The dispatcher's map on a DevKit: 512 MB slabs of the pool, a small pool
    # buffer, one slab mapped twice, and mappings that are not pool memory.
    MAPS = """\
fffa80000000-fffaa0000000 rw-s 16c0000000 00:05 91 /dev/simaai-mem
fffaa0000000-fffac0000000 rw-s 1680000000 00:05 91 /dev/simaai-mem
fffb00000000-fffb20000000 rw-s 1680000000 00:05 91 /dev/simaai-mem
fffac0000000-fffac0100000 rw-s 1300000000 00:05 91 /dev/simaai-mem
fffad0000000-fffad0400000 rw-s 1000400000 00:05 91 /dev/simaai-mem
ffff9a000000-ffff9a021000 rw-p 00000000 00:00 0 [heap]
ffff9b000000-ffff9b200000 r-xp 00000000 b3:02 77 /usr/lib/aarch64-linux-gnu/libc.so.6
"""

    def test_pool_regions_carry_the_address_too(self):
        base = self.reserved([("dms@0x1400000000", 0x1300000000, 16 << 30)])
        self.assertEqual(mla_pool_regions(base), self.POOL)

    def test_claimed_counts_each_mapped_piece_of_the_pool_once(self):
        # two 512 MB slabs + 1 MB; the repeat and the non-pool mappings do not count
        self.assertEqual(claimed_from_maps(self.MAPS, self.POOL), (1024 + 1) << 20)
        self.assertEqual(claimed_from_maps("", self.POOL), 0)
        self.assertEqual(claimed_from_maps("not a maps line\n", self.POOL), 0)

    def memory_patches(self, maps):
        return (patch("server.model_manager.mla_pool_regions", return_value=self.POOL),
                patch("server.model_manager.read_dispatcher_maps", return_value=maps))

    def test_usage_is_the_elf_size_of_what_is_loaded(self):
        for name, size in (("model-a", 3000), ("model-b", 500), (ASR, 70)):
            (self.tmp / name / "elf_files" / "stage0_mla.elf").write_bytes(b"x" * size)
            (self.tmp / name / "tokenizer.json").write_bytes(b"y" * 9999)   # not counted
        manager, _ = self.manager(2)
        pool, maps = self.memory_patches(self.MAPS)
        with pool, maps:
            self.assertEqual(manager.mla_memory()["usedBytes"], 70)   # speech model only
            manager.load("model-a")
            manager.load("model-b")
            mla = manager.status()["mla"]
            self.assertEqual(mla["totalBytes"], 16 << 30)
            self.assertEqual(mla["claimedBytes"], (1024 + 1) << 20)
            self.assertEqual(mla["models"], {"model-a": 3000, "model-b": 500, ASR: 70})
            self.assertEqual(mla["usedBytes"], 3570)
            self.assertTrue(mla["estimated"])
            manager.unload("model-a")
            self.assertEqual(manager.mla_memory()["usedBytes"], 570)

    def test_claimed_is_none_when_the_dispatcher_cannot_be_read(self):
        manager, _ = self.manager(1)
        pool, maps = self.memory_patches(None)
        with pool, maps as reader:
            first = manager.mla_memory()
            manager.mla_memory()
        self.assertIsNone(first["claimedBytes"])
        self.assertEqual(first["totalBytes"], 16 << 30)
        self.assertEqual(reader.call_count, 1)   # backs off instead of retrying each poll

    def test_unknown_pool_size_is_reported_as_none(self):
        manager, _ = self.manager(1)
        with patch("server.model_manager.mla_pool_regions", return_value=[]):
            memory = manager.mla_memory()
        self.assertIsNone(memory["totalBytes"])
        self.assertIsNone(memory["claimedBytes"])

    def test_a_load_the_runtime_rejects_names_a_full_pool(self):
        manager, server = self.manager(2)
        manager.load("model-a")
        full = "".join(
            f"{0xfff000000000 + i * 0x20000000:x}-{0xfff000000000 + (i + 1) * 0x20000000:x} "
            f"rw-s {0x1300000000 + i * 0x20000000:x} 00:05 91 /dev/simaai-mem\n"
            for i in range(31))                       # 15.5 of 16 GiB
        pool, maps = self.memory_patches(full)

        def refuse(path, name):
            raise RuntimeError("Failed to bulk load model through MLASHM dispatcher")

        with pool, maps, patch.object(server, "add_model", side_effect=refuse):
            with self.assertRaises(RuntimeError) as caught:
                manager.load("model-b")
        message = str(caught.exception)
        self.assertIn("nearly full", message)
        self.assertIn("15.5 of 16.0 GB", message)
        self.assertIn("Reset MLA", message)
        self.assertIn("model-a", message)             # still loaded, and named
        self.assertEqual(manager.residency()["resident"], ["model-a"])
        self.assertEqual(manager.load_logs()["lastError"]["kind"], "mla")


if __name__ == "__main__":
    unittest.main()
