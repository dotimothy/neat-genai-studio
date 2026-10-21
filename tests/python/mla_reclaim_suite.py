"""Giving unloaded models' accelerator memory back.

The MLA dispatcher frees memory only when it has no clients at all (measured
on runtime 0.5.0: remove_model returned none of it while the process stayed
connected, and reloading the same model took fresh memory). This server stays
a client while any model is registered — the speech model always is — so an
unload or a switch unloads everything for a moment, waits for the dispatcher
to go idle, and loads back what should stay, speech model first.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_switching_suite import FakeServer, make_model_dir
from server import model_manager
from server.model_manager import ModelManager, dispatcher_client_count
from shared.config import HubConfig

ASR = "whisper-small-a16w8"


class ReclaimTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.models = self.tmp / "models"
        self.models.mkdir()
        for name in ("model-a", "model-b", "model-c"):
            make_model_dir(self.models, name, "chat")
        for name in (ASR, "whisper-medium-a16w8"):
            make_model_dir(self.models, name, "asr")
        patcher = patch.object(ModelManager, "_stop_model_streams", lambda *a: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Other programs connected to the dispatcher (None: unreadable).
        self.other_clients = 0
        self.server = None
        patcher = patch.object(model_manager, "dispatcher_client_count", self._clients)
        patcher.start()
        self.addCleanup(patcher.stop)
        # What was registered each time the drain waited for the dispatcher.
        self.at_idle_wait = []
        self.idle_result = True
        patcher = patch.object(ModelManager, "_wait_dispatcher_idle", self._wait_idle)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _clients(self, *_a, **_k):
        if self.other_clients is None:
            return None
        return self.other_clients + (1 if self.server and self.server.names else 0)

    def _wait_idle(self, *_a, **_k):
        self.at_idle_wait.append(list(self.server.names))
        return self.idle_result

    def manager(self, loaded=(ASR,), reclaim=True, max_resident=1):
        self.server = FakeServer(loaded)
        manager = ModelManager(
            self.server, catalog_dir=self.models, max_resident_chat_models=max_resident,
            asr_name=ASR, hub=HubConfig(allow_download=False),
            openai_base_url="http://127.0.0.1:9998", warmup=False, asr_warmup=False,
            switch_settle_s=0.0, reclaim_memory=reclaim,
        )
        manager.scan_catalog()
        return manager, self.server

    @staticmethod
    def last(manager):
        return manager.residency()["reclaim"]["last"]

    def test_a_switch_drains_everything_then_loads_speech_first(self):
        manager, server = self.manager(loaded=(ASR, "model-a"))
        manager._resident = ["model-a"]
        result = manager.load("model-b")
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["evicted"], ["model-a"])
        self.assertEqual(self.at_idle_wait, [[]])           # nothing registered while waiting
        self.assertEqual(server.removed, ["model-a", ASR])
        self.assertEqual([n for _p, n in server.added], [ASR, "model-b"])
        self.assertEqual(manager.residency()["resident"], ["model-b"])
        self.assertEqual(manager._active_asr, ASR)
        self.assertEqual(self.last(manager)["result"], "freed")
        self.assertEqual(self.last(manager)["failed"], {})
        self.assertIsNone(manager._loading)

    def test_kept_models_go_back_oldest_first(self):
        manager, server = self.manager(loaded=(ASR, "model-a", "model-b"), max_resident=2)
        manager._resident = ["model-b", "model-a"]          # most recently used first
        manager.load("model-c")                              # evicts model-a
        self.assertEqual([n for _p, n in server.added], [ASR, "model-b", "model-c"])
        self.assertEqual(manager.residency()["resident"], ["model-c", "model-b"])

    def test_a_load_that_fits_does_not_drain(self):
        manager, server = self.manager(loaded=(ASR,))
        result = manager.load("model-a")
        self.assertEqual(result["state"], "ready")
        self.assertEqual(server.removed, [])
        self.assertEqual([n for _p, n in server.added], ["model-a"])
        self.assertEqual(self.at_idle_wait, [])
        self.assertIsNone(self.last(manager))

    def test_unload_keeps_the_rest_loaded(self):
        manager, server = self.manager(loaded=(ASR, "model-a", "model-b"), max_resident=2)
        manager._resident = ["model-b", "model-a"]
        result = manager.unload("model-a")
        self.assertEqual(result["state"], "unloaded")
        self.assertEqual(sorted(server.removed), sorted(["model-a", "model-b", ASR]))
        self.assertEqual([n for _p, n in server.added], [ASR, "model-b"])
        self.assertEqual(manager.residency()["resident"], ["model-b"])

    def test_unload_all_keeps_speech(self):
        manager, server = self.manager(loaded=(ASR, "model-a", "model-b"), max_resident=2)
        manager._resident = ["model-b", "model-a"]
        result = manager.unload_all()
        self.assertEqual(sorted(result["unloaded"]), ["model-a", "model-b"])
        self.assertEqual([n for _p, n in server.added], [ASR])
        self.assertEqual(server.names, [ASR])
        self.assertEqual(manager.residency()["resident"], [])

    def test_lowering_the_limit_keeps_the_most_recent(self):
        manager, server = self.manager(loaded=(ASR, "model-a", "model-b", "model-c"),
                                       max_resident=3)
        manager._resident = ["model-c", "model-b", "model-a"]
        result = manager.set_max_resident(1)
        self.assertEqual(result["evicted"], ["model-b", "model-a"])
        self.assertEqual([n for _p, n in server.added], [ASR, "model-c"])
        self.assertEqual(sorted(server.names), sorted([ASR, "model-c"]))

    def test_an_asr_switch_drains_and_loads_the_new_speech_model(self):
        manager, server = self.manager(loaded=(ASR, "model-a"))
        manager._resident = ["model-a"]
        result = manager.set_active_asr("whisper-medium-a16w8")
        self.assertEqual(result["activeAsr"], "whisper-medium-a16w8")
        self.assertEqual(server.removed, [ASR, "model-a"])
        self.assertEqual(self.at_idle_wait, [[]])
        # The old speech model is not loaded back; the new one comes last.
        self.assertEqual([n for _p, n in server.added], ["model-a", "whisper-medium-a16w8"])

    def test_another_program_on_the_accelerator_blocks_the_drain(self):
        self.other_clients = 1
        manager, server = self.manager(loaded=(ASR, "model-a"))
        manager._resident = ["model-a"]
        manager.unload("model-a")
        self.assertEqual(server.removed, ["model-a"])        # ASR stays put
        self.assertEqual(server.added, [])
        self.assertEqual(self.at_idle_wait, [])
        last = self.last(manager)
        self.assertEqual(last["result"], "blocked")
        self.assertIn("1 other program", last["detail"])

    def test_a_lone_client_is_another_program_when_this_server_holds_nothing(self):
        self.other_clients = 1
        manager, server = self.manager(loaded=())
        manager._reclaim_memory("test")
        self.assertEqual(self.last(manager)["result"], "blocked")
        self.assertEqual(self.at_idle_wait, [])

    def test_an_unreadable_client_count_still_drains(self):
        self.other_clients = None
        self.idle_result = None
        manager, server = self.manager(loaded=(ASR, "model-a"))
        manager._resident = ["model-a"]
        manager.load("model-b")
        self.assertEqual([n for _p, n in server.added], [ASR, "model-b"])
        self.assertEqual(self.last(manager)["result"], "unconfirmed")

    def test_a_dispatcher_that_stays_busy_still_loads_everything_back(self):
        self.idle_result = False
        manager, server = self.manager(loaded=(ASR, "model-a"))
        manager._resident = ["model-a"]
        result = manager.load("model-b")
        self.assertEqual(result["state"], "ready")
        self.assertEqual([n for _p, n in server.added], [ASR, "model-b"])
        self.assertEqual(self.last(manager)["result"], "not-idle")

    def test_a_model_that_will_not_load_back_is_reported(self):
        manager, server = self.manager(loaded=(ASR, "model-a", "model-b"), max_resident=2)
        manager._resident = ["model-b", "model-a"]
        add = server.add_model

        def add_model(path, name):
            if name == "model-b" and server.removed:
                raise RuntimeError("no luck")
            return add(path, name)

        server.add_model = add_model
        manager.unload("model-a")
        self.assertEqual([n for _p, n in server.added], [ASR])
        self.assertIn("model-b", self.last(manager)["failed"])

    def test_disabled_unloads_in_place(self):
        manager, server = self.manager(loaded=(ASR, "model-a"), reclaim=False)
        manager._resident = ["model-a"]
        result = manager.unload("model-a")
        self.assertEqual(result["state"], "unloaded")
        self.assertEqual(server.removed, ["model-a"])
        self.assertEqual(server.added, [])
        reclaim = manager.residency()["reclaim"]
        self.assertFalse(reclaim["enabled"])
        self.assertIsNone(reclaim["last"])


class DispatcherClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_counts_connected_sockets_on_the_control_path(self):
        table = self.tmp / "unix"
        table.write_text(
            "Num       RefCount Protocol Flags    Type St Inode Path\n"
            "0000: 00000002 00000000 00010000 0001 01 100 /tmp/mlactrl\n"     # listening
            "0000: 00000003 00000000 00000000 0001 03 101 /tmp/mlactrl\n"     # a client
            "0000: 00000003 00000000 00000000 0001 03 102 /tmp/mlactrl\n"     # a client
            "0000: 00000003 00000000 00000000 0001 03 103\n"                  # client end
            "0000: 00000003 00000000 00000000 0001 03 104 /run/other.sock\n")
        self.assertEqual(dispatcher_client_count(table=str(table)), 2)

    def test_unreadable_is_none(self):
        self.assertIsNone(dispatcher_client_count(table=str(self.tmp / "missing")))

    def test_waiting_for_idle(self):
        manager = ModelManager.__new__(ModelManager)
        counts = iter([2, 1, 0])
        with patch.object(model_manager, "dispatcher_client_count", lambda: next(counts)), \
                patch.object(model_manager.time, "sleep", lambda _s: None):
            self.assertTrue(manager._wait_dispatcher_idle(timeout=5))
        with patch.object(model_manager, "dispatcher_client_count", lambda: None), \
                patch.object(model_manager.time, "sleep", lambda _s: None):
            self.assertIsNone(manager._wait_dispatcher_idle(timeout=5))
        with patch.object(model_manager, "dispatcher_client_count", lambda: 1), \
                patch.object(model_manager.time, "sleep", lambda _s: None):
            self.assertFalse(manager._wait_dispatcher_idle(timeout=0))


if __name__ == "__main__":
    unittest.main()
