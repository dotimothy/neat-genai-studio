"""Resident-limit behaviour of the model manager: several chat/VLM models
loaded at once, least-recently-used eviction, and the runtime limit change."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_switching_suite import FakeServer, make_model_dir
from server.model_manager import MAX_RESIDENT_LIMIT, ModelManager
from shared.config import HubConfig

CHAT = ("model-a", "model-b", "model-c")
ASR = "whisper-small-a16w8"


class ResidentLimitTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
