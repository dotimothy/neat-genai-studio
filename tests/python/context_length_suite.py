"""Context length in the catalog: the tokens a compiled model holds, prompt plus
reply (vlm_config.json pipeline_cfg.max_num_tokens). The Studio's Max response
tokens slider tops out there and defaults to it.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_switching_suite import FakeServer, make_model_dir
from server.model_manager import ModelManager
from shared.config import HubConfig


class CatalogContextLengthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        patcher = patch.object(ModelManager, "_stop_model_streams", lambda *a: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def chat_model(self, name, vlm_config):
        model = make_model_dir(self.tmp, name, "chat")
        (model / "devkit" / "vlm_config.json").write_text(json.dumps(vlm_config))
        return model

    def manager(self):
        make_model_dir(self.tmp, "whisper-small-a16w8", "asr")
        return ModelManager(
            FakeServer(("whisper-small-a16w8",)), catalog_dir=self.tmp,
            max_resident_chat_models=1, asr_name="whisper-small-a16w8",
            hub=HubConfig(allow_download=False),
            openai_base_url="http://127.0.0.1:9998", warmup=False,
            asr_warmup=False, switch_settle_s=0.0)

    def test_the_compiled_token_budget_is_reported_not_the_rope_limit(self):
        # lm_cfg.max_position_embeddings is the per-chunk 2048, not the context.
        self.chat_model("r1", {"lm_cfg": {"max_position_embeddings": 2048},
                               "pipeline_cfg": {"max_num_tokens": 8192}})
        self.chat_model("small", {"pipeline_cfg": {"max_num_tokens": 2048}})
        self.chat_model("unknown", {})
        manager = self.manager()
        ctx = {e["name"]: e["contextLength"] for e in manager.catalog()}
        self.assertEqual(ctx["r1"], 8192)
        self.assertEqual(ctx["small"], 2048)
        self.assertIsNone(ctx["unknown"])
        self.assertIsNone(ctx["whisper-small-a16w8"])

    def test_a_broken_config_is_unknown_not_an_error(self):
        model = make_model_dir(self.tmp, "broken", "chat")
        (model / "devkit" / "vlm_config.json").write_text("{not json")
        self.assertIsNone(self.manager()._context_length(model))


if __name__ == "__main__":
    unittest.main()
