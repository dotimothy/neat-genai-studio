"""Reasoning that the chat template opens: DeepSeek-R1's generation prompt ends
in ``<think>``, so the model streams its reasoning with no opening tag and only
emits ``</think>``. The catalog reports it per model so every client shows the
reply as thinking from its first token instead of guessing.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_switching_suite import FakeServer, make_model_dir
from cli_think_suite import cli
from server.model_manager import ModelManager
from shared.chat_template import generation_opens_thinking
from shared.config import HubConfig

# Shaped like DeepSeek-R1-Distill's template: the generation prompt opens <think>.
DEEPSEEK = ("{{ bos_token }}{% for m in messages %}{% if m['role'] == 'user' %}"
            "<｜User｜>{{ m['content'] }}{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<｜Assistant｜><think>\n{% endif %}")
# Shaped like Qwen3's: the model opens <think> itself; with thinking off the
# template writes an empty, already closed block.
QWEN3 = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
         "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n"
         "{% if enable_thinking is defined and enable_thinking is false %}"
         "<think>\n\n</think>\n\n{% endif %}{% endif %}")
# Opens <think> only while thinking is on.
GATED = ("{% for m in messages %}{{ m.content }}{% endfor %}"
         "{% if add_generation_prompt %}ASSISTANT:"
         "{% if enable_thinking is not defined or enable_thinking %}<think>\n{% endif %}"
         "{% endif %}")
PLAIN = "{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}A:{% endif %}"


def model_with_template(root: Path, name: str, template: str, *, embedded=True) -> Path:
    model = make_model_dir(root, name, "chat")
    config = {"bos_token": {"content": "<｜begin▁of▁sentence｜>"}, "eos_token": "</s>"}
    if embedded:
        config["chat_template"] = template
    else:
        (model / "devkit" / "chat_template.jinja").write_text(template)
    (model / "devkit" / "tokenizer_config.json").write_text(json.dumps(config))
    return model


class TemplateOpensThinkingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_deepseek_style_template_opens_thinking_whatever_the_switch(self):
        model = model_with_template(self.tmp, "r1", DEEPSEEK)
        self.assertTrue(generation_opens_thinking(model))
        self.assertTrue(generation_opens_thinking(model, False))

    def test_a_model_that_opens_think_itself_is_not_reported(self):
        model = model_with_template(self.tmp, "qwen3", QWEN3, embedded=False)
        self.assertFalse(generation_opens_thinking(model))
        # Thinking off: the block the template writes is already closed.
        self.assertFalse(generation_opens_thinking(model, False))

    def test_the_switch_is_honoured_when_the_template_reads_it(self):
        model = model_with_template(self.tmp, "gated", GATED)
        self.assertTrue(generation_opens_thinking(model))
        self.assertFalse(generation_opens_thinking(model, False))

    def test_templates_without_reasoning_and_broken_ones_are_not_reported(self):
        self.assertFalse(generation_opens_thinking(model_with_template(self.tmp, "plain", PLAIN)))
        broken = model_with_template(self.tmp, "broken", "{% if %}<think>")
        self.assertFalse(generation_opens_thinking(broken))
        self.assertFalse(generation_opens_thinking(self.tmp / "missing"))

    def test_the_runtime_lookup_order_is_followed(self):
        # devkit/chat_template.jinja wins over the embedded template, as in the runtime.
        model = model_with_template(self.tmp, "both", DEEPSEEK)
        (model / "devkit" / "chat_template.jinja").write_text(PLAIN)
        self.assertFalse(generation_opens_thinking(model))


class CatalogReportsOpenedThinkingTests(unittest.TestCase):
    def test_catalog_entries_carry_both_flags(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        model_with_template(tmp, "r1", DEEPSEEK)
        model_with_template(tmp, "gated", GATED)
        model_with_template(tmp, "plain", PLAIN)
        make_model_dir(tmp, "whisper-small-a16w8", "asr")
        with patch.object(ModelManager, "_stop_model_streams", lambda *a: None):
            manager = ModelManager(
                FakeServer(("whisper-small-a16w8",)), catalog_dir=tmp,
                max_resident_chat_models=1, asr_name="whisper-small-a16w8",
                hub=HubConfig(allow_download=False),
                openai_base_url="http://127.0.0.1:9998", warmup=False,
                asr_warmup=False, switch_settle_s=0.0)
            flags = {e["name"]: (e["opensThinking"], e["opensThinkingWhenOff"])
                     for e in manager.catalog()}
        self.assertEqual(flags["r1"], (True, True))
        self.assertEqual(flags["gated"], (True, False))
        self.assertEqual(flags["plain"], (False, False))


def _split(deltas, opens_thinking):
    splitter = cli._ThinkSplitter(opens_thinking)
    pieces = []
    for delta in deltas:
        pieces += splitter.feed(delta)
    pieces += splitter.flush()
    kinds = []
    for kind, text in pieces:
        if kinds and kinds[-1][0] == kind:
            kinds[-1] = (kind, kinds[-1][1] + text)
        else:
            kinds.append((kind, text))
    return kinds


class CliOpenedThinkingTests(unittest.TestCase):
    def test_reasoning_is_shown_as_reasoning_from_the_first_delta(self):
        splitter = cli._ThinkSplitter(True)
        self.assertEqual(splitter.feed("Okay, so"), [("think", "Okay, so")])

    def test_long_reasoning_never_turns_into_an_answer(self):
        reasoning = "step " * (cli._ThinkSplitter.HOLD_CHARS // 2)
        self.assertEqual(_split([reasoning, "</think>", "\n\nThe answer is 391."], True),
                         [("think", reasoning), ("answer", "\n\nThe answer is 391.")])

    def test_a_reply_cut_off_while_thinking_has_no_answer(self):
        self.assertEqual(_split(["still ", "thinking"], True), [("think", "still thinking")])
        self.assertEqual(cli._answer_only("still thinking", opens_thinking=True), "")

    def test_without_the_flag_the_old_guess_is_kept(self):
        self.assertEqual(cli._answer_only("x" * 400), "x" * 400)
