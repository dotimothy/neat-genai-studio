"""CLI reasoning support: the <think> stream splitter and the /no_think rewrite.

Loads cli/main.py as a module (it only runs under __main__), no server needed.
"""

from __future__ import annotations

import importlib.util
import unittest
import unittest.mock
from pathlib import Path

CLI_MAIN = Path(__file__).resolve().parents[2] / "src" / "python" / "cli" / "main.py"
_spec = importlib.util.spec_from_file_location("studio_cli_main", CLI_MAIN)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _split(deltas):
    splitter = cli._ThinkSplitter()
    pieces = []
    for delta in deltas:
        pieces += splitter.feed(delta)
    pieces += splitter.flush()
    merged = []
    for kind, text in pieces:
        if merged and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1] + text)
        else:
            merged.append((kind, text))
    return merged


class ThinkSplitterTests(unittest.TestCase):
    def test_reasoning_block_is_separated_from_the_answer(self):
        self.assertEqual(
            _split(["<think>", "let me ", "reason", "</think>", "The answer."]),
            [("think", "let me reason"), ("answer", "The answer.")])

    def test_tags_split_across_deltas_are_resolved(self):
        self.assertEqual(_split(["<thi", "nk>abc</th", "ink>xyz"]),
                         [("think", "abc"), ("answer", "xyz")])

    def test_a_lone_angle_bracket_is_not_swallowed(self):
        self.assertEqual(_split(["plain ", "answer <", "3 done"]),
                         [("answer", "plain answer <3 done")])
        # ... also once the hold has been released.
        splitter = cli._ThinkSplitter()
        splitter.feed("x" * cli._ThinkSplitter.HOLD_CHARS)
        pieces = splitter.feed("a <") + splitter.feed("3 b") + splitter.flush()
        self.assertEqual("".join(t for k, t in pieces if k == "answer"), "a <3 b")

    def test_unterminated_reasoning_is_flushed_as_reasoning(self):
        self.assertEqual(_split(["<think>unterminated"]), [("think", "unterminated")])

    def test_template_injected_close_within_the_hold_is_shown_as_reasoning(self):
        # No <think> was emitted (the runtime put it in the prompt). While the
        # undecided prefix is still held, nothing was shown as an answer, so
        # the reasoning is emitted as reasoning and no reclassify is needed.
        self.assertEqual(
            _split(["step one ", "step two</think>", " Final answer."]),
            [("think", "step one step two"), ("answer", " Final answer.")])

    def test_prefix_is_held_until_the_hold_limit_then_released_as_answer(self):
        splitter = cli._ThinkSplitter()
        short = "x" * (cli._ThinkSplitter.HOLD_CHARS - 1)
        self.assertEqual(splitter.feed(short), [])            # still undecided
        released = splitter.feed("yy")                          # crosses the hold
        self.assertEqual("".join(t for k, t in released if k == "answer"), short + "yy")
        self.assertEqual(splitter.feed("more"), [("answer", "more")])

    def test_template_injected_close_after_the_hold_reclassifies(self):
        # A reasoning trace longer than the hold streamed as an answer; the
        # bare </think> then tells the caller to reclassify what it showed.
        long_reasoning = "reasoning " * 40
        pieces = _split([long_reasoning, "</think>", " Final answer."])
        kinds = [k for k, _ in pieces]
        self.assertIn("reclassify", kinds)
        marker = kinds.index("reclassify")
        self.assertEqual("".join(t for _, t in pieces[:marker]), long_reasoning)
        self.assertEqual(pieces[marker + 1:], [("answer", " Final answer.")])

    def test_short_plain_answer_is_flushed_at_the_end(self):
        self.assertEqual(_split(["Yes."]), [("answer", "Yes.")])

    def test_multiple_blocks(self):
        self.assertEqual(
            _split(["a<think>b</think>c<think>d</think>e"]),
            [("answer", "a"), ("think", "b"), ("answer", "c"), ("think", "d"), ("answer", "e")])


class _FakeResponse:
    def __init__(self, deltas):
        self._lines = [b'data: {"choices":[{"delta":{"content":' + __import__("json").dumps(d).encode() + b'}}]}\n'
                       for d in deltas] + [b"data: [DONE]\n"]

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class StreamTokenCountTests(unittest.TestCase):
    """stream_chat counts incoming deltas, whatever the splitter buffers."""

    def _stream(self, deltas, think=True):
        import io, contextlib
        from unittest import mock
        with mock.patch.object(cli.urllib.request, "urlopen", return_value=_FakeResponse(deltas)):
            with contextlib.redirect_stdout(io.StringIO()):
                return cli.stream_chat(("127.0.0.1", 1), "m", [{"role": "user", "content": "q"}],
                                       64, render=False, think=think)

    def test_short_plain_answer_counts_every_delta(self):
        text, _, _, tokens, reasoning = self._stream(["ab", "cd", "ef", "gh", "ij"])
        self.assertEqual((text, tokens, reasoning), ("abcdefghij", 5, 0))

    def test_long_answer_counts_held_and_released_deltas(self):
        deltas = ["word "] * 60                      # 300 chars, well past the hold
        text, _, _, tokens, reasoning = self._stream(deltas)
        self.assertEqual(text, "word " * 60)
        self.assertEqual((tokens, reasoning), (60, 0))

    def test_reasoning_block_is_counted_separately(self):
        deltas = ["<think>", "let ", "me ", "think", "</think>", "The ", "answer."]
        text, _, _, tokens, reasoning = self._stream(deltas)
        self.assertEqual(text, "The answer.")
        self.assertEqual((tokens, reasoning), (2, 5))

    def test_close_only_reasoning_within_the_hold_is_counted_as_reasoning(self):
        deltas = ["step ", "one ", "step ", "two", "</think>", " Final."]
        text, _, _, tokens, reasoning = self._stream(deltas)
        self.assertEqual(text.strip(), "Final.")
        self.assertEqual((tokens, reasoning), (1, 5))


class ResetDisconnectClassificationTests(unittest.TestCase):
    def test_only_a_mid_reply_disconnect_means_success(self):
        import http.client
        self.assertTrue(cli._is_reset_disconnect(http.client.RemoteDisconnected("gone")))
        self.assertTrue(cli._is_reset_disconnect(ConnectionResetError()))
        self.assertFalse(cli._is_reset_disconnect(ConnectionRefusedError()))
        self.assertFalse(cli._is_reset_disconnect(OSError("network unreachable")))


class NoThinkRewriteTests(unittest.TestCase):
    def test_last_user_text_turn_gets_the_switch_without_mutating_input(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
                {"role": "user", "content": "again"}]
        out = cli._without_thinking(msgs)
        self.assertEqual(out[-1]["content"], "again /no_think")
        self.assertEqual(out[1]["content"], "hi")           # earlier turns untouched
        self.assertEqual(msgs[-1]["content"], "again")      # input not mutated

    def test_multimodal_turn_appends_to_its_text_part(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "see"},
                                             {"type": "image", "image": "x"}]}]
        out = cli._without_thinking(msgs)
        self.assertEqual(out[0]["content"][0]["text"], "see /no_think")
        self.assertEqual(msgs[0]["content"][0]["text"], "see")

    def test_image_only_turn_gains_a_text_part(self):
        msgs = [{"role": "user", "content": [{"type": "image", "image": "x"}]}]
        out = cli._without_thinking(msgs)
        self.assertEqual(out[0]["content"][-1], {"type": "text", "text": "/no_think"})


class SeveralModelsTests(unittest.TestCase):
    """The pieces behind /max, /use, /compare and /pick."""

    STATUS = {
        "resident": ["model-b", "model-a"], "maxResident": 2,
        "catalog": [{"name": "model-a", "loaded": True}, {"name": "model-b", "loaded": True},
                    {"name": "whisper", "loaded": True, "type": "asr"}],
        "mla": {"totalBytes": 16 << 30, "claimedBytes": 12 << 30, "usedBytes": 3 << 30},
    }

    def status(self, **changes):
        return unittest.mock.patch.object(cli, "ctrl_get", return_value=dict(self.STATUS, **changes))

    def test_loaded_models_come_most_recently_used_first(self):
        with self.status():
            self.assertEqual(cli.loaded_chat_models(None), ["model-b", "model-a"])

    def test_an_older_server_without_the_order_still_lists_loaded_chat_models(self):
        with self.status(resident=None):
            self.assertEqual(cli.loaded_chat_models(None), ["model-a", "model-b"])   # no ASR
        with unittest.mock.patch.object(cli, "ctrl_get", side_effect=OSError("down")):
            self.assertEqual(cli.loaded_chat_models(None), [])

    def test_residency_line_reports_the_limit_and_the_measured_memory(self):
        with self.status():
            line = cli.residency_line(None)
        self.assertIn("2 of max 2 models loaded", line)
        self.assertIn("12 GB of 16 GB held by the runtime", line)
        with self.status(mla={"totalBytes": 16 << 30, "claimedBytes": None, "usedBytes": 3 << 30}):
            self.assertIn("(estimate)", cli.residency_line(None))

    def test_reasoning_is_removed_from_a_collected_reply(self):
        self.assertEqual(cli._answer_only("<think>let me see</think>The answer."), "The answer.")
        self.assertEqual(cli._answer_only("Just an answer."), "Just an answer.")
        self.assertEqual(cli._answer_only("<think>never closed"), "")

    def test_a_text_only_model_is_sent_the_turn_without_its_image(self):
        messages = [{"role": "system", "content": "be brief"},
                    {"role": "user", "content": [{"type": "text", "text": "what is this"},
                                                 {"type": "image", "image": "data:..."}]}]
        stripped = cli._without_images(messages)
        self.assertEqual(stripped[0], messages[0])
        self.assertEqual(stripped[1], {"role": "user", "content": "what is this"})
        self.assertIsInstance(messages[1]["content"], list)        # the original is untouched

    def test_reply_stats_line(self):
        self.assertEqual(cli.reply_stats(70, 0.25, 30.0), "70 tok  ·  ttft 250ms  ·  30.0 tok/s")
        self.assertEqual(cli.reply_stats(5, None, None, 12), "5 tok (+12 reasoning)")
        self.assertEqual(cli.reply_stats(0, None, None), "")


class LiveStatusLineTests(unittest.TestCase):
    """The animated status lines. Under the test runner stdout is not a
    terminal, so colour is off and the drawing can be compared as plain text."""

    def test_bar_fills_in_proportion_and_keeps_its_width(self):
        self.assertEqual(cli.progress_bar(0, width=8), "░" * 8)
        self.assertEqual(cli.progress_bar(50, width=8), "████░░░░")
        self.assertEqual(cli.progress_bar(100, width=8), "████████")
        for pct in (0, 3, 37.5, 99, 100, 250, -5):
            self.assertEqual(len(cli.progress_bar(pct, frame=7, width=24)), 24)

    def test_bar_edge_moves_in_eighths_of_a_cell(self):
        # 8 cells: 6.25% is half of the first cell.
        self.assertEqual(cli.progress_bar(6.25, width=8), "▌" + "░" * 7)
        self.assertEqual(cli.progress_bar(56.25, width=8), "████▌░░░")

    def test_bar_without_a_percentage_sweeps(self):
        frames = {cli.progress_bar(None, frame=f, width=20) for f in range(40)}
        self.assertGreater(len(frames), 5)
        self.assertTrue(all(len(f) == 20 and "█" in f for f in frames))

    def test_spinner_cycles_through_its_frames(self):
        seen = [cli.spinner_frame(i) for i in range(len(cli._SPIN_FRAMES))]
        self.assertEqual("".join(seen), cli._SPIN_FRAMES)
        self.assertEqual(cli.spinner_frame(len(cli._SPIN_FRAMES)), seen[0])

    def test_a_line_is_cut_to_the_terminal_width_keeping_colour_codes(self):
        coloured = "\x1b[38;2;1;2;3mabcdef\x1b[0mghij"
        self.assertEqual(cli._fit(coloured, 20), coloured)             # fits: untouched
        cut = cli._fit(coloured, 6)
        self.assertEqual(cli._visible_len(cut), 6)
        self.assertTrue(cli._ANSI_RE.sub("", cut).startswith("abcde"))
        self.assertIn("\x1b[38;2;1;2;3m", cut)
        self.assertEqual(cli._visible_len("\x1b[2mhi\x1b[0m"), 2)

    def test_load_line_shows_the_bar_scale_and_countdown(self):
        line = cli._load_progress_line(
            {"pct": 50, "stagesTotal": 182, "elapsedS": 3, "remainingS": 3})
        self.assertIn("50%", line)
        self.assertIn("182 stages", line)
        self.assertIn("~3s left", line)
        self.assertIn("█", line)
        done = cli._load_progress_line({"pct": 99, "elapsedS": 9, "remainingS": 0})
        self.assertIn("finishing…", done)
        counted = cli._load_progress_line({"pct": 10, "filesDone": 4, "filesTotal": 40})
        self.assertIn("stage 4/40", counted)

    def test_unload_line_fills_against_a_measured_time_or_sweeps(self):
        timed = cli._unload_progress_line("model-a", 4.0, 1.0)
        self.assertIn("25%", timed)
        self.assertIn("unloading model-a", timed)
        self.assertIn("~3s left", timed)
        late = cli._unload_progress_line("model-a", 4.0, 9.0)
        self.assertIn("99%", late)                      # never claims to be finished
        self.assertIn("finishing…", late)
        unknown = cli._unload_progress_line("model-a", None, 2.0)
        self.assertNotIn("%", unknown)
        self.assertNotIn("left", unknown)
        self.assertIn("unloading model-a", unknown)
        self.assertIn("2s", unknown)

    def test_load_victims_are_the_least_recently_used_beyond_the_limit(self):
        status = {"resident": ["b", "a"], "maxResident": 2,
                  "catalog": [{"name": "a", "estimatedUnloadS": 1.5}, {"name": "b"}]}
        with unittest.mock.patch.object(cli, "ctrl_get", return_value=status):
            self.assertEqual(cli._load_victims(None, "c"), [("a", 1.5)])
            self.assertEqual(cli._load_victims(None, "a"), [])      # already loaded
        with unittest.mock.patch.object(cli, "ctrl_get", return_value=dict(status, maxResident=3)):
            self.assertEqual(cli._load_victims(None, "c"), [])      # there is room
        with unittest.mock.patch.object(cli, "ctrl_get", return_value=dict(status, maxResident=1)):
            self.assertEqual(cli._load_victims(None, "c"), [("b", None), ("a", 1.5)])
        with unittest.mock.patch.object(cli, "ctrl_get", side_effect=OSError("down")):
            self.assertEqual(cli._load_victims(None, "c"), [])

    def test_nothing_is_animated_when_output_is_not_a_terminal(self):
        calls = []
        live = cli.LiveLine(lambda frame, elapsed: calls.append(frame) or "x")
        self.assertFalse(live.active)
        with live:
            pass
        live.stop()                                    # stopping twice is harmless
        self.assertEqual(calls, [])
        with cli.spinner("waiting…") as spin:
            self.assertFalse(spin.active)


if __name__ == "__main__":
    unittest.main()
