"""Stopping a reply that repeats itself. The board decodes greedily (it ignores
temperature and the repetition penalties), and DeepSeek-R1-Distill then often
repeats one passage of its reasoning until the token limit. The clients watch
the stream and stop it (shared/repetition.py).
"""

from __future__ import annotations

import contextlib
import io
import random
import unittest
from unittest import mock

from cli_think_suite import _FakeResponse, cli
from shared.repetition import LoopDetector

PASSAGE = ("Wait, maybe I should check that again. 7 times 13 is 7 times 10 plus "
           "7 times 3, which is 70 plus 21. ")


def feed(text, step=7):
    """Stream ``text`` in ``step``-sized deltas; the detector and the number of
    characters it had seen when it fired (None if it never did)."""
    detector = LoopDetector()
    for i in range(0, len(text), step):
        if detector.feed(text[i:i + step]):
            return detector, i + step
    return detector, None


class LoopDetectorTests(unittest.TestCase):
    def test_a_repeated_passage_is_caught_soon_after_its_third_copy(self):
        text = "Okay, let me think. " + PASSAGE * 20
        detector, seen = feed(text)
        self.assertTrue(detector.looped)
        self.assertLess(seen, len("Okay, let me think. ") + 5 * len(PASSAGE))

    def test_keep_ends_with_the_first_copy_whatever_the_delta_size(self):
        text = "Okay, let me think. " + PASSAGE * 20
        for step in (1, 5, 33, 200):
            detector, _ = feed(text, step)
            kept = text[:detector.keep]
            self.assertEqual(kept.count("Wait, maybe"), 1, step)
            self.assertTrue(kept.rstrip(". ").endswith("70 plus 21"), (step, kept[-30:]))

    def test_a_short_phrase_must_repeat_many_times(self):
        self.assertIsNone(feed("So " + "I think so. " * 20)[1])      # 240 chars
        self.assertIsNotNone(feed("So " + "I think so. " * 60)[1])

    def test_ordinary_text_is_left_alone(self):
        rng = random.Random(1)
        words = "the cat sat on a mat and then went home to eat some fish".split()
        prose = " ".join(rng.choice(words) for _ in range(3000))
        table = "| n | n² |\n|---|---|\n" + "".join(f"| {i} | {i * i} |\n" for i in range(300))
        rule = "Title\n" + "=" * 3000 + "\nBody"
        for text in (prose, table, rule, PASSAGE * 2):
            self.assertFalse(feed(text)[0].looped, text[:40])

    def test_nothing_more_is_taken_once_it_fired(self):
        detector, _ = feed(PASSAGE * 10)
        self.assertTrue(detector.feed("more"))
        self.assertTrue(detector.looped)


class CliLoopGuardTests(unittest.TestCase):
    def test_the_active_reply_is_stopped_and_the_server_told(self):
        deltas = ["Okay. "] + [PASSAGE] * 40 + ["</think>", "never reached"]
        stops = []
        with mock.patch.object(cli.urllib.request, "urlopen", return_value=_FakeResponse(deltas)), \
                mock.patch.object(cli, "_http", side_effect=lambda url, data=None, **kw: stops.append((url, data))), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            text, *_ = cli.stream_chat(("127.0.0.1", 1), "r1", [{"role": "user", "content": "q"}],
                                       8192, render=False, opens_thinking=True)
        self.assertEqual(text, "")
        self.assertEqual(stops, [("http://127.0.0.1:1/stop", {"model": "r1"})])
        self.assertIn("kept repeating", out.getvalue())

    def test_a_background_reply_keeps_one_copy(self):
        deltas = ["Here is the answer. "] + [PASSAGE] * 40
        job = cli.BackgroundReply(("127.0.0.1", 1), "m", [{"role": "user", "content": "q"}], 8192)
        with mock.patch.object(cli.urllib.request, "urlopen", return_value=_FakeResponse(deltas)), \
                mock.patch.object(job, "cancel") as cancel:
            job.run()
        self.assertTrue(job.looped)
        cancel.assert_called_once()
        self.assertEqual(job.raw.count("Wait, maybe"), 1)
        self.assertTrue(job.raw.startswith("Here is the answer. "))


if __name__ == "__main__":
    unittest.main()
