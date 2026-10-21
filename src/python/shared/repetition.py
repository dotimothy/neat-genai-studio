"""Spotting a reply that has fallen into a loop.

The SiMa runtime decodes greedily: it ignores temperature, top_p, seed and the
repetition penalties. Small reasoning models (DeepSeek-R1-Distill) then often
lock into one passage and repeat it until the token limit, which DeepSeek's
model card says to avoid by sampling at temperature 0.6. With no sampling to
turn on, the clients watch the stream instead and stop a reply once its tail is
the same passage back to back.

Greedy loops repeat the same tokens, so the text repeats exactly; nothing here
tries to match near-repeats.
"""

from __future__ import annotations


class LoopDetector:
    """Feed a reply's streamed text; ``feed`` returns True once the reply ends
    in one passage repeated back to back.

    A loop is a passage of MIN_PERIOD..MAX_PERIOD characters whose last
    MIN_REPEATS or more copies are identical and together cover MIN_SPAN
    characters: a sentence has to come back several times, a paragraph three
    times. The passage must contain words, so a long rule of ``=`` or a run of
    table borders is not a loop. ``keep`` is then the length of the text up to
    the end of the first copy, for callers that drop the repeats.
    """

    MIN_PERIOD = 8
    MAX_PERIOD = 2000
    MIN_REPEATS = 3
    MIN_SPAN = 400
    CHECK_EVERY = 32      # characters between checks
    KEY = 24              # tail length used to find candidate periods
    MAX_CANDIDATES = 64

    def __init__(self):
        self.text = ""
        self.looped = False
        self.keep = None
        self._unchecked = 0

    def feed(self, delta: str) -> bool:
        if self.looped or not delta:
            return self.looped
        self.text += delta
        self._unchecked += len(delta)
        if self._unchecked >= self.CHECK_EVERY:
            self._unchecked = 0
            self._check()
        return self.looped

    def _check(self) -> None:
        window = self.MAX_PERIOD * self.MIN_REPEATS
        t = self.text[-window:]
        offset = len(self.text) - len(t)
        n = len(t)
        if n < self.MIN_SPAN:
            return
        key = t[-self.KEY:]
        end = n - self.KEY        # search before the key's own position
        for _ in range(self.MAX_CANDIDATES):
            i = t.rfind(key, 0, end + self.KEY - 1)
            if i < 0:
                return
            period = end - i
            end = i
            if period < self.MIN_PERIOD:
                continue
            if period > self.MAX_PERIOD:
                return
            unit = t[n - period:]
            if sum(ch.isalpha() for ch in unit) < 4:
                continue
            reps = 1
            while (reps + 1) * period <= n and t[n - (reps + 1) * period:n - reps * period] == unit:
                reps += 1
            if reps >= self.MIN_REPEATS and reps * period >= self.MIN_SPAN:
                # The repeats need not start on a whole copy counted from the
                # end: walk back to where the text first turns periodic.
                start = n - reps * period
                while start > 0 and t[start - 1] == t[start - 1 + period]:
                    start -= 1
                self.looped = True
                self.keep = offset + start + period
                return
