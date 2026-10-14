#!/usr/bin/env python3
"""wrapper.py -- B1: the schedule model (Sequence B).

Pure, independently testable code: stdlib only, no world import, no
numpy. It consumes door observations (tick, door_open) as they arrive
and maintains an estimate of the door's periodic schedule.

Schedule math: the door is open at tick t iff
    (t - last_opening) % period < open_len
where last_opening is the tick of the latest *observed* closed->open
transition. On a stable schedule openings land exactly on
t % period == 0.

Events
------
An opening is recorded only when a closed->open transition is
*observed* between two consecutive ticks; likewise a closing on an
observed open->closed transition. The first state seen at an episode
start can therefore never be treated as a real opening (see
new_episode()).

Clean vs gapped spans
---------------------
Observations may have gaps (e.g. across episode boundaries, when the
model did not watch every tick). A gap means a transition could have
been missed, so an opening-to-opening gap spanning a missed interval
may be a multiple of the true period, and an opening-to-closing span
with a gap may skip the true closing. The model tracks whether the
span since the last opening was observed without gaps or episode
boundaries ("clean"). Estimates use clean spans only:
- period: the most recent clean opening-to-opening gap.
- open_len: the most recent clean opening-to-closing pair
  (closing_tick - opening_tick).
This keeps a missed opening across an episode boundary from ever
becoming a wrong period: a dirty gap is simply not trusted.

Estimates (all None until observed transitions support them)
------------------------------------------------------------
- period, open_len, last_opening_tick (latest observed opening tick).

Across episodes
---------------
new_episode() clears only the pending previous observation (and marks
the span gapped); events and estimates persist, so a model built over
earlier episodes predicts a new episode's phase from tick 0. Gaps the
model did not observe are filled by periodic projection, which is
exact on a stable schedule.

Consistency check
-----------------
Every observation is compared against the model's prediction *before*
it can form a transition. A mismatch means the world changed under the
model: observe() returns True, the stale events are dropped, and the
model re-estimates from subsequent (recent) events. Until new
transitions arrive the estimates are honestly None rather than wrong.

Time features (None while the schedule is unknown)
--------------------------------------------------
- ticks_since_opening(tick): (tick - last_opening) % period -- the
  cycle phase, exact for any tick by periodic projection.
- open_ticks_remaining(tick): open_len - phase while the door is open,
  0 while it is closed.
"""

_OPEN = True
_CLOSED = False


class ScheduleModel(object):
    """Estimates a periodic door schedule from (tick, door_open)
    observations."""

    def __init__(self, max_events=32):
        self.max_events = max_events
        # (tick, diff_to_prev_or_None, clean_diff)
        self.openings = []
        # (tick, paired_opening_tick_or_None, clean_span)
        self.closings = []
        self.period = None
        self.open_len = None
        self.last_opening_tick = None
        self.contradictions = 0
        self.last_contradiction_tick = None
        self._prev = None      # (tick, door_open) of previous observation
        self._last_tick = None
        self._gap_since_opening = False

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def new_episode(self):
        """Mark an episode boundary.

        Clears only the pending previous observation and marks the span
        gapped, so the next observation cannot form a transition: the
        first state seen at an episode start is never treated as a real
        opening. Events and estimates persist across episodes.
        """
        self._prev = None
        self._gap_since_opening = True

    def observe(self, tick, door_open):
        """Feed one observation. Returns True iff it contradicted the
        model (stale events dropped, re-estimating from recent events),
        else False."""
        door_open = bool(door_open)
        if self._prev is not None:
            if tick <= self._prev[0]:
                # New episode (or a repeated tick): the next observation
                # cannot form a transition.
                self._prev = None
                self._gap_since_opening = True
            elif tick > self._prev[0] + 1:
                # Skipped ticks: a transition could have been missed.
                self._gap_since_opening = True
        contradicted = False
        if self._has_estimates() and self.predict_open(tick) != door_open:
            # The world changed under the model. Any transition at this
            # tick would span the change, so it is dropped along with
            # the stale events; the model rebuilds from subsequent
            # observations.
            contradicted = True
            self.contradictions += 1
            self.last_contradiction_tick = tick
            self.openings = []
            self.closings = []
            self._gap_since_opening = True
            self._reestimate()
        elif self._prev is not None:
            prev_tick, prev_open = self._prev
            if tick == prev_tick + 1 and prev_open != door_open:
                if door_open:
                    self._record_opening(tick)
                else:
                    self._record_closing(tick)
                self._reestimate()
        self._prev = (tick, door_open)
        self._last_tick = tick
        return contradicted

    def _record_opening(self, tick):
        prev = self.openings[-1][0] if self.openings else None
        if prev is None:
            self.openings.append((tick, None, True))
        else:
            clean = not self._gap_since_opening
            self.openings.append((tick, tick - prev, clean))
        self._trim(self.openings)
        self._gap_since_opening = False

    def _record_closing(self, tick):
        paired = self.openings[-1][0] if self.openings else None
        clean = (not self._gap_since_opening) if paired is not None \
            else False
        self.closings.append((tick, paired, clean))
        self._trim(self.closings)

    # ------------------------------------------------------------------
    # estimates
    # ------------------------------------------------------------------
    def _trim(self, events):
        while len(events) > self.max_events:
            del events[0]

    def _has_estimates(self):
        return (self.period is not None and self.open_len is not None
                and self.last_opening_tick is not None)

    def _reestimate(self):
        self.period = None
        self.open_len = None
        self.last_opening_tick = (self.openings[-1][0] if self.openings
                                  else None)
        for _t, diff, clean in reversed(self.openings):
            if clean and diff is not None:
                self.period = diff
                break
        for _c, _o, clean in reversed(self.closings):
            if clean and _o is not None:
                self.open_len = _c - _o
                break

    # ------------------------------------------------------------------
    # time features
    # ------------------------------------------------------------------
    def predict_open(self, tick):
        """Predicted door state at tick, or None if unknown."""
        if not self._has_estimates():
            return None
        return ((tick - self.last_opening_tick) % self.period) \
            < self.open_len

    def ticks_since_opening(self, tick=None):
        """Cycle phase at tick (default: last observed tick), or None
        if unknown. Exact for any tick by periodic projection."""
        if not self._has_estimates():
            return None
        if tick is None:
            tick = self._last_tick
        if tick is None:
            return None
        return (tick - self.last_opening_tick) % self.period

    def open_ticks_remaining(self, tick=None):
        """Open ticks remaining at tick (default: last observed tick):
        open_len - phase while open, 0 while closed; None if unknown."""
        if not self._has_estimates():
            return None
        if tick is None:
            tick = self._last_tick
        if tick is None:
            return None
        phase = (tick - self.last_opening_tick) % self.period
        return self.open_len - phase if phase < self.open_len else 0

    def __repr__(self):
        return ("ScheduleModel(period=%r, open_len=%r, last_opening=%r, "
                "contradictions=%d)"
                % (self.period, self.open_len, self.last_opening_tick,
                   self.contradictions))
