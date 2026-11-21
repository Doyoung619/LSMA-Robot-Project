"""Finite rhythm grammar. Blocks have beat-origin, not detected downbeat, phase."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Rhythm:
    id: str
    intervals: tuple
    strengths: tuple
    rests: tuple = ()

    @property
    def events(self): return tuple(np.r_[0., np.cumsum(self.intervals)[:-1]])
    @property
    def durations(self): return self.intervals


RHYTHMS = [
    Rhythm("straight", (1,1,1,1), (1,.5,.8,.5)),
    Rhythm("dotted", (1.5,.5,1,1), (1,.7,.9,.4)),
    Rhythm("anticipation", (.5,1.5,1,1), (.6,1,.8,.5)),
    Rhythm("backbeat", (1,1,1,1), (.4,1,.5,1)),
    Rhythm("split_two", (1,.5,.5,1,1), (1,.5,.8,.6,.9)),
    Rhythm("split_four", (1,1,1,.5,.5), (1,.5,.8,.4,1)),
    Rhythm("syncopated", (.5,1,.5,1,1), (.8,1,.4,.9,.5)),
    Rhythm("five_rest", (.5,.5,1,1,1), (1,.7,0,.9,.4), (2,)),
    Rhythm("double_pair", (.5,.5,1,.5,.5,1), (1,.5,.8,.7,1,.4)),
    Rhythm("offbeats", (.5,1,.5,.5,1,.5), (.5,1,.6,.8,1,.4)),
    Rhythm("six_rest", (1,.5,.5,.5,.5,1), (1,0,.7,1,.5,.8), (1,)),
    Rhythm("eighths", (.5,)*8, (1,.4,.8,.4,1,.4,.7,.4)),
    Rhythm("eighth_rest", (.5,)*8, (1,0,.8,.5,1,0,.7,.4), (1,5)),
]
assert len(RHYTHMS) >= 12
assert all(abs(sum(r.intervals)-4) < 1e-9 and len(r.intervals) == len(r.strengths) for r in RHYTHMS)


def event_times(analysis, start_beat, rhythm, level=0):
    return np.array([analysis.beat_time(start_beat+x*2**level) for x in rhythm.events])
