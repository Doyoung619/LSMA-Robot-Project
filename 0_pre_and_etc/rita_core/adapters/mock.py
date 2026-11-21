"""No-network adapter with deterministic delayed tracking and failure injection."""
import time
from bisect import bisect_right
import numpy as np
from ..contracts import RobotState
from .base import RobotAdapter, AdapterError, DispatchDeadlineMiss, require_segment


class MockAdapter(RobotAdapter):
    def __init__(self, profile, *, clock=time.monotonic, command_delay=0., tracking_delay=0.,
                 feedback_dropout=0., disconnect_at=None, seed=7, execute=False, **kwargs):
        if kwargs:
            raise TypeError(f"Unknown mock options: {sorted(kwargs)}")
        if command_delay < 0 or tracking_delay < 0 or not 0 <= feedback_dropout <= 1:
            raise ValueError("Mock delays must be nonnegative; dropout must be in [0,1]")
        self.profile = profile
        self.clock = clock
        self.command_delay = float(command_delay)
        self.tracking_delay = float(tracking_delay)
        self.feedback_dropout = float(feedback_dropout)
        self.disconnect_at = disconnect_at
        self.rng = np.random.default_rng(seed)
        self.connected = False
        self.stopped = False
        self.history = []
        self.commands = []
        self._virtual_now = None
        self._last_state = None
        self._hold = np.array(profile.neutral, dtype=float)
        self.stop_count = 0

    def now(self):
        return float(self.clock() if self._virtual_now is None else self._virtual_now)

    def advance(self, t):
        """Set absolute virtual monotonic seconds (not a sleep)."""
        if self._virtual_now is not None and t < self._virtual_now-1e-9:
            raise ValueError("Mock virtual clock cannot move backwards")
        self._virtual_now = float(t)

    set_music_time = advance

    def connect(self):
        self.connected = True
        self.stopped = False
        return self.get_capabilities()

    def _check(self):
        if not self.connected:
            raise AdapterError("Mock is disconnected")
        if self.disconnect_at is not None and self.now() >= self.disconnect_at:
            self.connected = False
            raise ConnectionError("Injected mock communication loss")

    def get_capabilities(self):
        return {"backend": "mock", "hardware": False, "joint_names": list(self.profile.joint_names),
                "units": "rad", "feedback_freshness": self.profile.feedback_freshness,
                "command_mode": "mock", "command_period": self.profile.command_period,
                "queue": True, "cancel": True, "future_execution": True, "streaming_c2": True,
                "reference_exact": self.command_delay == 0 and self.tracking_delay == 0,
                "stop_semantics": "simulated immediate hold, not a hardware braking guarantee",
                "command_delay": self.command_delay, "tracking_delay": self.tracking_delay,
                "feedback_dropout": self.feedback_dropout}

    def _target(self, now):
        delayed = now-self.command_delay-self.tracking_delay
        index = bisect_right(self.commands, delayed, key=lambda item:item[0])-1
        # Match the original subtraction order at floating-point boundaries.
        def elapsed_at(i):
            return now-self.commands[i][0]-self.command_delay-self.tracking_delay
        if index+1 < len(self.commands) and elapsed_at(index+1)>=0:
            index+=1
        if index>=0 and elapsed_at(index)<0:
            index-=1
        if index < 0:
            return self._hold.copy(), np.zeros_like(self._hold)
        # Preserve the earlier command's exact end at a shared boundary, even
        # for deliberately discontinuous adapter-contract fixtures.
        if index > 0:
            previous_start, previous_segment = self.commands[index-1]
            if elapsed_at(index-1) <= previous_segment.duration:
                index -= 1
        start,segment=self.commands[index]
        elapsed=now-start-self.command_delay-self.tracking_delay
        q,dq,_=segment.at(min(segment.end,segment.start+max(0.,elapsed)))
        return np.asarray(q),np.asarray(dq)

    def read_state(self):
        self._check()
        now = self.now()
        if self.rng.random() < self.feedback_dropout:
            previous = self._last_state
            return RobotState(t=previous.t if previous else now-self.profile.feedback_freshness-1,
                              q=previous.q.copy() if previous else self._hold.copy(),
                              dq=previous.dq.copy() if previous else np.zeros_like(self._hold),
                              joint_names=self.profile.joint_names, units="rad", fresh=False,
                              metadata={"backend": "mock", "injected_feedback_missing": True})
        q, dq = self._target(now)
        self._last_state = RobotState(t=now, q=q, dq=dq, joint_names=self.profile.joint_names,
                                      units="rad", fresh=True,
                                      metadata={"backend": "mock", "measurement": "simulated delayed reference"})
        return self._last_state

    def execute_segment(self, segment, scheduled_monotonic=None, now_music=None):
        self._check()
        require_segment(segment, self.profile)
        now = self.now()
        if scheduled_monotonic is None:
            scheduled_monotonic = now + (max(0, segment.start-now_music) if now_music is not None else 0)
        if scheduled_monotonic < now-self.profile.command_period:
            raise DispatchDeadlineMiss("Expired mock segment; catch-up dispatch is forbidden")
        if self.commands and scheduled_monotonic < self.commands[-1][0]+self.commands[-1][1].duration-1e-7:
            raise AdapterError("Mock queue overlaps an already reserved segment")
        self.commands.append((float(scheduled_monotonic), segment))
        self.stopped = False
        result = {"sent_monotonic": now, "accepted_monotonic": now+self.command_delay,
                  "scheduled_monotonic": float(scheduled_monotonic), "backend": "mock", "completed": False}
        self.history.append(result)
        return result

    def stop(self):
        if self.connected:
            self._hold = self._target(self.now())[0].copy()
        self.commands.clear()
        self.stopped = True
        self.stop_count += 1
        return {"backend": "mock", "stop_requested": True, "queue_cleared": True}

    def close(self):
        self.stop()
        self.connected = False
