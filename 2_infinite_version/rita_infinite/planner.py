"""Small receding-horizon search over new continuous joint-space trajectories.

No finite motion primitive is loaded here. The finite rhythm grammar is shared;
joint amplitudes, phases, directions and knot values are sampled afresh online.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, fields
import copy
import math
import time
from typing import Any

import numpy as np

from rita_core.contracts import BoundaryState
from rita_core.rhythm import RHYTHMS
from rita_core.trajectory import make_segment, validate_segment, descriptor, descriptor_distance


@dataclass
class InfiniteConfig:
    candidate_count: int = 24
    horizon: int = 2
    recent_window: int = 16
    novelty_threshold: float = 0.025
    novelty_factors: tuple = (1.0, 0.75, 0.5, 0.25, 0.0)
    metrical_levels: tuple = (0, 1, 2, 3)
    min_feasible_choices: int = 3
    max_validated_candidates: int = 8
    faster_level_hysteresis: int = 2
    planning_budget_seconds: float = 1.5
    boundary_tolerance: float = 1e-6
    amplitude_cell_threshold: float = 0.18
    single_rhythm: bool = False
    no_coverage_novelty: bool = False
    stop_to_stop: bool = False
    terminal_hold_seconds: float = 0.0
    force_failure: bool = False
    force_timeout: bool = False
    # Opt-in variety: search every level and alternate between feasible ones.
    level_mix: bool = False
    level_mix_max_run: int = 2
    amplitude_scales: tuple = (1.0, .85, .70)

    @classmethod
    def from_value(cls, value):
        if value is None:
            result = cls()
        elif isinstance(value, cls):
            result = copy.deepcopy(value)
        elif isinstance(value, dict):
            known = {f.name for f in fields(cls)}
            unknown = set(value)-known
            if unknown:
                raise ValueError(f"Unknown infinite settings: {sorted(unknown)}")
            result = cls(**value)
        else:
            raise TypeError("Infinite config must be a dict or InfiniteConfig")
        if not 16 <= result.candidate_count <= 32:
            raise ValueError("candidate_count must be 16..32")
        if not 1 <= result.horizon <= 4:
            raise ValueError("horizon must be 1..4 phrases")
        if result.recent_window < 1 or result.novelty_threshold < 0:
            raise ValueError("Invalid novelty configuration")
        result.metrical_levels = tuple(int(k) for k in result.metrical_levels)
        if not result.metrical_levels or any(k not in range(4) for k in result.metrical_levels):
            raise ValueError("metrical_levels must be a nonempty subset of 0..3")
        result.metrical_levels = tuple(sorted(set(result.metrical_levels)))
        result.novelty_factors = tuple(float(x) for x in result.novelty_factors)
        result.amplitude_scales = tuple(float(x) for x in result.amplitude_scales)
        if (not result.amplitude_scales or any(not 0 < x <= 1 for x in result.amplitude_scales)
                or tuple(sorted(result.amplitude_scales, reverse=True)) != result.amplitude_scales):
            raise ValueError("amplitude_scales must be decreasing values in (0,1]")
        if result.level_mix_max_run < 1:
            raise ValueError("level_mix_max_run must be positive")
        if (not result.novelty_factors or result.novelty_factors[-1] != 0
                or any(x < 0 or x > 1 for x in result.novelty_factors)
                or tuple(sorted(result.novelty_factors, reverse=True)) != result.novelty_factors):
            raise ValueError("novelty_factors must decrease from <=1 to 0")
        if result.planning_budget_seconds <= 0 or result.min_feasible_choices < 1:
            raise ValueError("Planning budget and min_feasible_choices must be positive")
        if not result.min_feasible_choices <= result.max_validated_candidates <= result.candidate_count:
            raise ValueError("max_validated_candidates must be between min_feasible_choices and candidate_count")
        if result.terminal_hold_seconds < 0 or not np.isfinite(result.terminal_hold_seconds):
            raise ValueError("terminal_hold_seconds must be finite and nonnegative")
        if result.terminal_hold_seconds and not result.stop_to_stop:
            raise ValueError("terminal_hold_seconds requires stop_to_stop=true")
        return result


class InfinitePlanningError(RuntimeError):
    """No certified plan remains: the executor must use its adapter stop path."""

    def __init__(self, reason, diagnostic=None):
        self.reason = str(reason)
        self.stop_required = True
        self.diagnostic = diagnostic or {}
        super().__init__(f"Infinite planner: {reason}; adapter stop required")


class InfinitePlanner:
    def __init__(self, analysis, profile, seed=7, config=None):
        self.analysis, self.profile = analysis, profile
        self.config = InfiniteConfig.from_value(config)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.nj = len(np.asarray(profile.neutral))
        self.span = np.asarray(profile.upper)-np.asarray(profile.lower)
        if self.nj < 1 or np.any(self.span <= 0):
            raise ValueError("Invalid robot joint ranges")
        self.groups = [np.arange(self.nj)[np.arange(self.nj) % 2 == i] for i in range(min(2, self.nj))]
        self.cells = [f"g{g}:{sign}:{tier}" for g in range(len(self.groups))
                      for sign in ("+", "-") for tier in ("low", "high")]
        self.recent = deque(maxlen=self.config.recent_window)
        self.service = deque()  # Only cells with an actual validated witness enter.
        self.witnesses = {}
        self.visits = Counter()
        self.rhythm_visits = Counter()
        self._suffix = []
        self._level = None
        self._faster_streak = 0
        self._mix_rng = np.random.default_rng(int(seed)+101)
        self._mix_uses, self._mix_run = Counter(), 0
        self._serial = 0
        self._calls = 0
        self._metrics = Counter()
        self._planning_times = []
        self._last_service = {}
        self._max_service_interval = 0
        self._last = {}
        self._previous_committed = None

    def _state_key(self, start_beat, state):
        arrays = np.concatenate([np.asarray(state.q), np.asarray(state.dq), np.asarray(state.ddq)])
        return (int(start_beat), tuple(np.round(arrays, 9)), str(self.profile.fingerprint()))

    def _check_boundary(self, start_beat, state):
        expected = float(self.analysis.beat_time(start_beat))
        if abs(float(state.t)-expected) > self.config.boundary_tolerance:
            raise ValueError(f"Boundary time {state.t} differs from beat {start_beat} time {expected}")
        if int(start_beat) != start_beat or int(start_beat) % 4:
            raise ValueError("Infinite phrases must start on a beat-origin 4-beat block")
        for name in ("q", "dq", "ddq"):
            value = np.asarray(getattr(state, name), dtype=float)
            if value.shape != (self.nj,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Boundary {name} must have {self.nj} finite values in profile joint order")
        if np.any(state.q < self.profile.lower) or np.any(state.q > self.profile.upper):
            raise InfinitePlanningError("actual_boundary_outside_profile", {"candidate_search_failed": True})
        if self._previous_committed is not None:
            old = self._previous_committed.end_state()
            if (abs(state.t-old.t) > self.config.boundary_tolerance
                    or any(not np.allclose(getattr(state, k), getattr(old, k), atol=1e-7, rtol=0)
                           for k in ("q", "dq", "ddq"))):
                raise InfinitePlanningError("boundary_does_not_match_reserved_state",
                                            {"suffix_valid": False, "recovery_required": True})

    def _rhythm(self, serial, phrase):
        if self.config.single_rhythm:
            return RHYTHMS[0]
        # Least-served rhythms remain available, without selecting motion IDs.
        order = sorted(range(len(RHYTHMS)), key=lambda i: (self.rhythm_visits[RHYTHMS[i].id], i))
        return RHYTHMS[order[(serial+phrase) % len(order)]]

    def _latent(self, cell):
        group_name, sign_name, tier = cell.split(":")
        group = int(group_name[1:])
        sign = 1 if sign_name == "+" else -1
        amplitudes = self.rng.uniform(.067, .085, self.nj) if tier == "low" else self.rng.uniform(.145, .185, self.nj)
        # Group balance is a geometric feature, measured again after validation.
        mask = np.ones(self.nj, dtype=bool)
        mask[self.groups[group]] = False
        amplitudes[mask] *= self.rng.uniform(.28, .48)
        phases = self.rng.uniform(-.075, .075, self.nj)
        phases[mask] += self.rng.uniform(-.16, .16, np.count_nonzero(mask))
        return {"amplitude": amplitudes, "phase": phases,
                "secondary_phase": self.rng.uniform(-math.pi, math.pi, self.nj),
                "sign": sign, "harmonic": self.rng.uniform(.12, .28),
                "drift": self.rng.uniform(-.025, .025, self.nj)}

    def _control_filter(self, seg):
        controls = np.asarray(seg.controls)
        durations = np.diff(np.asarray(seg.knots))[:, None, None]
        if not np.all(np.isfinite(controls)) or np.any(durations <= 0):
            return False
        first = 5*np.diff(controls, axis=1)/durations
        second = 20*np.diff(controls, n=2, axis=1)/(durations*durations)
        tol = 1e-8
        return bool(np.all(controls >= np.asarray(self.profile.lower)-tol)
                    and np.all(controls <= np.asarray(self.profile.upper)+tol)
                    and np.all(np.abs(first) <= np.asarray(self.profile.vmax)+tol)
                    and np.all(np.abs(second) <= np.asarray(self.profile.amax)+tol))

    def _make_horizon(self, start_beat, state, level, latent, serial, scale=1.0):
        width = 4*(2**level)
        available_phrases = int((len(self.analysis.beats)-1-start_beat+1e-8)//width)
        horizon = min(self.config.horizon, available_phrases)
        if horizon < 1:
            return None
        all_beats, targets, phrase_ranges, patterns = [float(start_beat)], [np.asarray(state.q).copy()], [], []
        neutral = np.asarray(self.profile.neutral)
        lo = np.asarray(self.profile.lower)+.09*self.span
        hi = np.asarray(self.profile.upper)-.09*self.span
        for h in range(horizon):
            rhythm = self._rhythm(serial, h)
            patterns.append(rhythm)
            events = np.asarray(rhythm.events, dtype=float)
            offsets = sorted(set([float(x) for x in events if x > 1e-9] + [4.0]))
            begin = len(all_beats)-1
            for off in offsets:
                x = h+off/4
                ph = latent["phase"] + .018*h*latent["drift"]
                shape = np.sin(math.tau*(x+ph)) + latent["harmonic"]*np.sin(2*math.tau*x+latent["secondary_phase"])
                shape /= 1+latent["harmonic"]
                amp = latent["amplitude"]*scale
                # An accent rest suppresses the accent contribution, not every joint.
                event_idx = int(np.argmin(np.abs(events-off))) if len(events) else -1
                is_rest = event_idx in set(getattr(rhythm, "rests", ())) and off < 4
                strengths = tuple(rhythm.strengths)
                strength = float(strengths[event_idx % len(strengths)]) if strengths else 1.0
                gain = .70+.30*strength
                if is_rest:
                    gain = .60
                q = neutral+latent["sign"]*amp*self.span*shape*gain
                q += latent["drift"]*self.span*math.sin(math.pi*x)
                q = np.clip(q, lo, hi)
                all_beats.append(float(start_beat+h*width+off*(2**level)))
                targets.append(q)
            phrase_ranges.append((begin, len(all_beats)-1))
        # The retained horizon ends in a certified stationary neutral boundary.
        targets[-1] = neutral.copy()
        knots = np.array([self.analysis.beat_time(x) for x in all_beats], dtype=float)
        q = np.asarray(targets)
        velocity = np.zeros_like(q)
        accel = np.zeros_like(q)
        velocity[0] = state.dq
        accel[0] = state.ddq
        if len(q) > 2:
            left = np.diff(q, axis=0)/np.diff(knots)[:, None]
            velocity[1:-1] = .40*(left[:-1]+left[1:])
            velocity[1:-1] = np.clip(velocity[1:-1], -.32*np.asarray(self.profile.vmax), .32*np.asarray(self.profile.vmax))
        if self.config.stop_to_stop:
            for _, end in phrase_ranges:
                velocity[end] = 0
        segments = []
        for h, ((begin, end), rhythm) in enumerate(zip(phrase_ranges, patterns)):
            events = np.asarray(rhythm.events, dtype=float)
            rests = set(getattr(rhythm, "rests", ()))
            accent_times = [float(self.analysis.beat_time(start_beat+h*width+off*(2**level)))
                            for j, off in enumerate(events) if j not in rests]
            strengths = [float(rhythm.strengths[j]) for j in range(len(events)) if j not in rests]
            metadata = {"primitive_id": f"continuous-{self._serial+1:07d}-{serial:02d}-{h}",
                        "generation_kind": "continuous", "start_beat": int(start_beat+h*width),
                        "end_beat": int(start_beat+(h+1)*width), "horizon": horizon,
                        "horizon_requested": self.config.horizon,
                        "accent_event_type": "designated_waypoint_passage",
                        "accent_rest_indices": sorted(rests), "projection_scale": float(scale),
                        "terminal_condition": "neutral_zero_velocity_acceleration" if h == horizon-1 else "C2_reserved_boundary",
                        "stop_to_stop": self.config.stop_to_stop,
                        "future_audio_lookahead_seconds": float(knots[-1]-knots[0]),
                        "fallback": False}
            seg = make_segment(knots[begin:end+1], q[begin:end+1],
                               velocities=velocity[begin:end+1], accelerations=accel[begin:end+1],
                               accent_times=accent_times, accent_strengths=strengths,
                               rhythm_id=rhythm.id, metrical_level=level, metadata=metadata)
            if self.config.terminal_hold_seconds:
                from rita_core.trajectory import add_terminal_hold
                try:
                    seg = add_terminal_hold(seg, self.config.terminal_hold_seconds)
                except ValueError:
                    return None
                seg.metadata["terminal_hold_seconds"] = self.config.terminal_hold_seconds
            if not self._control_filter(seg):
                return None
            segments.append(seg)
        terminal = segments[-1].end_state()
        if (np.max(np.abs(terminal.q-neutral)) > 1e-7
                or np.max(np.abs(terminal.dq)) > 1e-7 or np.max(np.abs(terminal.ddq)) > 1e-7):
            return None
        return segments

    def _feature_cell(self, seg):
        times = np.linspace(seg.start, seg.end, 25)
        path = np.array([seg.at(float(t))[0] for t in times])
        excursion = np.ptp(path, axis=0)/self.span
        group = int(np.argmax([np.mean(excursion[g]) for g in self.groups]))
        delta = (path[6]-path[0])/self.span
        sign = "+" if np.sum(delta[self.groups[group]]) >= 0 else "-"
        tier = "high" if np.max(excursion) >= self.config.amplitude_cell_threshold else "low"
        return f"g{group}:{sign}:{tier}"

    def _distance(self, desc):
        return min((float(descriptor_distance(desc, old)) for old in self.recent), default=1.0)

    def _candidate_score(self, seg, novelty):
        # Geometry novelty and measured audio accent agreement are separate terms.
        if self.config.no_coverage_novelty:
            novelty = 0.0
        audio_match = 0.0
        try:
            strengths = np.asarray(self.analysis.accent, dtype=float)
            beats = np.asarray(self.analysis.beats, dtype=float)
            if len(beats) == len(strengths) and len(beats):
                sampled = np.interp(seg.accent_times, beats, strengths)
                audio_match = float(np.mean(sampled*np.asarray(seg.accent_strengths)))
        except (AttributeError, TypeError, ValueError):
            pass
        controls = np.asarray(seg.controls)
        smooth_cost = float(np.mean(np.diff(controls, n=3, axis=1)**2))
        rhythm_bonus = 1/(1+self.rhythm_visits[seg.rhythm_id])
        return novelty + .12*audio_match + .08*rhythm_bonus-.05*smooth_cost

    def _search_level(self, start_beat, state, level, started):
        provisional = []
        target = self.service[0] if self.service and not self.config.no_coverage_novelty else None
        for serial in range(self.config.candidate_count):
            if time.perf_counter()-started >= self.config.planning_budget_seconds:
                raise TimeoutError("candidate search exceeded the wall-clock planning budget")
            # Half of proposals pursue the persistent oldest cell. The rest explore.
            cell = target if target is not None and serial % 2 == 0 else self.cells[(serial+self._calls) % len(self.cells)]
            latent = self._latent(cell)
            plan = None
            for scale in self.config.amplitude_scales:
                plan = self._make_horizon(start_beat, state, level, latent, serial, scale)
                if plan:
                    break
            self._metrics["candidate_proposals"] += 1
            if plan is None:
                continue
            first = plan[0]
            desc = descriptor(first, self.profile)
            novelty = self._distance(desc)
            actual_cell = self._feature_cell(first)
            provisional.append({"plan": plan, "descriptor": desc, "distance": novelty,
                                "cell": actual_cell, "score": self._candidate_score(first, novelty)})
        provisional.sort(key=lambda x: (0 if x["cell"] == target else 1 if not self.visits[x["cell"]] else 2,
                                         -x["score"]))
        # Exact quartic-root activity checks are reserved for a small, geometrically
        # varied finalist pool. All 16..32 proposals were generated continuously.
        counts, accepted, deferred = Counter(), [], []
        ordered = []
        for candidate in provisional:
            if counts[candidate["cell"]] < 2:
                ordered.append(candidate)
                counts[candidate["cell"]] += 1
            else:
                deferred.append(candidate)
        ordered.extend(deferred)
        for candidate in ordered:
            if time.perf_counter()-started >= self.config.planning_budget_seconds:
                raise TimeoutError("finalist certification exceeded the wall-clock planning budget")
            previous, valid = self._previous_committed, True
            for seg in candidate["plan"]:
                report = validate_segment(seg, self.profile, previous=previous)
                self._metrics["exact_segment_validations"] += 1
                if not report.get("valid", False):
                    valid = False
                    break
                seg.validation = report
                previous = seg
            if valid:
                accepted.append(candidate)
                if len(accepted) >= self.config.max_validated_candidates:
                    break
        return accepted

    def _choose_level(self, candidates):
        if self.config.level_mix:
            return self._mix_level(candidates)
        enough = [k for k, c in candidates.items() if len(c) >= self.config.min_feasible_choices]
        possible = enough or [k for k, c in candidates.items() if c]
        if not possible:
            return None
        fastest = min(possible)
        if self._level is not None and fastest < self._level:
            self._faster_streak += 1
            if self._faster_streak < self.config.faster_level_hysteresis and candidates.get(self._level):
                return self._level
        else:
            self._faster_streak = 0
        return fastest

    def _mix_level(self, candidates):
        """Weighted random over levels with enough validated candidates (least-used
        favoured); one level may not run longer than level_mix_max_run phrases."""
        ok = [k for k, c in candidates.items() if len(c) >= self.config.min_feasible_choices]
        ok = ok or [k for k, c in candidates.items() if c]
        if not ok:
            return None
        current = self._level
        if current in ok and len(ok) > 1 and self._mix_run >= self.config.level_mix_max_run:
            ok.remove(current)
        weights = np.array([1/(1+self._mix_uses[k]) for k in ok])
        chosen = int(self._mix_rng.choice(ok, p=weights/weights.sum()))
        self._mix_run = self._mix_run+1 if chosen == current else 1
        self._mix_uses[chosen] += 1
        return chosen

    def _fallback(self, start_beat, state, reason, started):
        self._metrics["timeouts" if reason == "planning_timeout" else "search_failures"] += 1
        if self._suffix:
            seg = self._suffix[0]
            reserved = seg.start_state()
            same = (abs(seg.start-state.t) <= self.config.boundary_tolerance
                    and seg.metadata.get("start_beat") == int(start_beat)
                    and all(np.allclose(getattr(state, x), getattr(reserved, x), atol=1e-7, rtol=0)
                            for x in ("q", "dq", "ddq")))
            report = validate_segment(seg, self.profile, previous=self._previous_committed) if same else {"valid": False}
            if report.get("valid", False):
                self._suffix = self._suffix[1:]
                result = copy.deepcopy(seg)
                result.metadata.update({"generation_kind": "fallback_suffix", "fallback": True,
                                        "fallback_reason": reason, "planning_seconds": time.perf_counter()-started,
                                        "novelty_success": False, "coverage_success": False,
                                        "reserved_suffix_remaining": len(self._suffix)})
                result.validation = report
                self._previous_committed = result
                self._metrics["fallback_count"] += 1
                self._last = {"reason": reason, "suffix_used": True}
                return result
        self._metrics["stop_required_count"] += 1
        self._suffix = []
        self._last = {"reason": reason, "suffix_used": False}
        raise InfinitePlanningError(reason, {"candidate_search_failed": True,
                                            "mathematical_infeasibility_proven": False,
                                            "suffix_valid": False})

    def next_segment(self, start_beat: int, state: BoundaryState):
        started = time.perf_counter()
        self._calls += 1
        self._check_boundary(start_beat, state)
        try:
            if len(self.analysis.beats)-1-start_beat < 4-1e-8:
                raise InfinitePlanningError("audio_grid_exhausted", {"remaining_beats": len(self.analysis.beats)-1-start_beat})
            if self.config.force_timeout:
                return self._fallback(start_beat, state, "planning_timeout", started)
            if self.config.force_failure:
                return self._fallback(start_beat, state, "candidate_search_failed", started)
            candidates = {}
            try:
                for level in self.config.metrical_levels:
                    candidates[level] = self._search_level(start_beat, state, level, started)
                    enough = len(candidates[level]) >= self.config.min_feasible_choices
                    if enough and not self.config.level_mix and (self._level is None or level >= self._level
                                   or self._faster_streak+1 >= self.config.faster_level_hysteresis):
                        break
            except TimeoutError:
                return self._fallback(start_beat, state, "planning_timeout", started)
            selected_level = self._choose_level(candidates)
            if selected_level is None:
                return self._fallback(start_beat, state, "candidate_search_failed", started)
            pool = candidates[selected_level]
            boundary_key = self._state_key(start_beat, state)
            for candidate in pool:
                cell = candidate["cell"]
                self.witnesses[cell] = {"boundary_key": boundary_key, "call": self._calls,
                                        "validated": True}
                if cell not in self.service:
                    self.service.append(cell)
            target = self.service[0] if self.service else None
            target_pool = [x for x in pool if x["cell"] == target]
            use_pool = target_pool or pool
            if self.config.no_coverage_novelty:
                use_pool = pool
            threshold = 0.0 if self.config.no_coverage_novelty else self.config.novelty_threshold
            chosen = None
            relaxation = 0
            for relaxation, factor in enumerate(self.config.novelty_factors):
                allowed = [x for x in use_pool if x["distance"]+1e-12 >= threshold*factor]
                if self.config.no_coverage_novelty:
                    allowed = use_pool
                if allowed:
                    chosen = max(allowed, key=lambda x: x["score"])
                    break
            if chosen is None:
                return self._fallback(start_beat, state, "candidate_search_failed", started)
            plan, cell = chosen["plan"], chosen["cell"]
            result = plan[0]
            # Recheck the exact post-projection object which is about to be committed.
            report = validate_segment(result, self.profile, previous=self._previous_committed)
            if not report.get("valid", False):
                return self._fallback(start_beat, state, "final_validation_failed", started)
            elapsed = time.perf_counter()-started
            if elapsed > self.config.planning_budget_seconds:
                return self._fallback(start_beat, state, "planning_timeout", started)
            self._serial += 1
            if self._level is not None and selected_level != self._level:
                self._metrics["k_changes"] += 1
            self._level = selected_level
            self._suffix = plan[1:]
            self.recent.append(chosen["descriptor"])
            self.visits[cell] += 1
            self.rhythm_visits[result.rhythm_id] += 1
            if target == cell and self.service:
                self.service.rotate(-1)
            if cell in self._last_service:
                self._max_service_interval = max(self._max_service_interval, self._calls-self._last_service[cell])
            self._last_service[cell] = self._calls
            self._metrics["generated_segments"] += 1
            self._metrics["novelty_relaxations"] += relaxation
            self._metrics["relaxed_segments"] += int(relaxation > 0)
            result.validation = report
            self._previous_committed = result
            result.metadata.update({"primitive_id": f"continuous-{self._serial:07d}",
                                    "planning_seconds": elapsed, "candidate_count": self.config.candidate_count,
                                    "validated_candidates": len(pool), "feature_cell": cell,
                                    "service_target": target, "service_target_witnessed_now": bool(target_pool),
                                    "service_target_served": target == cell,
                                    "novelty_distance": float(chosen["distance"]),
                                    "novelty_threshold_requested": self.config.novelty_threshold,
                                    "novelty_threshold_used": threshold*self.config.novelty_factors[relaxation],
                                    "novelty_relaxation_steps": relaxation,
                                    "novelty_success": not self.config.no_coverage_novelty and chosen["distance"] >= threshold,
                                    "coverage_success": not self.config.no_coverage_novelty,
                                    "novelty_policy_enabled": not self.config.no_coverage_novelty,
                                    "reserved_suffix_remaining": len(self._suffix),
                                    "search_failure_is_not_impossibility": True})
            self._last = dict(result.metadata)
            return result
        finally:
            self._planning_times.append(time.perf_counter()-started)

    def terminal_suffix(self, start_beat: int, state: BoundaryState):
        """Drain the existing certified horizon at session end, without replanning.

        Returns None only if no suffix remains and the boundary is already at
        rest. No timeout/search-failure or normal diversity success is recorded.
        """
        started = time.perf_counter()
        self._check_boundary(start_beat, state)
        if not self._suffix:
            at_rest = (np.max(np.abs(state.dq)) <= 1e-7
                       and np.max(np.abs(state.ddq)) <= 1e-7)
            if at_rest:
                return None
            raise InfinitePlanningError("terminal_suffix_unavailable",
                                        {"suffix_valid": False, "reason": "session_end",
                                         "mathematical_infeasibility_proven": False})
        reserved = self._suffix[0]
        boundary = reserved.start_state()
        matches = (reserved.metadata.get("start_beat") == int(start_beat)
                   and abs(reserved.start-state.t) <= self.config.boundary_tolerance
                   and all(np.allclose(getattr(state, k), getattr(boundary, k), atol=1e-7, rtol=0)
                           for k in ("q", "dq", "ddq")))
        if not matches:
            raise InfinitePlanningError("terminal_suffix_boundary_mismatch",
                                        {"suffix_valid": False, "reason": "session_end"})
        result = copy.deepcopy(reserved)
        report = validate_segment(result, self.profile, previous=self._previous_committed)
        if not report.get("valid", False):
            raise InfinitePlanningError("terminal_suffix_validation_failed",
                                        {"suffix_valid": False, "reason": "session_end",
                                         "validation": report})
        self._suffix = self._suffix[1:]
        result.metadata.update({"generation_kind": "terminal_suffix", "recovery": True,
                                "excluded_from_diversity": True, "reason": "session_end",
                                "novelty_success": False, "coverage_success": False,
                                "fallback": False, "planning_seconds": time.perf_counter()-started,
                                "reserved_suffix_remaining": len(self._suffix)})
        result.validation = report
        self._previous_committed = result
        self._metrics["terminal_suffix_count"] += 1
        self._last = dict(result.metadata)
        return result

    def stats(self) -> dict[str, Any]:
        return {**dict(self._metrics), "calls": self._calls, "seed": self.seed,
                "config": asdict(self.config), "current_metrical_level": self._level,
                "feature_cells_total": len(self.cells), "feature_cells_visited": len(self.visits),
                "feature_cells_ever_witnessed": len(self.witnesses),
                "feature_cell_coverage": len(self.visits)/len(self.cells),
                "unwitnessed_cells": sorted(set(self.cells)-set(self.witnesses)),
                "feature_cell_visits": dict(self.visits), "service_queue": list(self.service),
                "max_observed_cell_service_interval_calls": self._max_service_interval,
                "rhythm_visits": dict(self.rhythm_visits),
                "reserved_suffix_remaining": len(self._suffix),
                "planning_seconds": list(self._planning_times), "last": copy.deepcopy(self._last),
                "coverage_guarantee": "Conditional service of persistently witnessed targets; no random-search finite-time bound.",
                "novelty_guarantee": "Reported recent-window distance at the logged threshold; threshold may relax to zero.",
                "recursive_feasibility": "Only the validated retained suffix and terminal stop are certified; replanning can fail."}
