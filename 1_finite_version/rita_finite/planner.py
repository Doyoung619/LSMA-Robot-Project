"""Finite, frozen geometry; timing certificates; persistent fair service queue."""
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import time
import numpy as np

from rita_core.contracts import DanceSegment
from rita_core.rhythm import RHYTHMS
from rita_core.trajectory import make_segment, validate_segment, descriptor, descriptor_distance
from .graph import MacroAction, CertifiedGraph


class FinitePlanningError(RuntimeError):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(f"{status}: {detail}")


DEFAULTS = {"library_size":48,"amplitude_fraction":.18,"min_motion_choices":32,
            "min_rhythm_choices":12,"hysteresis_blocks":3,"duplicate_distance":.012,
            "max_metrical_level":3,"no_tempo_adaptation":False,
            "single_rhythm":False,"no_coverage":False,"terminal_hold_seconds":0.,
            # Opt-in variety: alternate metrical levels instead of always taking the
            # fastest sufficient one, optionally with a smaller motion at faster levels.
            "level_mix":False,"level_amplitude":None,"level_mix_min_actions":6,"level_mix_max_run":2}


class FinitePlanner:
    def __init__(self, analysis, profile, seed=7, config=None):
        began = time.perf_counter()
        self.analysis, self.profile = analysis, profile
        if isinstance(config,(str,Path)):
            config = json.loads(Path(config).read_text(encoding="utf-8"))
        self.config = dict(DEFAULTS)
        self.config.update(config or {})
        size = int(self.config["library_size"])
        if size < 32 or size > 64:
            raise ValueError("finite library_size must be between 32 and 64")
        if not 0 < self.config["amplitude_fraction"] <= 1:
            raise ValueError("amplitude_fraction must be in (0,1]")
        hold = float(self.config["terminal_hold_seconds"])
        if not np.isfinite(hold) or hold<0:
            raise ValueError("terminal_hold_seconds must be finite and nonnegative")
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self._level_ratio = {}
        if self.config["level_amplitude"]:
            for key,value in dict(self.config["level_amplitude"]).items():
                k,value = int(key),float(value)
                if k not in range(4) or not 0 < value <= 1:
                    raise ValueError("level_amplitude maps levels 0..3 to amplitude fractions in (0,1]")
                self._level_ratio[k] = value/float(self.config["amplitude_fraction"])
        self._mix_rng = np.random.default_rng(seed+101)
        self._mix_uses, self._mix_run, self._mix_beat, self._prepare_beat = Counter(), 0, None, None
        payload = json.loads(Path(__file__).with_name("library.json").read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(json.dumps(payload["primitives"],sort_keys=True).encode()).hexdigest()
        if actual_hash != payload["geometry_sha256"]:
            raise ValueError("frozen library geometry hash mismatch; run offline authoring intentionally")
        self.library_hash = actual_hash
        self.primitives = {p["id"]:p for p in payload["primitives"][:size]}
        self.rhythms = {r.id:r for r in RHYTHMS}
        selected_rhythm = self.config["single_rhythm"]
        if selected_rhythm:
            rid = selected_rhythm if isinstance(selected_rhythm,str) else "straight"
            if rid not in self.rhythms:
                raise ValueError(f"unknown single rhythm {rid}")
            self.rhythms = {rid:self.rhythms[rid]}
        margin = np.minimum(profile.upper-profile.neutral,profile.neutral-profile.lower)
        self.mapping_scale = margin*float(self.config["amplitude_fraction"])
        self.geometry, self.midpoints, self.geometry_descriptors, self.projected_duplicates = {}, {}, {}, {}
        for pid, primitive in self.primitives.items():
            raw = np.asarray(primitive["normalized_waypoints"],dtype=float)
            geometry = profile.neutral+raw[:,np.arange(profile.nj)%raw.shape[1]]*self.mapping_scale
            geometry.flags.writeable = False
            self.geometry[pid] = geometry
            midpoint_raw = np.asarray(primitive["normalized_midpoints"],dtype=float)
            midpoint = profile.neutral+midpoint_raw[:,np.arange(profile.nj)%midpoint_raw.shape[1]]*self.mapping_scale
            midpoint.flags.writeable = False
            self.midpoints[pid] = midpoint
            canonical = make_segment(np.linspace(0,1,len(geometry)),geometry)
            desc = descriptor(canonical,profile)
            duplicate = next((other for other,od in self.geometry_descriptors.items()
                              if descriptor_distance(desc,od)<self.config["duplicate_distance"]),None)
            if duplicate is None:
                self.geometry_descriptors[pid] = desc
            else:
                self.projected_duplicates[pid] = duplicate
        self._context = None
        self._graph = None
        self._component = set()
        self._segments = {}
        self._all_level_candidates = {}
        self._current_level = None
        self._level_desire = None
        self._level_streak = 0
        self._current = None
        self._last_segment = None
        self._bag, self._path, self._goal = [], deque(), None
        self._bag_start, self._bag_actions = None, 0
        self._bags = []
        self._recent = deque(maxlen=16)
        self._motion_visits, self._rhythm_visits = Counter(), Counter()
        self._context_motion_visits, self._context_rhythm_visits = Counter(), Counter()
        self._contexts = []
        self._certificate_resets = 0
        self._k_changes = 0
        self._recoveries = 0
        self._failures = []
        self._latencies = []
        self._candidate_statuses = Counter()
        self._certified_motion_ids = set()
        self._entry_nodes = set()
        self._base_certificate = None
        self._lazy_edges_checked = 0
        self._cached_actions = {}
        # Finite geometry mapping, duplicate checks and initial graph/timing
        # certification happen before playback, rather than in an audio callback.
        self._prepare(0, None)
        self.preparation_seconds = time.perf_counter()-began

    def _timing_context(self, start_beat):
        remaining = len(self.analysis.beats)-1-start_beat
        count = min(32,remaining)
        offsets = np.arange(0,count+.01,.5)
        points = np.array([self.analysis.beat_time(start_beat+x) for x in offsets])
        # Translation is excluded, beat duration/phase structure is included.
        data = {"intervals":np.round(np.diff(points),9).tolist(),
                "profile":self.profile.fingerprint(),"library":self.library_hash,
                "size":len(self.primitives),"scale":self.mapping_scale.tolist(),
                "latency_version":self.config.get("latency_version",1),
                "terminal_hold_seconds":float(self.config["terminal_hold_seconds"]),
                "rhythms":sorted(self.rhythms)}
        return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

    def _make_action(self, action, start_beat):
        rhythm = self.rhythms[action.rhythm_id]
        geometry = self.geometry[action.primitive_id]
        midpoints = self.midpoints[action.primitive_id]
        ratio = self._level_ratio.get(action.metrical_level,1.)
        if ratio != 1.:
            # Same stored chords, uniformly scaled about neutral for this level.
            geometry = self.profile.neutral+(geometry-self.profile.neutral)*ratio
            midpoints = self.profile.neutral+(midpoints-self.profile.neutral)*ratio
        if len(rhythm.intervals) != len(geometry)-1:
            raise ValueError("rhythm event count cannot alter frozen primitive geometry")
        factor = 2**action.metrical_level
        end_beat = start_beat+4*factor
        offsets = tuple(rhythm.events)+(4.,)
        event_knots = [self.analysis.beat_time(start_beat+x*factor) for x in offsets]
        # Midpoints are frozen in JSON, not inserted/generated at runtime.
        # Rhythm strength changes their arrival TIME, shaping real approach
        # velocity while traversing the identical stored geometric chords.
        knots,waypoints,velocities = [],[],[]
        for i in range(len(geometry)-1):
            left,right = event_knots[i:i+2]
            strength = rhythm.strengths[(i+1)%len(rhythm.strengths)]
            fraction = .44+.12*strength
            knots.extend((left,left+fraction*(right-left)))
            waypoints.extend((geometry[i],midpoints[i]))
            velocities.extend((np.zeros(self.profile.nj),1.875*(geometry[i+1]-geometry[i])/(right-left)))
        knots.append(event_knots[-1])
        waypoints.append(geometry[-1])
        velocities.append(np.zeros(self.profile.nj))
        keep = [i for i,strength in enumerate(rhythm.strengths)
                if i not in rhythm.rests and strength>0]
        segment = make_segment(knots,waypoints,velocities=velocities,
            accent_times=[event_knots[i] for i in keep],
            accent_strengths=[rhythm.strengths[i] for i in keep],
            rhythm_id=rhythm.id,metrical_level=action.metrical_level,
            metadata={"mode":"finite","primitive_id":action.primitive_id,
                      "rhythm_id":rhythm.id,"metrical_level":action.metrical_level,
                      "start_beat":start_beat,"end_beat":end_beat,
                      "frozen_geometry_sha256":self.library_hash,
                      "accent_definition":"arrival/turning at a stored joint-pose knot; rest suppresses accent, not body motion",
                      "strength_expression":"stored midpoint timing changes actual approach-speed profile; positions stay frozen",
                      "geometry_generated_online":False,"recovery":False,"planning_seconds":0.})
        hold = float(self.config["terminal_hold_seconds"])
        if hold:
            from rita_core.trajectory import add_terminal_hold
            segment = add_terminal_hold(segment,hold)
        return segment

    @staticmethod
    def _shift(seg, delta, beat_delta=0):
        metadata = dict(seg.metadata)
        for key in ("start_beat","end_beat"):
            if key in metadata:
                metadata[key] += beat_delta
        return DanceSegment(seg.start+delta,seg.duration,seg.knots+delta,
                            seg.controls.copy(),seg.accent_times+delta,
                            seg.accent_strengths.copy(),seg.rhythm_id,
                            seg.metrical_level,metadata,dict(seg.validation))

    def _collect(self, start_beat, timing_key):
        if timing_key in self._cached_actions:
            reference_start, reference_beat, levels, statuses = self._cached_actions[timing_key]
            delta = self.analysis.beat_time(start_beat)-reference_start
            shifted = {k:{a:self._shift(s,delta,start_beat-reference_beat) for a,s in group.items()}
                       for k,group in levels.items()}
            return shifted, statuses.copy()
        maximum = min(3,int(self.config["max_metrical_level"]))
        levels = {k:{} for k in range(maximum+1)}
        statuses = Counter()
        validation_cache = {}
        for k in levels:
            if self.config["no_tempo_adaptation"] and k:
                continue
            if start_beat+4*2**k > len(self.analysis.beats)-1:
                statuses["horizon_unavailable"] += 1
                continue
            for pid,primitive in self.primitives.items():
                if pid in self.projected_duplicates:
                    statuses["projected_duplicate_excluded"] += 1
                    continue
                for rid in primitive["allowed_rhythms"]:
                    if rid not in self.rhythms:
                        continue
                    action = MacroAction(pid,rid,k)
                    try:
                        seg = self._make_action(action,start_beat)
                    except ValueError:
                        # A controller's terminal hold may consume a fast
                        # action's final piece. Keep searching slower levels;
                        # this candidate failure is not global impossibility.
                        statuses["timing_construction_rejected"] += 1
                        continue
                    cache_key = (pid,tuple(self.rhythms[rid].intervals),tuple(self.rhythms[rid].strengths),k)
                    if cache_key in validation_cache:
                        result = dict(validation_cache[cache_key])
                        seg.validation = result
                    else:
                        result = validate_segment(seg,self.profile)
                        validation_cache[cache_key] = result
                    if result["valid"]:
                        levels[k][action] = seg
                        statuses["certified_macro_actions"] += 1
                    else:
                        statuses["rejected_by_sufficient_certificate"] += 1
        if len(self._cached_actions)<16:
            self._cached_actions[timing_key] = (self.analysis.beat_time(start_beat),start_beat,levels,statuses.copy())
        return levels,statuses

    def _enough(self, actions):
        motions = len({a.primitive_id for a in actions})
        rhythms = len({a.rhythm_id for a in actions})
        return (motions >= min(int(self.config["min_motion_choices"]),len(self.geometry_descriptors)) and
                rhythms >= min(int(self.config["min_rhythm_choices"]),len(self.rhythms)))

    def _mix_level(self, levels):
        """Weighted random over feasible levels: least-used levels are favoured and
        one level may not run longer than level_mix_max_run blocks if another fits."""
        current = self._current_level
        if self._prepare_beat == self._mix_beat and current is not None and levels.get(current):
            return current
        ok = [k for k,a in levels.items() if len(a) >= int(self.config["level_mix_min_actions"])]
        ok = ok or [k for k,a in levels.items() if a]
        if not ok:
            return None
        if current in ok and len(ok) > 1 and self._mix_run >= int(self.config["level_mix_max_run"]):
            ok.remove(current)
        weights = np.array([1/(1+self._mix_uses[k]) for k in ok])
        chosen = int(self._mix_rng.choice(ok,p=weights/weights.sum()))
        self._mix_run = self._mix_run+1 if chosen == current else 1
        self._mix_uses[chosen] += 1
        self._mix_beat = self._prepare_beat
        if current is not None and current != chosen:
            self._k_changes += 1
        self._current_level = chosen
        return chosen

    def _select_level(self, levels):
        if self.config["level_mix"]:
            chosen = self._mix_level(levels)
            if chosen is not None:
                return chosen
        enough = [k for k,actions in levels.items() if actions and self._enough(actions)]
        available = [k for k,actions in levels.items() if actions]
        if not available:
            if np.max(self.profile.upper-self.profile.lower)<self.profile.min_excursion:
                raise FinitePlanningError("proven_infeasible", "position box is narrower than required excursion in every joint")
            raise FinitePlanningError("candidate_search_failed", "no stored motion/rhythm at k=0..3 passed sufficient certificates; this is not a proof of physical impossibility")
        wanted = min(enough) if enough else max(available,key=lambda k:(
            len({a.primitive_id for a in levels[k]}),len({a.rhythm_id for a in levels[k]}),-k))
        current = self._current_level
        if current is None or not levels.get(current) or wanted>current:
            chosen = wanted
        elif wanted<current and self._enough(levels[current]):
            self._level_streak = self._level_streak+1 if self._level_desire == wanted else 1
            self._level_desire = wanted
            chosen = wanted if self._level_streak>=int(self.config["hysteresis_blocks"]) else current
        else:
            chosen = wanted
            self._level_streak = 0
        if current is not None and current != chosen:
            self._k_changes += 1
        self._current_level = chosen
        return chosen

    def _connection(self, first, second):
        # Individual continuous certificates plus exact C2 boundary and the
        # tail+head low-speed bound certify the whole concatenation. No dense
        # sampling or hypothetical unseen edge is treated as a certificate.
        a,b = first.end_state(),second.start_state()
        error = max(float(np.max(np.abs(a.q-b.q))),float(np.max(np.abs(a.dq-b.dq))),
                    float(np.max(np.abs(a.ddq-b.ddq))))
        low = first.validation["tail_low_seconds"]+second.validation["head_low_seconds"]
        valid = first.validation["valid"] and second.validation["valid"] and error<=1e-6 and low<=self.profile.max_stationary_seconds+1e-8
        return valid,{"boundary_C2_error":error,"joined_low_seconds":low,
                      "individual_continuous_certificates":True,
                      "timing_context":self._context,
                      "time_placement":"successor translated exactly to predecessor.end; reusable only under identical timing context"}

    def _prepare(self, start_beat, previous):
        timing_key = self._timing_context(start_beat)
        levels,statuses = self._collect(start_beat,timing_key)
        self._prepare_beat = start_beat
        k = self._select_level(levels)
        context = f"{timing_key}:k{k}"
        self._all_level_candidates = levels
        self._candidate_statuses = statuses
        self._certified_motion_ids = {a.primitive_id for group in levels.values() for a in group}
        if self._context == context:
            self._segments = levels[k]
            return
        reset_reason = "initial_certificate" if self._context is None else "timing_profile_latency_or_metrical_context_changed"
        if self._context is not None:
            self._certificate_resets += 1
        self._context = context
        self._segments = levels[k]
        graph = CertifiedGraph(context)
        for action,seg in self._segments.items():
            graph.add_node(action,seg.duration)
        hub = min(self._segments,key=lambda a:(
            self._segments[a].validation["head_low_seconds"]+
            self._segments[a].validation["tail_low_seconds"],a.key))
        for action in sorted(self._segments):
            for source,target in {(action,action),(action,hub),(hub,action)}:
                valid,certificate = self._connection(self._segments[source],self._segments[target])
                graph.check_edge(source,target,valid,certificate)
        entries = set(self._segments)
        if previous is not None:
            entries = {a for a in entries if self._connection(previous,self._segments[a])[0]}
        component = graph.choose_component(entries)
        if not component:
            raise FinitePlanningError("candidate_search_failed", "no enterable recurrent SCC in the explicitly checked sparse graph")
        self._graph,self._component,self._entry_nodes = graph,component,entries
        self._base_certificate = graph.coverage_certificate(component)
        self._current = None
        self._path.clear()
        self._bag.clear()
        self._goal = None
        self._bag_start = None
        self._context_motion_visits.clear()
        self._context_rhythm_visits.clear()
        self._contexts.append({"context":context,"reason":reset_reason,
                               "start_beat":start_beat,"metrical_level":k,
                               "feasible_motion_count":len(self._certified_motion_ids),
                               "scc_motion_count":len({a.primitive_id for a in component})})

    def _new_bag(self, now):
        goals = [("motion",m) for m in sorted({a.primitive_id for a in self._component})]
        goals += [("rhythm",r) for r in sorted({a.rhythm_id for a in self._component})]
        self.rng.shuffle(goals)
        self._bag = goals
        self._bag_start,self._bag_actions = now,0
        self._goal = None

    def _score(self, action):
        seg = self._segments[action]
        # Music accent and repeat avoidance only break equally short routes.
        if len(seg.accent_times):
            musical = np.interp(seg.accent_times,self.analysis.beats,self.analysis.accent)
            accent = float(np.mean(musical*seg.accent_strengths))
        else:
            accent = 0.
        history = list(self._recent)
        repeats = sum(x.primitive_id == action.primitive_id for x in history[-4:])
        if len(history)>1:
            repeats += 2*sum(history[i].primitive_id == history[-1].primitive_id and
                             history[i+1].primitive_id == action.primitive_id
                             for i in range(len(history)-1))
        return .1*accent-repeats

    def _service(self, now):
        if self._path:
            return self._path.popleft()
        if self.config["no_coverage"]:
            candidates = sorted(self._component if self._current is None else
                                self._graph.adjacency[self._current]&self._component)
            if self._current is None:
                candidates = [a for a in candidates if a in self._entry_nodes]
            return candidates[int(self.rng.integers(len(candidates)))]
        if not self._bag:
            self._new_bag(now)
        self._goal = self._bag[0]
        kind,identifier = self._goal
        targets = {a for a in self._component if
                   (a.primitive_id if kind == "motion" else a.rhythm_id) == identifier}
        if self._current is not None:
            # Certify just the relevant shortcut candidates. The fixed sparse
            # hub graph remains a valid base certificate; adding safe edges
            # cannot increase its diameter or postpone a persistent goal.
            for target in sorted(targets):
                if self._graph.edge_status(self._current,target) == "unknown":
                    valid,certificate = self._connection(self._segments[self._current],self._segments[target])
                    self._graph.check_edge(self._current,target,valid,certificate)
                    self._lazy_edges_checked += 1
        if self._current is None:
            targets &= self._entry_nodes
            if not targets:
                # Entry path is a certified path into the selected SCC; it does
                # not replace or discard the pending visitation goal.
                entry = max(sorted(self._entry_nodes&self._component),key=self._score)
                route = [entry]
                route += self._graph.shortest_service_path(entry,
                    {a for a in self._component if (a.primitive_id if kind == "motion" else a.rhythm_id)==identifier},
                    self._component,self._score)
            else:
                route = self._graph.shortest_service_path(None,targets,self._component,self._score)
        else:
            route = self._graph.shortest_service_path(self._current,targets,self._component,self._score)
        self._path = deque(route)
        return self._path.popleft()

    def _recovery(self, start_beat, state, began):
        # Explicit entry/recovery is allowed new waypoint geometry and is never
        # counted as a finite library motion or diversity/novelty success.
        recovery_profile = replace(self.profile,min_excursion=1e-10,
                                   max_stationary_seconds=1e6)
        for k in range(4):
            end_beat = start_beat+4*2**k
            if end_beat>len(self.analysis.beats)-1:
                continue
            end = self.analysis.beat_time(end_beat)
            seg = make_segment([state.t,end],[state.q,self.profile.neutral],
                    velocities=[state.dq,np.zeros(self.profile.nj)],
                    accelerations=[state.ddq,np.zeros(self.profile.nj)],
                    metrical_level=k,rhythm_id="recovery",
                    metadata={"mode":"finite","primitive_id":"__entry_recovery__",
                              "rhythm_id":"recovery","metrical_level":k,
                              "start_beat":start_beat,"end_beat":end_beat,
                              "recovery":True,"excluded_from_diversity":True,
                              "activity_policy":"entry/recovery exception; q/dq/ddq limits still certified"})
            if validate_segment(seg,recovery_profile)["valid"]:
                self._recoveries += 1
                self._certificate_resets += 1
                self._context = None
                self._current = None
                self._path.clear()
                self._bag.clear()
                self._last_segment = seg
                seg.metadata["planning_seconds"] = time.perf_counter()-began
                return seg
        raise FinitePlanningError("candidate_search_failed", "no certified entry/recovery from the supplied measured boundary; stop is required")

    def next_segment(self, start_beat, state):
        began = time.perf_counter()
        if int(start_beat) != start_beat or int(start_beat)%4:
            raise ValueError("finite macro-actions must start on beat-origin 4-beat block boundaries")
        start_beat = int(start_beat)
        now = self.analysis.beat_time(start_beat)
        if abs(state.t-now)>1e-6:
            raise ValueError("supplied boundary timestamp differs from analyzed music beat")
        if len(state.q) != self.profile.nj:
            raise ValueError("boundary joint order/count must equal the profile")
        if max(np.max(np.abs(state.q-self.profile.neutral)),np.max(np.abs(state.dq)),np.max(np.abs(state.ddq)))>1e-6:
            return self._recovery(start_beat,state,began)
        try:
            self._prepare(start_beat,self._last_segment)
            action = self._service(now)
            if self._current is not None and self._graph.edge_status(self._current,action)!="certified":
                raise FinitePlanningError("uncertified_edge", "persistent route attempted an unknown/rejected edge")
            seg = self._make_action(action,start_beat)
            validation = validate_segment(seg,self.profile,previous=self._last_segment)
            if not validation["valid"]:
                raise FinitePlanningError("candidate_search_failed", f"actual-timestamp final check rejected {action.key}: {validation['errors']}")
        except FinitePlanningError as exc:
            self._failures.append({"start_beat":start_beat,"status":exc.status,"detail":exc.detail})
            raise
        repeat = self._current is not None and self._current.primitive_id == action.primitive_id
        pending_goal = self._goal
        self._current = action
        self._last_segment = seg
        self._recent.append(action)
        self._motion_visits[action.primitive_id] += 1
        self._rhythm_visits[action.rhythm_id] += 1
        self._context_motion_visits[action.primitive_id] += 1
        self._context_rhythm_visits[action.rhythm_id] += 1
        if not self.config["no_coverage"]:
            self._bag = [goal for goal in self._bag if goal not in
                         (("motion",action.primitive_id),("rhythm",action.rhythm_id))]
            self._bag_actions += 1
            if not self._bag:
                certificate = self._base_certificate
                elapsed = seg.end-self._bag_start
                self._bags.append({"context":self._context,"seconds":elapsed,
                                   "actions":self._bag_actions,
                                   "bound_seconds":certificate["bag_completion_bound_seconds"],
                                   "within_bound":elapsed<=certificate["bag_completion_bound_seconds"]+1e-7})
                self._path.clear()
                self._goal = None
        elapsed = time.perf_counter()-began
        self._latencies.append(elapsed)
        seg.metadata.update({"planning_seconds":elapsed,"service_goal":pending_goal,
                             "remaining_service_path":[a.key for a in self._path],
                             "coverage_context":self._context,
                             "repeat_reason":"minimum-certified-service-path; progress takes precedence" if repeat else None})
        return seg

    def stats(self):
        component = self._component
        graph = self._graph
        scc_motions = {a.primitive_id for a in component}
        scc_rhythms = {a.rhythm_id for a in component}
        checked = len(graph.checked_edges) if graph else 0
        nodes = len(graph.nodes) if graph else 0
        return {"mode":"finite","seed":self.seed,"library_count":len(self.primitives),
                "library_hash":self.library_hash,"geometry_frozen_before_playback":True,
                "projected_unique_motion_count":len(self.geometry_descriptors),
                "projected_duplicates":dict(self.projected_duplicates),
                "feasible_motion_count":len(self._certified_motion_ids),
                "feasible_count_note":"motions with at least one certified tested macro-action; conservative certificate failures are not proofs of physical impossibility",
                "scc_motion_count":len(scc_motions),"scc_rhythm_count":len(scc_rhythms),
                "scheduled_motion_count":len(self._motion_visits),
                "motion_coverage":len(self._motion_visits)/len(self.primitives),
                "library_motion_coverage":len(self._motion_visits)/len(self.primitives),
                "feasible_motion_coverage":len(set(self._context_motion_visits)&self._certified_motion_ids)/max(1,len(self._certified_motion_ids)),
                "scc_motion_coverage":len(set(self._context_motion_visits)&scc_motions)/max(1,len(scc_motions)),
                "rhythm_coverage":len(self._rhythm_visits)/len(RHYTHMS),
                "scc_rhythm_coverage":len(set(self._context_rhythm_visits)&scc_rhythms)/max(1,len(scc_rhythms)),
                "coverage_observation_scope":"scheduled reference segments; execution success must be confirmed separately by runner/feedback",
                "macro_action_nodes":nodes,"certified_edges":sum(len(v) for v in graph.adjacency.values()) if graph else 0,
                "checked_edges":checked,"unknown_edges":nodes*nodes-checked,
                "rejected_edges":sum(v["status"]!="certified" for v in graph.checked_edges.values()) if graph else 0,
                "whole_dense_graph_certified":False,
                "candidate_statuses":dict(self._candidate_statuses),
                "coverage_certificate":({"valid":False,"reason":"shuffle-bag fair service is disabled by no_coverage ablation"}
                    if self.config["no_coverage"] else (self._base_certificate if graph else {"valid":False})),
                "base_graph_certificate":self._base_certificate if graph else {"valid":False},
                "lazy_shortcut_edges_checked":self._lazy_edges_checked,
                "certificate_resets":self._certificate_resets,"contexts":list(self._contexts),
                "metrical_level":self._current_level,"k_changes":self._k_changes,
                "completed_bags":list(self._bags),"pending_goal":self._goal,
                "remaining_goal_count":len(self._bag),"remaining_path":[a.key for a in self._path],
                "motion_visits":dict(self._motion_visits),"rhythm_visits":dict(self._rhythm_visits),
                "recovery_count":self._recoveries,"failures":list(self._failures),
                "preparation_seconds":getattr(self,"preparation_seconds",0.),
                "planning_latency_seconds":{"median":float(np.median(self._latencies)) if self._latencies else 0.,
                                            "p95":float(np.percentile(self._latencies,95)) if self._latencies else 0.,
                                            "max":max(self._latencies,default=0.)}}
