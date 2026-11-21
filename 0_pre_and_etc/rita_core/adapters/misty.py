"""Misty II HTTP pose compiler and WebSocket actuator feedback.

The reference spline is certified separately: Misty's internal interpolation is
not assumed to reproduce it. No wheel command is implemented by this adapter.
"""
import json
import threading
import time
from urllib.request import Request, urlopen
import numpy as np
from ..contracts import RobotState
from .base import (RobotAdapter, AdapterError, StaleFeedback, DispatchDeadlineMiss,
                   CapabilityError, require_motion_permission, require_segment, terminal_hold_start)

JOINTS = ("head_pitch", "head_roll", "head_yaw", "left_arm", "right_arm")
SENSORS = ("Actuator_HeadPitch", "Actuator_HeadRoll", "Actuator_HeadYaw", "Actuator_LeftArm", "Actuator_RightArm")
# Head-only mode: arms receive no commands and are not part of feedback checks.
MODELS = {"misty_ii_head_arms": (JOINTS, SENSORS), "misty_ii_head": (JOINTS[:3], SENSORS[:3])}


def _key(s):
    return "".join(c for c in str(s).lower() if c.isalnum())


class MistyAdapter(RobotAdapter):
    def __init__(self, profile, *, execute=False, clock=time.monotonic, http=None,
                 websocket_factory=None, enable_feedback=True):
        if profile.model not in MODELS or tuple(profile.joint_names) != MODELS[profile.model][0]:
            raise CapabilityError(f"Misty requires a model/joint order from {MODELS}")
        self.joints, self.sensors = MODELS[profile.model]
        self.arms = "left_arm" in self.joints
        self.wheels = bool(profile.endpoint.get("wheels_enabled", False))
        if profile.command_mode != "misty_pose" or profile.command_period < .1:
            raise CapabilityError("Misty pose mode requires command_period >= 0.1 seconds")
        self.profile, self.execute, self.clock = profile, execute, clock
        self._http_override = http
        self._ws_factory = websocket_factory
        self.enable_feedback = enable_feedback
        self.connected = False
        self._ws = None
        self._feedback_thread = self._worker = None
        self._closing = threading.Event()
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._positions = {}
        self._device = {}
        self._error = None
        self._last_q = None
        self._last_t = None
        self._last_dq = np.zeros(len(self.joints))
        self._last_dispatch = -float("inf")
        self._active_until = -float("inf")
        self._active_reference = None
        self._active_start = None
        self.command_log = []
        self.skipped_commands = 0
        self._last_by_kind = {}

    def _http(self, method, path, payload=None):
        if self._http_override is not None:
            response = self._http_override(method, path, payload)
        else:
            host = self.profile.endpoint.get("host", "")
            if not host or host == "REPLACE_ME":
                raise AdapterError("Set endpoint.host once in a gitignored Misty local profile")
            port = int(self.profile.endpoint.get("http_port", 80))
            url = f"http://{host}:{port}{path}"
            data = None if payload is None else json.dumps(payload).encode()
            request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=float(self.profile.endpoint.get("http_timeout_seconds", .5))) as reply:
                response = json.load(reply)
        if isinstance(response, dict) and (str(response.get("status", "Success")).lower() == "failed" or response.get("result") is False):
            raise AdapterError(f"Misty rejected {method} {path}: {response.get('error', 'result=false')}")
        return response

    def connect(self):
        # Read-only connection: no movement, parameter change, reset, or homing.
        self._device = self._http("GET", "/api/device")
        self._websocket_version = self._http("GET", "/api/websocket/version")
        self._closing.clear()
        self.connected = True
        if self.enable_feedback:
            factory = self._ws_factory
            if factory is None:
                try:
                    from websocket import create_connection
                except ImportError as exc:
                    self.connected = False
                    raise RuntimeError("Misty feedback requires optional websocket-client: pip install -e '.[misty]'") from exc
                factory = create_connection
            host = self.profile.endpoint.get("host", "")
            if not host or host == "REPLACE_ME":
                raise AdapterError("Set endpoint.host in Misty local profile")
            self._ws = factory(f"ws://{host}:{int(self.profile.endpoint.get('ws_port', 80))}/pubsub", timeout=.25)
            debounce = int(self.profile.endpoint.get("feedback_period_ms", 50))
            filters = self._feedback_filters()
            for joint, sensor in zip(self.joints, self.sensors):
                subscription = {"Operation": "subscribe", "Type": "ActuatorPosition", "DebounceMs": debounce,
                                "EventName": f"Rita_{joint}", "ReturnProperty": None,
                                "EventConditions": [filters[joint]]}
                self._ws.send(json.dumps(subscription))
            self._feedback_thread = threading.Thread(target=self._receive, name="rita-misty-feedback", daemon=True)
            self._feedback_thread.start()
        if self.execute and self.profile.endpoint.get("park_on_connect", False):
            # Explicit, opt-in move to the neutral head pose before playback (the runtime requires a neutral start).
            require_motion_permission(self.profile, self.execute)
            self._park()
            time.sleep(float(self.profile.endpoint.get("park_seconds", 1.5))+.5)
        return self.get_capabilities()

    def _park(self):
        """Move the head (and arms, if controlled) to neutral slowly instead of cutting motor power."""
        q = np.rad2deg(self.profile.neutral)
        duration = float(self.profile.endpoint.get("park_seconds", 1.5))
        reply = self._http("POST", "/api/head", {"Pitch": float(q[0]), "Roll": float(q[1]), "Yaw": float(q[2]),
                                                 "Duration": duration, "Units": "degrees"})
        if self.arms:
            percent = float(self.profile.endpoint.get("arm_velocity_percent", 15))
            self._http("POST", "/api/arms/set", {"LeftArmPosition": float(q[3]), "RightArmPosition": float(q[4]),
                                                 "LeftArmVelocity": percent, "RightArmVelocity": percent, "Units": "degrees"})
        return reply

    def _safe_stop(self):
        """stop_mode "halt" = POST /api/halt (head drops on Misty II); "park" = glide back to neutral."""
        if self.wheels:
            self._http("POST", "/api/drive/stop", {})
        if self.profile.endpoint.get("stop_mode", "halt") == "park":
            return self._park()
        return self._http("POST", "/api/halt", {})

    def _feedback_filters(self):
        """Use advertised field names instead of assuming one firmware schema."""
        ep = self.profile.endpoint
        custom_property = ep.get("feedback_filter_property")
        custom_values = ep.get("feedback_sensor_values", {})
        if custom_property:
            if set(custom_values) != set(self.joints):
                raise AdapterError("feedback_sensor_values must explicitly map all five joints when a custom filter property is set")
            return {j: {"Property": custom_property, "Inequality": "=", "Value": custom_values[j]} for j in self.joints}
        response = self._http("GET", "/api/websockets?websocketClass=ActuatorPosition")
        entries = response.get("result", []) if isinstance(response, dict) else response
        if isinstance(entries, dict): entries = [entries]
        properties = []
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict):
                properties.extend(entry.get("nestedProperties", entry.get("NestedProperties", [])))
        property_map = {_key(p): p for p in properties}
        if "sensorname" in property_map:
            return {j: {"Property": property_map["sensorname"], "Inequality": "=", "Value": s} for j, s in zip(self.joints, self.sensors)}
        device = self._device.get("result", self._device) if isinstance(self._device, dict) else {}
        for prop, capability_name in (("sensorid", "sensorCapabilities"), ("actuatorid", "outputCapabilities")):
            if prop not in property_map: continue
            mapping = {_key(k): v for k, v in device.get(capability_name, {}).items()}
            if all(_key(s) in mapping for s in self.sensors):
                return {j: {"Property": property_map[prop], "Inequality": "=", "Value": mapping[_key(s)]} for j, s in zip(self.joints, self.sensors)}
        raise AdapterError("Firmware does not advertise a usable per-actuator filter; set feedback_filter_property and feedback_sensor_values explicitly after read-only inspection")

    def _receive(self):
        try:
            while not self._closing.is_set():
                try:
                    message = self._ws.recv()
                except TimeoutError:
                    continue
                except Exception as exc:
                    # websocket-client defines its own timeout class.
                    if "timeout" in type(exc).__name__.lower():
                        continue
                    raise
                if not message:
                    raise ConnectionError("Misty WebSocket closed")
                self.ingest_feedback(message)
        except Exception as exc:
            if not self._closing.is_set():
                self._error = f"feedback: {type(exc).__name__}: {exc}"
                self._cancel.set()
                if self.execute:
                    try: self._safe_stop()
                    except Exception: pass

    def ingest_feedback(self, message, received_monotonic=None):
        """Handle current/deprecated field casing and per-sensor event names."""
        if isinstance(message, str):
            message = json.loads(message)
        if not isinstance(message, dict):
            raise AdapterError("Misty feedback must be a JSON object")
        outer = {_key(k): v for k, v in message.items()}
        payload = outer.get("message", outer.get("data", message))
        if not isinstance(payload, dict):
            return False  # Subscription acknowledgements are not actuator data.
        fields = {_key(k): v for k, v in payload.items()}
        sensor = fields.get("sensorname", outer.get("sensorname"))
        event = outer.get("eventname", "")
        joint = None
        for j, s in zip(self.joints, self.sensors):
            if _key(sensor) == _key(s) or _key(event) == _key("Rita_"+j):
                joint = j
                break
        if joint is None or "value" not in fields:
            return False
        value = float(fields["value"])
        if not np.isfinite(value):
            raise AdapterError("Nonfinite Misty actuator position")
        now = self.clock() if received_monotonic is None else float(received_monotonic)
        with self._lock:
            self._positions[joint] = (np.deg2rad(value), now, fields.get("created"))
        return True

    def get_capabilities(self):
        return {"backend": "misty", "hardware": True, "status": "implemented, hardware-unverified",
                "joint_names": list(self.joints), "units": "rad", "feedback_freshness": self.profile.feedback_freshness,
                "command_mode": "misty_pose", "command_period": self.profile.command_period,
                "queue": False, "cancel": True, "future_execution": False, "streaming_c2": False,
                "reference_exact": False, "head_timing": "HTTP Duration in seconds",
                "arms_timing": "velocity percent; duration unsupported", "wheels": "in-place twist only" if self.wheels else False,
                "feedback_timestamp": "host monotonic reception; robot created timestamp retained but not clock-synchronized",
                "completion": "HTTP acceptance is NOT motion completion", "stop_path": "/api/halt"}

    def read_state(self):
        if not self.connected:
            raise AdapterError("Misty is disconnected")
        if self._error:
            raise AdapterError(self._error)
        now = self.clock()
        with self._lock:
            missing = [j for j in self.joints if j not in self._positions]
            if missing:
                raise StaleFeedback(f"No ActuatorPosition feedback yet for {missing}")
            ages = [now-self._positions[j][1] for j in self.joints]
            q = np.array([self._positions[j][0] for j in self.joints])
            stamp = min(self._positions[j][1] for j in self.joints)
        fresh = max(ages) <= self.profile.feedback_freshness
        dq = self._last_dq.copy()
        if self._last_t is not None and stamp > self._last_t+1e-5:
            dq = (q-self._last_q)/(stamp-self._last_t)
        if self._last_t is None or stamp > self._last_t:
            self._last_q, self._last_t, self._last_dq = q.copy(), stamp, dq.copy()
        return RobotState(t=stamp, q=q, dq=dq, joint_names=self.joints, units="rad", fresh=fresh,
                          metadata={"backend": "misty", "feedback_ages_seconds": ages,
                                    "velocity_estimator": "unfiltered first difference of asynchronous received positions; not a certified physical dq/ddq",
                                    "actual_motor_completion": "not inferred from HTTP result"})

    def compile_segment(self, segment):
        require_segment(segment, self.profile, stop_to_stop=True)
        period = self.profile.command_period
        percent = float(self.profile.endpoint.get("arm_velocity_percent", 15))
        if not 0 < percent <= 100:
            raise ValueError("Misty arm_velocity_percent must be in (0,100]")
        # Count chosen so every interval is >= rate limit; final target included.
        execution_duration = terminal_hold_start(segment)-segment.start
        if execution_duration <= 0:
            raise CapabilityError("A stationary-only segment is not a dance chunk")
        if self.profile.endpoint.get("pose_mode", "sampled") == "waypoints":
            # Misty's head restarts its own interpolation on every request and cannot follow a dense
            # stream of short poses (measured: it lags and the pitch sags). Send one pose per stored
            # zero-velocity knot instead, with Duration = time to that knot; Misty interpolates the leg.
            hold_at = segment.start+execution_duration
            knots_mode = self.profile.endpoint.get("waypoint_knots", "auto")
            if knots_mode == "auto":
                knots_mode = "all" if "generation_kind" in segment.metadata else "stops"  # infinite segments carry generation_kind
            if knots_mode == "all":
                # Flowing references (infinite planner) pass through waypoints without stopping:
                # use every stored knot, dropping ones closer than min_leg_seconds to the previous pose.
                min_leg = float(self.profile.endpoint.get("min_leg_seconds", period))
                stops, last = [], segment.start
                for k in segment.knots:
                    k = float(k)
                    if segment.start+1e-9 < k < hold_at-1e-9 and k-last >= min_leg and hold_at-k >= min_leg:
                        stops.append(k)
                        last = k
                stops.append(hold_at)
            else:
                stops = [float(k) for k in segment.knots if segment.start+1e-9 < k <= hold_at+1e-9
                         and np.max(np.abs(segment.at(k)[1])) < 1e-6]
                if not stops or abs(stops[-1]-hold_at) > 1e-6:
                    stops.append(hold_at)
            offsets = np.r_[0., np.array(stops)-segment.start]
            if np.min(np.diff(offsets)) < period-1e-9:
                raise CapabilityError("Two pose waypoints are closer than the Misty command period")
        else:
            count = max(1, int(np.floor(execution_duration/period)))
            offsets = np.linspace(0, execution_duration, count+1)
        commands = []
        for a, b in zip(offsets[:-1], offsets[1:]):
            q, _, _ = segment.at(segment.start+b)
            q = np.rad2deg(q)
            commands.append({"offset": float(a), "target_audio_time": float(segment.start+b),
                             "head": {"Pitch": float(q[0]), "Roll": float(q[1]), "Yaw": float(q[2]),
                                      "Duration": float(b-a), "Units": "degrees"},
                             "arms": {"LeftArmPosition": float(q[3]), "RightArmPosition": float(q[4]),
                                      "LeftArmVelocity": percent, "RightArmVelocity": percent, "Units": "degrees"} if self.arms else None})
        if self.arms and self.profile.endpoint.get("pose_mode", "sampled") == "waypoints":
            commands = self._split_arm_commands(segment, commands)
        if self.wheels:
            commands = self._add_twist_commands(commands)
        return commands

    def _add_twist_commands(self, commands):
        """Opt-in body twist: rotate in place (LinearVelocity 0) for each pose leg, alternating direction
        so every pair of legs returns the base to its starting heading. Never drives forward/backward."""
        ep = self.profile.endpoint
        percent = float(ep.get("twist_angular_percent", 25))
        if not 0 < percent <= 60:
            raise ValueError("twist_angular_percent must be in (0,60]")
        fraction = float(ep.get("twist_time_fraction", .8))
        legs = [c for c in commands if c.get("head") is not None]
        legs = legs[:len(legs)//2*2]  # even count keeps the net rotation near zero
        out = list(commands)
        for i, c in enumerate(legs):
            # Both twists of a left/right pair get the same duration, so the pair cancels exactly.
            pair = legs[i-i % 2:i-i % 2+2]
            seconds = fraction*min(x["head"]["Duration"] for x in pair)
            out.append({**c, "head": None, "arms": None,
                        "drive": {"LinearVelocity": 0, "AngularVelocity": percent*(1 if i % 2 == 0 else -1),
                                  "TimeMs": int(1000*seconds)}})
        out.sort(key=lambda c: (c["offset"], c["head"] is None))
        return out

    def _tolerance(self):
        """max_tracking_error_rad: one value, or one per joint (arms are velocity-driven and run ahead)."""
        return np.broadcast_to(np.asarray(self.profile.endpoint.get("max_tracking_error_rad", .15), dtype=float), (len(self.joints),))

    def _arm_percent(self, degrees, seconds):
        """Arms take a velocity percent, not a duration. Measured on Misty II (60-degree moves):
        30% ~ 78 deg/s, 60% ~ 111 deg/s, 100% ~ 177 deg/s, i.e. deg/s ~ 45 + 1.3*percent."""
        ep = self.profile.endpoint
        needed = float(ep.get("arm_speed_margin", 1.25))*degrees/max(seconds, 1e-3)
        percent = (needed-float(ep.get("arm_speed_offset_deg_s", 45)))/float(ep.get("arm_speed_gain_deg_s_per_percent", 1.3))
        return float(np.clip(percent, float(ep.get("arm_min_velocity_percent", 10)), 100))

    def _split_arm_commands(self, segment, commands):
        """Separate arm requests: sized per leg, and sent earlier because the arms start ~0.45 s after a
        request versus ~0.2 s for the head (the head lag is already covered by latency_seconds)."""
        lead = float(self.profile.endpoint.get("arm_extra_lead_seconds", .25))
        out = []
        for c in commands:
            start_q = np.rad2deg(segment.at(segment.start+c["offset"])[0])
            arms = dict(c["arms"])
            seconds = c["head"]["Duration"]
            moves = (abs(arms["LeftArmPosition"]-start_q[3]), abs(arms["RightArmPosition"]-start_q[4]))
            arms["LeftArmVelocity"] = self._arm_percent(moves[0], seconds)
            arms["RightArmVelocity"] = self._arm_percent(moves[1], seconds)
            out.append({**c, "arms": None})
            out.append({**c, "offset": max(0., c["offset"]-lead), "head": None, "arms": arms})
        out.sort(key=lambda c: (c["offset"], c["head"] is None))
        return out

    def ready_for_next_segment(self):
        if self._error:
            raise AdapterError(self._error)
        self.not_ready_reason = None
        if self._worker is not None and self._worker.is_alive():
            self.not_ready_reason = "dispatch worker still sending"
            return False
        if self.clock() < self._active_until-.001:
            self.not_ready_reason = "previous action not at its nominal end"
            return False
        state = self.read_state()
        if not state.fresh:
            self.not_ready_reason = "feedback stale"
            return False
        if self._active_reference is None:
            return True
        q = self._active_reference.at(self._active_reference.end)[0]
        if not np.all(np.abs(state.q-q) <= self._tolerance()):
            self.not_ready_reason = f"not at segment end pose (max error {np.degrees(np.max(np.abs(state.q-q))):.1f} deg)"
            return False
        if np.max(np.abs(state.dq)) > float(self.profile.endpoint.get("standstill_velocity_tolerance_rad_s", .02)):
            self.not_ready_reason = f"still moving (|dq| {np.max(np.abs(state.dq)):.2f} rad/s)"
            return False
        return True

    def execute_segment(self, segment, scheduled_monotonic=None, now_music=None):
        require_motion_permission(self.profile, self.execute)
        if not self.connected:
            raise AdapterError("Misty is disconnected")
        if self._worker is not None and self._worker.is_alive():
            raise CapabilityError("Misty has no verified queued chunk protocol; dispatch at the next stop-to-stop boundary")
        if self.clock() < self._active_until-.001:
            raise CapabilityError("Previous Misty pose chunk has not reached its nominal end; no verified future queue is supported")
        state = self.read_state()
        if not state.fresh:
            raise StaleFeedback("Misty feedback is stale; no motion dispatched")
        if np.any(np.abs(state.q-segment.at(segment.start)[0]) > self._tolerance()):
            raise AdapterError("Actual Misty state does not match planned boundary; verified recovery is required")
        commands = self.compile_segment(segment)
        now = self.clock()
        if scheduled_monotonic is None:
            scheduled_monotonic = now+(max(0, segment.start-now_music) if now_music is not None else 0)
        if scheduled_monotonic < now-self.profile.command_period:
            raise DispatchDeadlineMiss("Expired Misty command; no catch-up burst will be sent")
        self._cancel.clear()
        self._error = None
        self._active_until = scheduled_monotonic+(terminal_hold_start(segment)-segment.start)
        self._active_reference, self._active_start = segment, float(scheduled_monotonic)
        self._worker = threading.Thread(target=self._dispatch, args=(commands, float(scheduled_monotonic)),
                                        name="rita-misty-dispatch", daemon=True)
        self._worker.start()
        return {"sent_monotonic": now, "scheduled_monotonic": scheduled_monotonic,
                "accepted_monotonic": None, "completed": False, "backend": "misty",
                "controller_action_end_monotonic": self._active_until,
                "nominal_segment_end_monotonic": scheduled_monotonic+segment.duration,
                "reference_tracking": "approximate pose compilation; inspect feedback"}

    def _dispatch(self, commands, start):
        skipped = 0
        try:
            for command in commands:
                due = start+command["offset"]
                if self._cancel.wait(max(0, due-self.clock())):
                    return
                # A few ms of HTTP/timer jitter on the previous command must not abort the run:
                # wait out the remaining rate-limit gap; the deadline check below still rejects truly late commands.
                kind = "head" if command.get("head") is not None else ("drive" if command.get("drive") is not None else "arms")
                last = self._last_by_kind.get(kind, -float("inf"))
                gap = last+self.profile.command_period-self.clock()
                if gap > 0 and self._cancel.wait(gap):
                    return
                now = self.clock()
                if now-due > float(self.profile.endpoint.get("max_command_lateness_seconds", self.profile.command_period)):
                    # Drop the stale pose (never burst to catch up); the next on-time pose supersedes it.
                    skipped += 1
                    self.skipped_commands += 1
                    if skipped > int(self.profile.endpoint.get("max_consecutive_skips", 3)):
                        raise DispatchDeadlineMiss("Misty pose dispatch repeatedly late; stopping instead of catching up")
                    continue
                skipped = 0
                if now-last < self.profile.command_period-.002:
                    raise DispatchDeadlineMiss("Misty HTTP pose rate limit would be exceeded")
                state = self.read_state()
                if not state.fresh:
                    raise StaleFeedback("Misty feedback stale during dispatch")
                lag = float(self.profile.endpoint.get("tracking_lag_seconds", 0))
                target = self._active_reference.at(self._active_reference.start+max(0, now-self._active_start-lag))[0]
                if np.any(np.abs(state.q-target) > self._tolerance()):
                    raise AdapterError("Misty feedback tracking error exceeded local threshold; stopping, not snapping to reference")
                head_reply = self._http("POST", "/api/head", command["head"]) if command.get("head") is not None else None
                arms_reply = self._http("POST", "/api/arms/set", command["arms"]) if command.get("arms") is not None else None
                if command.get("drive") is not None:
                    self._http("POST", "/api/drive/time", command["drive"])
                self._last_dispatch = now
                self._last_by_kind[kind] = now
                self.command_log.append({"sent_monotonic": now, "accepted_monotonic": self.clock(),
                                         "planned_target_audio_time": command["target_audio_time"],
                                         "head_accepted": head_reply, "arms_accepted": arms_reply,
                                         "completion": "not measured"})
                hook = getattr(self, "telemetry_hook", None)
                if hook is not None:
                    hook({**self.command_log[-1], "backend": "misty", "command": command})
        except Exception as exc:
            self._error = f"dispatch: {type(exc).__name__}: {exc}"
            self._cancel.set()
            try: self._safe_stop()
            except Exception: pass

    def stop(self):
        self._cancel.set()
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=1)
        if self.connected and self.execute:
            return self._safe_stop()
        return {"stop_requested": False, "reason": "read-only connection; no motor command sent"}

    def close(self):
        try:
            self.stop()
        finally:
            self._closing.set()
            if self._ws:
                self._ws.close()
            if self._feedback_thread and self._feedback_thread is not threading.current_thread():
                self._feedback_thread.join(timeout=1)
            self.connected = False
