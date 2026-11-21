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


def _key(s):
    return "".join(c for c in str(s).lower() if c.isalnum())


class MistyAdapter(RobotAdapter):
    def __init__(self, profile, *, execute=False, clock=time.monotonic, http=None,
                 websocket_factory=None, enable_feedback=True):
        if profile.model != "misty_ii_head_arms" or tuple(profile.joint_names) != JOINTS:
            raise CapabilityError(f"Misty requires model misty_ii_head_arms and joint order {JOINTS}")
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
        self._last_dq = np.zeros(5)
        self._last_dispatch = -float("inf")
        self._active_until = -float("inf")
        self._active_reference = None
        self._active_start = None
        self.command_log = []

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
            for joint, sensor in zip(JOINTS, SENSORS):
                subscription = {"Operation": "subscribe", "Type": "ActuatorPosition", "DebounceMs": debounce,
                                "EventName": f"Rita_{joint}", "ReturnProperty": None,
                                "EventConditions": [filters[joint]]}
                self._ws.send(json.dumps(subscription))
            self._feedback_thread = threading.Thread(target=self._receive, name="rita-misty-feedback", daemon=True)
            self._feedback_thread.start()
        return self.get_capabilities()

    def _feedback_filters(self):
        """Use advertised field names instead of assuming one firmware schema."""
        ep = self.profile.endpoint
        custom_property = ep.get("feedback_filter_property")
        custom_values = ep.get("feedback_sensor_values", {})
        if custom_property:
            if set(custom_values) != set(JOINTS):
                raise AdapterError("feedback_sensor_values must explicitly map all five joints when a custom filter property is set")
            return {j: {"Property": custom_property, "Inequality": "=", "Value": custom_values[j]} for j in JOINTS}
        response = self._http("GET", "/api/websockets?websocketClass=ActuatorPosition")
        entries = response.get("result", []) if isinstance(response, dict) else response
        if isinstance(entries, dict): entries = [entries]
        properties = []
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict):
                properties.extend(entry.get("nestedProperties", entry.get("NestedProperties", [])))
        property_map = {_key(p): p for p in properties}
        if "sensorname" in property_map:
            return {j: {"Property": property_map["sensorname"], "Inequality": "=", "Value": s} for j, s in zip(JOINTS, SENSORS)}
        device = self._device.get("result", self._device) if isinstance(self._device, dict) else {}
        for prop, capability_name in (("sensorid", "sensorCapabilities"), ("actuatorid", "outputCapabilities")):
            if prop not in property_map: continue
            mapping = {_key(k): v for k, v in device.get(capability_name, {}).items()}
            if all(_key(s) in mapping for s in SENSORS):
                return {j: {"Property": property_map[prop], "Inequality": "=", "Value": mapping[_key(s)]} for j, s in zip(JOINTS, SENSORS)}
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
                    try: self._http("POST", "/api/halt", {})
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
        for j, s in zip(JOINTS, SENSORS):
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
                "joint_names": list(JOINTS), "units": "rad", "feedback_freshness": self.profile.feedback_freshness,
                "command_mode": "misty_pose", "command_period": self.profile.command_period,
                "queue": False, "cancel": True, "future_execution": False, "streaming_c2": False,
                "reference_exact": False, "head_timing": "HTTP Duration in seconds",
                "arms_timing": "velocity percent; duration unsupported", "wheels": False,
                "feedback_timestamp": "host monotonic reception; robot created timestamp retained but not clock-synchronized",
                "completion": "HTTP acceptance is NOT motion completion", "stop_path": "/api/halt"}

    def read_state(self):
        if not self.connected:
            raise AdapterError("Misty is disconnected")
        if self._error:
            raise AdapterError(self._error)
        now = self.clock()
        with self._lock:
            missing = [j for j in JOINTS if j not in self._positions]
            if missing:
                raise StaleFeedback(f"No ActuatorPosition feedback yet for {missing}")
            ages = [now-self._positions[j][1] for j in JOINTS]
            q = np.array([self._positions[j][0] for j in JOINTS])
            stamp = min(self._positions[j][1] for j in JOINTS)
        fresh = max(ages) <= self.profile.feedback_freshness
        dq = self._last_dq.copy()
        if self._last_t is not None and stamp > self._last_t+1e-5:
            dq = (q-self._last_q)/(stamp-self._last_t)
        if self._last_t is None or stamp > self._last_t:
            self._last_q, self._last_t, self._last_dq = q.copy(), stamp, dq.copy()
        return RobotState(t=stamp, q=q, dq=dq, joint_names=JOINTS, units="rad", fresh=fresh,
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
                                      "LeftArmVelocity": percent, "RightArmVelocity": percent, "Units": "degrees"}})
        return commands

    def ready_for_next_segment(self):
        if self._error:
            raise AdapterError(self._error)
        if self._worker is not None and self._worker.is_alive():
            return False
        if self.clock() < self._active_until-.001:
            return False
        state = self.read_state()
        if not state.fresh:
            return False
        if self._active_reference is None:
            return True
        q = self._active_reference.at(self._active_reference.end)[0]
        return bool(np.max(np.abs(state.q-q)) <= float(self.profile.endpoint.get("max_tracking_error_rad", .15))
                    and np.max(np.abs(state.dq)) <= float(self.profile.endpoint.get("standstill_velocity_tolerance_rad_s", .02)))

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
        if np.max(np.abs(state.q-segment.at(segment.start)[0])) > float(self.profile.endpoint.get("max_tracking_error_rad", .15)):
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
        try:
            for command in commands:
                due = start+command["offset"]
                if self._cancel.wait(max(0, due-self.clock())):
                    return
                now = self.clock()
                if now-due > self.profile.command_period:
                    raise DispatchDeadlineMiss("Misty pose dispatch deadline missed; stopping instead of catching up")
                if now-self._last_dispatch < self.profile.command_period-.002:
                    raise DispatchDeadlineMiss("Misty HTTP pose rate limit would be exceeded")
                state = self.read_state()
                if not state.fresh:
                    raise StaleFeedback("Misty feedback stale during dispatch")
                target = self._active_reference.at(self._active_reference.start+max(0, now-self._active_start))[0]
                if np.max(np.abs(state.q-target)) > float(self.profile.endpoint.get("max_tracking_error_rad", .15)):
                    raise AdapterError("Misty feedback tracking error exceeded local threshold; stopping, not snapping to reference")
                head_reply = self._http("POST", "/api/head", command["head"])
                arms_reply = self._http("POST", "/api/arms/set", command["arms"])
                self._last_dispatch = now
                self.command_log.append({"sent_monotonic": now, "accepted_monotonic": self.clock(),
                                         "planned_target_audio_time": command["target_audio_time"],
                                         "head_accepted": head_reply, "arms_accepted": arms_reply,
                                         "completion": "not measured"})
        except Exception as exc:
            self._error = f"dispatch: {type(exc).__name__}: {exc}"
            self._cancel.set()
            try: self._http("POST", "/api/halt", {})
            except Exception: pass

    def stop(self):
        self._cancel.set()
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=1)
        if self.connected and self.execute:
            return self._http("POST", "/api/halt", {})
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
