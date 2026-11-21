"""Optional ROS2 FollowJointTrajectory path for the archived Gen3 L53 6-DOF.

No Python high-frequency actuator servo and no legacy JACO/MICO support. Queue
handoff is deliberately conservative: one stop-to-stop action at a time.
"""
import threading
import time
import numpy as np
from ..contracts import RobotState
from .base import (RobotAdapter, AdapterError, CapabilityError, StaleFeedback,
                   DispatchDeadlineMiss, require_motion_permission, require_segment, terminal_hold_start)


class KinovaAdapter(RobotAdapter):
    def __init__(self, profile, *, execute=False, clock=time.monotonic):
        if profile.model != "kinova_gen3_l53_6dof" or profile.nj != 6:
            raise CapabilityError("Only the repository's Gen3 L53 spherical-wrist 6-DOF is supported; Gen3 lite and legacy JACO/MICO need separate verified adapters")
        if tuple(profile.joint_names) != tuple(f"joint_{i}" for i in range(1, 7)):
            raise CapabilityError("This Gen3 adapter requires explicit joint_1..joint_6 order; configure ROS prefix outside this adapter")
        if profile.command_mode != "follow_joint_trajectory":
            raise CapabilityError("Kinova backend requires FollowJointTrajectory mode")
        self.profile, self.execute, self.clock = profile, execute, clock
        self.connected = False
        self.node = self.executor = self.context = self.client = None
        self._spin_thread = None
        self._state = None
        self._state_lock = threading.Lock()
        self._error = None
        self._goal_handle = self._result_future = None
        self._active_until = -float("inf")
        self.command_log = []

    def get_capabilities(self):
        return {"backend": "kinova", "hardware": True, "status": "implemented, hardware-unverified",
                "model": self.profile.model, "joint_names": list(self.profile.joint_names), "units": "rad",
                "feedback_freshness": self.profile.feedback_freshness,
                "command_mode": "follow_joint_trajectory", "command_period": self.profile.command_period,
                "queue": False, "cancel": True, "future_execution": True, "streaming_c2": False,
                "reference_exact": "conditional on verified controller quintic spline interpolation with q/dq/ddq",
                "chunk_boundary": "zero velocity and acceleration; no action preemption for normal chaining",
                "future_time_clock": "ROS header stamp mapped from host monotonic; use_sim_time must be false",
                "stop_semantics": "cancel active action; controller hold/deceleration must be verified locally",
                "legacy_models": False}

    def connect(self):
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.action import ActionClient
            from control_msgs.action import FollowJointTrajectory
            from sensor_msgs.msg import JointState
        except ImportError as exc:
            raise RuntimeError("Kinova requires a sourced ROS2 environment with rclpy, control_msgs, sensor_msgs, trajectory_msgs and the matching official ros2_kortex driver. Mock requires none of these.") from exc
        if self.profile.endpoint.get("use_sim_time", False):
            raise CapabilityError("Host DAC-to-ROS scheduling requires use_sim_time=false; simulation clocks need an explicit mapping")
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node("rita_dance_adapter", namespace=self.profile.endpoint.get("namespace", ""), context=self.context)
        self.node.create_subscription(JointState, self.profile.endpoint.get("joint_states_topic", "/joint_states"), self.ingest_feedback, 10)
        self.client = ActionClient(self.node, FollowJointTrajectory, self.profile.endpoint.get("action", "/joint_trajectory_controller/follow_joint_trajectory"))
        self.executor = MultiThreadedExecutor(context=self.context, num_threads=2)
        self.executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self.executor.spin, name="rita-ros2", daemon=True)
        self._spin_thread.start()
        if not self.client.wait_for_server(timeout_sec=3.):
            self.close()
            raise ConnectionError("FollowJointTrajectory action server is not available; verify the configured action and sourced driver")
        self.connected = True
        return self.get_capabilities()

    def ingest_feedback(self, message, received_monotonic=None):
        names = tuple(message.name)
        if names != tuple(self.profile.joint_names):
            self._error = f"ROS joint state order {names} differs from configured {self.profile.joint_names}; configure a dedicated ordered joint-state stream"
            return False
        q = np.asarray(message.position, dtype=float)
        dq = np.asarray(message.velocity, dtype=float)
        if q.shape != (6,) or dq.shape != (6,) or not np.all(np.isfinite(np.r_[q, dq])):
            self._error = "ROS feedback requires six finite radian positions and rad/s velocities"
            return False
        now = self.clock() if received_monotonic is None else float(received_monotonic)
        # JointState SI units are explicit; retain ROS stamp without assuming its
        # clock is identical to monotonic receipt time.
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        metadata = {"backend": "kinova", "timestamp_basis": "host monotonic reception",
                    "ros_stamp": {"sec": getattr(stamp, "sec", None), "nanosec": getattr(stamp, "nanosec", None)},
                    "velocity": "controller-reported rad/s; acceleration not measured"}
        with self._state_lock:
            self._state = RobotState(t=now, q=q, dq=dq, joint_names=names, units="rad", fresh=True, metadata=metadata)
        return True

    def read_state(self):
        if not self.connected:
            raise AdapterError("ROS2 Kinova adapter is disconnected")
        if self._error:
            raise AdapterError(self._error)
        with self._state_lock:
            if self._state is None:
                raise StaleFeedback("No ordered Gen3 joint-state feedback has been received")
            s = self._state
            return RobotState(t=s.t, q=s.q.copy(), dq=s.dq.copy(), joint_names=s.joint_names,
                              units="rad", fresh=self.clock()-s.t <= self.profile.feedback_freshness,
                              metadata=dict(s.metadata))

    def compile_segment(self, segment):
        require_segment(segment, self.profile, stop_to_stop=True)
        points = []
        # Include EVERY polynomial knot; q/dq/ddq at the endpoints reproduce
        # each quintic if the actual JTC uses its documented quintic interpolator.
        moving_end = terminal_hold_start(segment)
        if moving_end <= segment.start:
            raise CapabilityError("A stationary-only segment is not a dance chunk")
        for t in segment.knots[segment.knots <= moving_end+1e-10]:
            q, dq, ddq = segment.at(float(t))
            points.append({"time_from_start": float(t-segment.start), "positions": q.tolist(),
                           "velocities": dq.tolist(), "accelerations": ddq.tolist()})
        return {"joint_names": list(self.profile.joint_names), "points": points,
                "execution_duration": moving_end-segment.start,
                "terminal_hold_seconds": segment.end-moving_end,
                "interpolation_requirement": "quintic spline q+dq+ddq; verify installed controller version"}

    def _wait(self, future, timeout=2.):
        deadline = self.clock()+timeout
        while not future.done():
            if self.clock() >= deadline:
                raise TimeoutError("ROS2 action response deadline exceeded")
            time.sleep(.005)
        result = future.result()
        if result is None:
            raise AdapterError("ROS2 action returned no result")
        return result

    def ready_for_next_segment(self):
        if self._error:
            raise AdapterError(self._error)
        if self.clock() < self._active_until-.001:
            return False
        if self._result_future is None:
            return True
        if not self._result_future.done():
            return False
        result = self._result_future.result()
        if result is None or getattr(result, "status", None) != 4 or getattr(getattr(result, "result", None), "error_code", -1) != 0:
            raise AdapterError("Previous Kinova action failed or was canceled; no automatic continuation")
        state = self.read_state()
        return bool(state.fresh and np.max(np.abs(state.dq)) <= float(self.profile.endpoint.get("standstill_velocity_tolerance_rad_s", .02)))

    def execute_segment(self, segment, scheduled_monotonic=None, now_music=None):
        require_motion_permission(self.profile, self.execute)
        if not self.connected:
            raise AdapterError("ROS2 Kinova adapter is disconnected")
        ep = self.profile.endpoint
        if not ep.get("controller_parameters_verified", False) or ep.get("interpolation_method") not in ("splines", "spline"):
            raise CapabilityError("Verify installed controller q/dq/ddq quintic interpolation and stop/cancel behavior in the local profile")
        if self._result_future is not None:
            if not self._result_future.done():
                raise CapabilityError("An action is active; no verified queue/preempt handoff is available")
            previous = self._result_future.result()
            if getattr(previous, "status", None) != 4 or getattr(previous.result, "error_code", -1) != 0:
                raise AdapterError("Previous Kinova action did not succeed; automatic restart is forbidden")
        state = self.read_state()
        if not state.fresh:
            raise StaleFeedback("Kinova feedback is stale; no trajectory dispatched")
        first = segment.at(segment.start)
        if np.max(np.abs(state.q-first[0])) > float(ep.get("max_tracking_error_rad", .07)):
            raise AdapterError("Actual Gen3 pose differs from planned boundary; do not snap or automatically home")
        compiled = self.compile_segment(segment)
        now = self.clock()
        if scheduled_monotonic is None:
            scheduled_monotonic = now+(max(0, segment.start-now_music) if now_music is not None else 0)
        if scheduled_monotonic < now:
            raise DispatchDeadlineMiss("Kinova header time has passed; no catch-up action is sent")
        if scheduled_monotonic-now < self.profile.dispatch_lead_seconds:
            raise DispatchDeadlineMiss("Insufficient Kinova action dispatch lead time")
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        from rclpy.time import Time
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = compiled["joint_names"]
        now_ros = self.node.get_clock().now()
        mapped_ns = now_ros.nanoseconds+int((scheduled_monotonic-self.clock())*1e9)
        goal.trajectory.header.stamp = Time(nanoseconds=mapped_ns, clock_type=now_ros.clock_type).to_msg()
        for p in compiled["points"]:
            pt = JointTrajectoryPoint()
            pt.positions, pt.velocities, pt.accelerations = p["positions"], p["velocities"], p["accelerations"]
            ns = round(p["time_from_start"]*1e9)
            pt.time_from_start = Duration(sec=ns//1_000_000_000, nanosec=ns % 1_000_000_000)
            goal.trajectory.points.append(pt)
        self._goal_handle = self._wait(self.client.send_goal_async(goal))
        if not self._goal_handle.accepted:
            raise AdapterError("Kinova FollowJointTrajectory goal was rejected")
        self._result_future = self._goal_handle.get_result_async()
        self._active_until = scheduled_monotonic+compiled["execution_duration"]
        receipt = {"backend": "kinova", "sent_monotonic": now, "accepted_monotonic": self.clock(),
                   "scheduled_monotonic": scheduled_monotonic, "completed": False,
                   "controller_action_end_monotonic": self._active_until,
                   "nominal_segment_end_monotonic": scheduled_monotonic+segment.duration,
                   "terminal_hold_seconds": compiled["terminal_hold_seconds"],
                   "latency_policy": "header is execution time; dispatch lead is not subtracted again"}
        self.command_log.append(receipt)
        hook = getattr(self, "telemetry_hook", None)
        if hook is not None: hook(receipt)
        return receipt

    def stop(self):
        # A cancel request is not a claim of physical standstill.
        if self._result_future is not None and self._result_future.done():
            self._goal_handle = None
            return {"cancel_requested": False, "reason": "action already completed; inspect feedback for physical standstill"}
        if self._goal_handle is not None and self.execute:
            response = self._wait(self._goal_handle.cancel_goal_async(), timeout=2.)
            if not getattr(response, "goals_canceling", []):
                raise AdapterError("Kinova cancel was not accepted; use the hardware emergency stop if needed")
            self._goal_handle = None
            return {"cancel_accepted": True, "physical_standstill": "must be verified from feedback"}
        return {"cancel_requested": False, "reason": "no active motion goal"}

    def close(self):
        try:
            self.stop()
        finally:
            if self.executor:
                self.executor.shutdown(timeout_sec=1.)
            if self.node:
                self.node.destroy_node()
            if self.context and self.context.ok():
                self.context.shutdown()
            if self._spin_thread and self._spin_thread is not threading.current_thread():
                self._spin_thread.join(timeout=1.)
            self.connected = False
