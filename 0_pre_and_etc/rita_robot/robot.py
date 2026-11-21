"""Kortex high-level joint velocity control. No low-level mode or limit changes."""
from contextlib import contextmanager
import json
import multiprocessing as mp
import os
import socket
from pathlib import Path
import threading
import time
import numpy as np
from google.protobuf.json_format import MessageToDict
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2, ControlConfig_pb2
from .motion import signed, smoother, SPEED, ACCELERATION


class Robot:
    def __init__(self, ip, feedback_udp=False):
        self.ip = ip
        self.feedback_udp = feedback_udp
        self.udp_transport = None
        self.udp_session = None
        self.transport = TCPTransport()
        # Small RPCs must not wait for Nagle/delayed-ACK batching on Windows.
        self.transport.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.router = RouterClient(self.transport, lambda e: None)
        self.session = SessionManager(self.router)
        self.options = RouterClientSendOptions()
        self.options.timeout_ms = 250
        self.base = BaseClient(self.router)
        self.cyclic = BaseCyclicClient(self.router)
        self.config = ControlConfigClient(self.router)

    def __enter__(self):
        self.transport.connect(self.ip, 10000)
        info = Session_pb2.CreateSessionInfo(
            username=os.getenv("KINOVA_USERNAME", "admin"),
            password=os.getenv("KINOVA_PASSWORD", "admin"),
            session_inactivity_timeout=3000, connection_inactivity_timeout=500)
        try:
            options = RouterClientSendOptions(); options.timeout_ms = 3000
            self.session.CreateSession(info, options=options)
            if self.feedback_udp:
                self.udp_transport = UDPTransport()
                self.udp_transport.connect(self.ip,10001)
                udp_router = RouterClient(self.udp_transport,lambda e: None)
                self.udp_session = SessionManager(udp_router)
                self.udp_session.CreateSession(info,options=options)
                self.cyclic = BaseCyclicClient(udp_router)
        except BaseException:
            if self.udp_transport is not None: self.udp_transport.disconnect()
            self.transport.disconnect()
            raise
        return self

    def __exit__(self, *args):
        try:
            if self.udp_transport is not None:
                try: self.udp_session.CloseSession(options=self.options)
                finally: self.udp_transport.disconnect()
        finally:
            try: self.session.CloseSession(options=self.options)
            finally: self.transport.disconnect()

    def feedback(self): return self.cyclic.RefreshFeedback(options=self.options)

    def stop(self): self.base.Stop(options=self.options)

    def speeds(self, velocity):
        command = Base_pb2.JointSpeeds()
        for i, v in enumerate(velocity):
            command.joint_speeds.add(joint_identifier=i, value=float(v))
        self.base.SendJointSpeedsCommand(command, options=self.options)

    def snapshot(self):
        product = self.base.GetProductConfiguration(options=self.options)
        if MessageToDict(product).get("model") != "MODEL_ID_L53" or product.degree_of_freedom != 6:
            raise RuntimeError("Only the Gen3 6-DOF (L53) model is supported")
        if MessageToDict(product).get("wristType") != "WRIST_TYPE_SPHERICAL":
            raise RuntimeError("Expected Gen3 spherical wrist")
        mode = self.base.GetServoingMode(options=self.options)
        if mode.servoing_mode != Base_pb2.SINGLE_LEVEL_SERVOING:
            raise RuntimeError("Single-level servoing is required; mode will not be changed automatically")
        f = self.feedback()
        check_faults(f)
        if len(f.actuators) != 6: raise RuntimeError("Expected six actuator feedback values")
        hard = self.config.GetKinematicHardLimits(options=self.options)
        limits = self.config.GetAllKinematicSoftLimits(options=self.options)
        soft = next(x for x in limits.kinematic_limits_list
                    if x.control_mode == ControlConfig_pb2.ANGULAR_JOYSTICK)
        velocity = np.minimum(SPEED, np.minimum(hard.joint_speed_limits, soft.joint_speed_limits))
        acceleration = np.minimum(ACCELERATION, hard.joint_acceleration_limits)
        if len(soft.joint_acceleration_limits) == 6:
            acceleration = np.minimum(acceleration, soft.joint_acceleration_limits)
        if np.any(velocity <= 0) or np.any(acceleration <= 0): raise RuntimeError("Invalid firmware limits")
        return {"product": MessageToDict(product), "feedback": MessageToDict(f),
                "q": signed([a.position for a in f.actuators]).tolist(),
                "speed_limits": velocity.tolist(), "accel_limits": acceleration.tolist()}


def check_faults(f):
    if f.base.fault_bank_a or f.base.fault_bank_b or any(
            a.fault_bank_a or a.fault_bank_b for a in f.actuators):
        raise RuntimeError("Robot reports a fault; no automatic fault clearing")


def _posix_key_pressed():
    """Non-blocking Q/Esc check on macOS/Linux; requires operator_keys() cbreak mode."""
    import select, sys
    if not sys.stdin.isatty(): return False
    while select.select([sys.stdin], [], [], 0)[0]:
        ch = os.read(sys.stdin.fileno(), 1)
        if not ch: return False
        if ch.lower() in (b"q", b"\x1b", b"\x03"): return True
    return False


@contextmanager
def operator_keys():
    """Put a POSIX terminal in cbreak mode so single Q/Esc keys are read without Enter.
    Ctrl+C keeps working (signals stay enabled). The terminal is always restored."""
    import sys
    if os.name == "nt" or not sys.stdin.isatty():
        yield; return
    import termios, tty
    fd = sys.stdin.fileno(); saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd); yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _watchdog(ip, pulse, armed, done, tripped, ready):
    """Independent process remains able to stop if the controller hangs/dies."""
    try:
        with Robot(ip) as robot:
            ready.set()
            while not done.wait(.025):
                if armed.is_set() and time.monotonic()-pulse.value > .4:
                    tripped.set()
                    for _ in range(3):
                        try: robot.stop()
                        except Exception: pass
                        time.sleep(.03)
                    return
    except BaseException:
        tripped.set()
        ready.set()


@contextmanager
def watchdog(ip):
    ctx = mp.get_context("spawn")
    pulse = ctx.Value("d", time.monotonic(), lock=False)
    armed, done, tripped, ready = (ctx.Event() for _ in range(4))
    process = ctx.Process(target=_watchdog, args=(ip,pulse,armed,done,tripped,ready),
                          name="Rita safety watchdog")
    process.start()
    if not ready.wait(8) or tripped.is_set():
        done.set(); process.join(3)
        raise RuntimeError("Independent watchdog did not connect")
    try:
        yield pulse, armed, tripped
    finally:
        done.set(); process.join(3)
        if process.is_alive():
            process.terminate(); process.join()


class Controller:
    def __init__(self, robot, model, snapshot, plan, pulse, armed, tripped, log):
        self.robot, self.model, self.plan = robot, model, plan
        self.initial = np.array(snapshot["q"])
        self.vmax = np.array(snapshot["speed_limits"])
        self.amax = np.array(snapshot["accel_limits"])
        self.pulse, self.armed, self.tripped = pulse, armed, tripped
        self.log = log
        self.previous_v = np.zeros(6)
        self.owner = None
        self.first_command_time = None
        self.last_tick = time.monotonic()
        self.initial_tool = None
        self.guard_low = np.minimum(self.initial-.4, model.lower+5)
        self.guard_high = np.maximum(self.initial+.4, model.upper-5)
        self.max_tracking_error = 0.
        self.max_loop_gap = 0.

    def tick(self, target, feedforward, stage, audio_time=None):
        now = time.monotonic(); dt = now-self.last_tick
        self.max_loop_gap = max(self.max_loop_gap, dt if self.armed.is_set() else 0)
        if self.tripped.is_set(): raise RuntimeError("Safety watchdog stopped the robot")
        if self.armed.is_set() and dt > .2: raise RuntimeError("Control deadline exceeded")
        if Path("STOP_RITA").exists(): raise RuntimeError("STOP_RITA requested")
        if os.name == "nt":
            import msvcrt
            if msvcrt.kbhit() and msvcrt.getwch().lower() in ("q", "\x1b", "\x03"):
                raise KeyboardInterrupt("Operator stop")
        elif _posix_key_pressed():
            raise KeyboardInterrupt("Operator stop")
        f = self.robot.feedback(); check_faults(f)
        q = signed([a.position for a in f.actuators])
        measured_v = np.array([a.velocity for a in f.actuators])
        tool = np.array([f.base.tool_pose_x, f.base.tool_pose_y, f.base.tool_pose_z])
        if self.initial_tool is None: self.initial_tool = tool
        state = f.base.active_state
        if self.first_command_time is None:
            if state != Base_pb2.ARMSTATE_SERVOING_READY:
                raise RuntimeError("Robot is not ready; release manual controls")
        elif state == Base_pb2.ARMSTATE_SERVOING_MANUALLY_CONTROLLED:
            connection = f.base.active_state_connection_identifier
            if self.owner is None: self.owner = connection
            if connection != self.owner: raise RuntimeError("Another controller took control")
        elif now-self.first_command_time > .25:
            raise RuntimeError("Robot stopped or left active control; will not restart automatically")
        if np.linalg.norm(tool-self.initial_tool) > .22 or tool[2] < .25:
            raise RuntimeError("Tool exceeded workspace envelope")
        low, high = self.guard_low, self.guard_high
        if stage != "prepare":
            low, high = self.model.lower+5, self.model.upper-5
        if np.any(q < low) or np.any(q > high): raise RuntimeError("Joint guard reached")
        if np.max(np.abs(signed(q-self.plan.center))) > 35:
            raise RuntimeError("Robot left the local dance volume")
        error = signed(target-q)
        tracking = float(np.max(np.abs(error)))
        self.max_tracking_error = max(self.max_tracking_error, tracking)
        if tracking > 4: raise RuntimeError("Joint tracking error exceeded 4 degrees")
        if np.any(np.abs(measured_v) > self.vmax*1.3+2): raise RuntimeError("Unexpected measured joint speed")
        command = np.clip(feedforward+2.5*error, -self.vmax, self.vmax)
        command = np.clip(command, self.previous_v-self.amax*min(dt,.05),
                          self.previous_v+self.amax*min(dt,.05))
        # Braking-distance guard. During inward recovery only, an initial angle
        # slightly outside the conservative URDF range is allowed to move inward.
        for i in range(6):
            if stage == "prepare" and q[i] < self.model.lower[i]+5:
                if command[i] < -.001: raise RuntimeError("Outward recovery velocity rejected")
            elif stage == "prepare" and q[i] > self.model.upper[i]-5:
                if command[i] > .001: raise RuntimeError("Outward recovery velocity rejected")
            else:
                stopping = command[i]**2/(2*self.amax[i]) + abs(command[i])*.15 + .2
                if command[i] < 0 and q[i]-stopping < low[i]: raise RuntimeError("Lower braking margin reached")
                if command[i] > 0 and q[i]+stopping > high[i]: raise RuntimeError("Upper braking margin reached")
        self.pulse.value = time.monotonic()
        self.armed.set()
        self.robot.speeds(command)
        if self.first_command_time is None: self.first_command_time = now
        self.previous_v = command
        self.last_tick = now
        self.log.write(json.dumps({"monotonic":now,"stage":stage,"audio_time":audio_time,
                                  "q":q.tolist(),"target":target.tolist(),
                                  "command_speed":command.tolist(),"measured_speed":measured_v.tolist(),
                                  "tool":tool.tolist()})+"\n")
        return tracking

    def prepare(self):
        # No homing, joint unlocking, emergency reset, or Cartesian IK changes.
        delta = self.plan.center-self.initial
        if np.any(self.initial < self.model.lower-2) or np.any(self.initial > self.model.upper+2):
            raise RuntimeError("Starting joint pose too far outside official model")
        duration = max(8., float(np.max(1.875*np.abs(delta)/np.minimum(self.vmax,5.))))
        began = time.monotonic(); deadline = began
        while True:
            t = time.monotonic()-began; u = min(t/duration,1.)
            target = self.initial+smoother(u)*delta
            velocity = (30*u*u*(1-u)**2/duration)*delta if u < 1 else np.zeros(6)
            self.tick(target,velocity,"prepare")
            if u >= 1: break
            deadline += .025
            time.sleep(max(0, deadline-time.monotonic()))

    def settle(self, duration=1.):
        began = time.monotonic(); deadline = began
        while time.monotonic()-began < duration:
            error = self.tick(self.plan.center,np.zeros(6),"settle")
            deadline += .025; time.sleep(max(0,deadline-time.monotonic()))
        if error > .5: raise RuntimeError("Neutral posture did not settle")
