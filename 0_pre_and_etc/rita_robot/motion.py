"""C2 blended musical primitives in joint space, with numerical limit checks."""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

NAMES = ["sway", "shoulder_groove", "wrist_wave", "diagonal", "roll", "counterwave", "nod", "open_close"]
AMPLITUDE = np.array([6., 3., 4., 6., 4., 7.])
SPEED = np.array([16., 12., 14., 20., 18., 22.])
ACCELERATION = np.array([30., 30., 32., 50., 45., 55.])


def signed(angles):
    return (np.asarray(angles)+180.) % 360.-180.


def smoother(x):
    x = np.clip(x, 0, 1)
    return x*x*x*(10+x*(-15+6*x))


class RobotModel:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        joints = [j for j in root.findall("joint") if j.get("name", "").startswith("joint_")]
        self.lower = np.full(6, -np.inf)
        self.upper = np.full(6, np.inf)
        self.origins = []
        for i, j in enumerate(joints):
            limit = j.find("limit")
            if j.get("type") != "continuous":
                self.lower[i] = np.degrees(float(limit.get("lower")))
                self.upper[i] = np.degrees(float(limit.get("upper")))
            origin = j.find("origin")
            xyz = np.fromstring(origin.get("xyz"), sep=" ")
            rpy = np.fromstring(origin.get("rpy"), sep=" ")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            transform[:3, 3] = xyz
            self.origins.append(transform)

    def fk(self, q):
        """Joint origins and flange; enough for a conservative workspace envelope."""
        q = np.atleast_2d(q)
        transform = np.tile(np.eye(4), (len(q), 1, 1))
        points = []
        for i, origin in enumerate(self.origins):
            transform = transform @ origin
            points.append(transform[:, :3, 3].copy())
            rz = np.tile(np.eye(4), (len(q), 1, 1))
            angle = np.radians(q[:, i]); c, s = np.cos(angle), np.sin(angle)
            rz[:, 0, 0] = c; rz[:, 0, 1] = -s
            rz[:, 1, 0] = s; rz[:, 1, 1] = c
            transform = transform @ rz
        tip = np.tile(np.eye(4), (len(q), 1, 1))
        tip[:, 2, 3] = -.061525
        flange = transform @ tip
        return np.stack(points, axis=1), flange

    def center(self, initial):
        # Keep the complete oscillation at least 10 degrees from finite limits.
        return np.clip(initial, self.lower+AMPLITUDE+12, self.upper-AMPLITUDE-12)


@dataclass
class Plan:
    times: np.ndarray
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    center: np.ndarray
    sequence: list
    scale: float
    report: dict

    def at(self, t):
        t = float(np.clip(t, 0, self.times[-1]))
        i = min(int(t/(self.times[1]-self.times[0])), len(self.times)-2)
        u = (t-self.times[i])/(self.times[i+1]-self.times[i])
        return (1-u)*self.q[i]+u*self.q[i+1], (1-u)*self.v[i]+u*self.v[i+1]


def create_plan(model, initial, analysis, duration, start=0., seed=7, intensity=1.,
                speed_limits=SPEED, accel_limits=ACCELERATION, dof=6):
    if not 1 <= dof <= 6: raise ValueError("DOF must be in 1..6")
    if not .05 <= intensity <= 1: raise ValueError("Intensity must be in .05..1")
    center = model.center(np.asarray(initial, dtype=float))
    times = np.arange(round(duration*100)+1)/100.
    beat = (times+start-analysis["first_beat_seconds"])/analysis["period_seconds"]
    rng = np.random.default_rng(seed)
    count = int(np.ceil(max(beat.max(), 0)/8))+3
    sequence = [0]
    for _ in range(count):
        sequence.append(int(rng.choice([i for i in range(len(NAMES)) if i != sequence[-1]])))
    # Each phrase has a distinct phase/amplitude pattern. No rest-to-rest waypoint stops.
    strength = np.array([[1,.6,.7,.5,.7,.6], [.5,1,1,.5,.4,.5],
                         [.4,.4,.5,1,1,1], [1,1,.7,1,.5,.6],
                         [.5,.4,.5,1,.8,1], [1,.7,1,.8,1,.7],
                         [.5,1,1,.6,.8,.5], [.9,.6,.8,1,1,.9]])
    phase = np.array([[0,.2,2.,1.,2.,0.], [.6,0,3.1,1.,2.,.5],
                      [0,1.,2.,0.,1.6,3.1], [0,0,1.6,3.1,.4,2.],
                      [1.,.5,2.,0.,2.,4.], [0,3.1,0,3.1,0,3.1],
                      [1.,0,3.1,1.,0,2.], [0,.6,2.,0,3.1,.4]])
    periods = np.array([8.,4.,4.,4.,4.,4.])
    waves = np.stack([np.sin(2*np.pi*beat[:, None]/periods+ph)*st
                      for ph, st in zip(phase, strength)])
    # Fixed 8-motion vocabulary with C2 blending in the last four beats.
    phrase = np.maximum(beat/8., 0)
    index = np.floor(phrase).astype(int)
    mix = smoother((phrase-index-.5)*2)
    a = np.array(sequence)[index]; b = np.array(sequence)[index+1]
    offsets = ((1-mix[:,None])*waves[a,np.arange(len(times))] +
               mix[:,None]*waves[b,np.arange(len(times))])*AMPLITUDE
    offsets[:, dof:] = 0
    # Fade from/to the interior neutral posture; middle phrases remain continuous.
    envelope = smoother(times/2.)*smoother((duration-times)/2.)
    offsets *= envelope[:,None]
    v = np.gradient(offsets, .01, axis=0, edge_order=2)
    acc = np.gradient(v, .01, axis=0, edge_order=2)
    speed_limits, accel_limits = np.asarray(speed_limits), np.asarray(accel_limits)
    scale = min(intensity, float(np.min(speed_limits/(np.max(np.abs(v),axis=0)+1e-9)))*.8,
                float(np.min(accel_limits/(np.max(np.abs(acc),axis=0)+1e-9)))*.8)
    q, v, acc = center+scale*offsets, scale*v, scale*acc
    if np.any(q < model.lower+8) or np.any(q > model.upper-8):
        raise RuntimeError("Plan violates joint position margin")
    points, _ = model.fk(q[::5])
    center_points, _ = model.fk(center)
    excursion = float(np.max(np.linalg.norm(points-center_points, axis=-1)))
    if excursion > .20: raise RuntimeError("Plan exceeds 20 cm arm workspace envelope")
    # Upper arm onwards must stay above the robot's base plane. This does not
    # replace a scene collision model; the operator supplies a cleared workspace.
    if np.min(points[:, 2:, 2]) < .12: raise RuntimeError("Plan approaches base/table plane")
    # Exclude neighboring links; avoid folding the distal wrist into the shoulder.
    separation = np.linalg.norm(points[:, 4:6] - points[:, 1:2], axis=-1)
    if np.min(separation) < .15: raise RuntimeError("Wrist approaches shoulder")
    report = {"duration_seconds": duration, "sample_hz": 100,
              "amplitude_scale": scale, "center_deg": center.tolist(),
              "joint_min_deg": q.min(axis=0).tolist(), "joint_max_deg": q.max(axis=0).tolist(),
              "peak_speed_deg_s": np.max(np.abs(v),axis=0).tolist(),
              "peak_acceleration_deg_s2": np.max(np.abs(acc),axis=0).tolist(),
              "max_arm_origin_excursion_m": excursion,
              "primitive_sequence": [NAMES[i] for i in sequence[:int(duration/(8*analysis['period_seconds']))+2]],
              "collision_scope": "joint margins, arm workspace, base plane and wrist/shoulder separation; no obstacle mesh collision checking"}
    return Plan(times, q, v, acc, center, sequence, scale, report)
