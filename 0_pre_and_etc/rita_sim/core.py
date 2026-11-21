"""Finite joint-pose graph and rest-to-rest kinematic playback.

Transition times are sufficient times for one prescribed quintic path, not
minimum-time solutions or measurements of any physical robot controller.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import mujoco
from .assets import ROOT, MODELS, ROBOT_IDS

POSES = ROOT / 'sim_poses'
TITLES = {'kinova': 'Kinova Gen3 6-DOF', 'g1': 'Unitree G1 (29 joints)',
          'misty': 'Misty simplified (5 joints)', 'go2': 'Unitree Go2 (12 joints)'}
CAMERAS = {'kinova': ([0, 0, .45], 1.65), 'g1': ([0, 0, .7], 2.4),
           'misty': ([0, 0, .17], .85), 'go2': ([0, 0, .24], 1.4)}


def write_default_poses(robot, model):
    path = POSES / (robot + '.json')
    if path.exists():
        return
    names = [model.joint(i).name for i in range(model.njnt)]
    base = dict.fromkeys(names, 0.)
    if robot == 'kinova':
        base.update(zip(names, [-25, -50, -85, 58, -90, 117]))
        edits = [{}, {'joint_1': -18, 'joint_4': 10}, {'joint_1': 18, 'joint_4': -10},
                 {'joint_2': -14, 'joint_3': 12}, {'joint_2': 12, 'joint_3': -10},
                 {'joint_4': 20, 'joint_6': -20}, {'joint_4': -15, 'joint_6': 20},
                 {'joint_1': 8, 'joint_5': 20, 'joint_6': 10}]
    elif robot == 'g1':
        for side in ('left', 'right'):
            base[side + '_shoulder_pitch_joint'] = 11.46
            base[side + '_shoulder_roll_joint'] = 11.46 if side == 'left' else -11.46
            base[side + '_elbow_joint'] = 73.34
        edits = [{}, {'waist_yaw_joint': 18, 'left_shoulder_pitch_joint': -25},
                 {'waist_yaw_joint': -18, 'right_shoulder_pitch_joint': -25},
                 {'left_shoulder_roll_joint': 30, 'right_shoulder_roll_joint': -30},
                 {'waist_pitch_joint': 12, 'left_elbow_joint': -20, 'right_elbow_joint': -20},
                 {'left_shoulder_pitch_joint': -25, 'right_shoulder_pitch_joint': 20},
                 {'left_shoulder_pitch_joint': 20, 'right_shoulder_pitch_joint': -25},
                 {'waist_roll_joint': 10, 'left_shoulder_roll_joint': 15, 'left_wrist_yaw_joint': 20}]
    elif robot == 'go2':
        for leg in ('FL', 'FR', 'RL', 'RR'):
            base[leg + '_thigh_joint'] = np.degrees(.9)
            base[leg + '_calf_joint'] = np.degrees(-1.8)
        edits = [{}, {'FL_thigh_joint': -15, 'FL_calf_joint': -15},
                 {'FR_thigh_joint': -15, 'FR_calf_joint': -15},
                 {'RL_thigh_joint': -15, 'RL_calf_joint': -15},
                 {'RR_thigh_joint': -15, 'RR_calf_joint': -15},
                 {leg + '_thigh_joint': 15 for leg in ('FL', 'FR', 'RL', 'RR')},
                 {'FL_hip_joint': 18, 'FR_hip_joint': -18},
                 {'RL_hip_joint': 18, 'RR_hip_joint': -18}]
    else:
        base.update(left_arm=60, right_arm=60)
        edits = [{}, {'head_yaw': 25, 'left_arm': -35},
                 {'head_yaw': -25, 'right_arm': -35},
                 {'head_pitch': -20, 'left_arm': -30, 'right_arm': -30},
                 {'head_pitch': 15, 'left_arm': 20, 'right_arm': 20},
                 {'head_roll': 20, 'left_arm': -25},
                 {'head_roll': -20, 'right_arm': -25},
                 {'head_yaw': 12, 'head_pitch': -12, 'left_arm': -40, 'right_arm': -40}]
    labels = ['neutral', 'left', 'right', 'open', 'bow', 'accent_a', 'accent_b', 'finish']
    poses = []
    for label, edit in zip(labels, edits):
        values = [base[name] + edit.get(name, 0) for name in names]
        poses.append({'name': label, 'degrees': values})
    result = {'robot': robot, 'description': TITLES[robot],
              'mode': 'kinematic_fixed_base_no_contacts',
              'limits_note': 'Illustrative experiment parameters, not manufacturer speed ratings.',
              'joint_names': names, 'speed_deg_s': [120.] * len(names),
              'acceleration_deg_s2': [600.] * len(names), 'poses': poses}
    POSES.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')


@dataclass
class Environment:
    robot: str
    model: object
    data: object
    config: dict
    poses: np.ndarray
    speed: np.ndarray
    acceleration: np.ndarray

    def put(self, q, t=0.):
        self.data.qpos[:] = q
        self.data.qvel[:] = 0
        self.data.time = t
        mujoco.mj_forward(self.model, self.data)

    def camera(self):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = CAMERAS[self.robot][0]
        camera.distance = CAMERAS[self.robot][1]
        camera.azimuth = 135
        camera.elevation = -16 if self.robot == 'g1' else -22
        if self.config.get('profile') == 'expressive':
            if self.robot == 'kinova':
                camera.distance = 2.0
            elif self.robot == 'g1':
                camera.lookat[:] = [0, 0, .86]
                camera.distance = 2.65
        return camera

    def times(self):
        distance = np.abs(self.poses[:, None, :] - self.poses[None, :, :])
        # max(s') = 15/8; max(abs(s'')) = 10/sqrt(3) for s=10u^3-15u^4+6u^5.
        velocity_time = np.max(1.875 * distance / self.speed, axis=-1)
        acceleration_time = np.sqrt(np.max((10 / np.sqrt(3)) * distance / self.acceleration, axis=-1))
        return np.maximum(velocity_time, acceleration_time)


def load(robot, profile='default'):
    if robot not in ROBOT_IDS:
        raise ValueError(f'Unknown robot: {robot}')
    path = MODELS / robot / 'model.xml'
    if not path.exists():
        raise ValueError('Models are not prepared. Run Setup_Sim.ps1 or sim.py setup first.')
    model = mujoco.MjModel.from_xml_path(str(path))
    if model.nq != model.njnt or np.any(model.jnt_type != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError('The sandbox expects a fixed base and scalar hinge joints.')
    if profile == 'default':
        write_default_poses(robot, model)
        pose_file = POSES/(robot+'.json')
    elif profile == 'expressive':
        from .expressive import write_expressive_poses
        write_expressive_poses(robot, model)
        pose_file = POSES/'expressive'/(robot+'.json')
    else:
        raise ValueError(f'Unknown pose profile: {profile}')
    config = json.loads(pose_file.read_text(encoding='utf-8'))
    names = [model.joint(i).name for i in range(model.njnt)]
    if config['joint_names'] != names:
        raise ValueError(f'{robot}: pose joint_names must match the model order.')
    if len(config['poses']) < 2 or len({p['name'] for p in config['poses']}) != len(config['poses']):
        raise ValueError('Define at least two uniquely named poses.')
    values = np.asarray([p['degrees'] for p in config['poses']], dtype=float)
    speed = np.asarray(config['speed_deg_s'], dtype=float)
    acceleration = np.asarray(config['acceleration_deg_s2'], dtype=float)
    if values.shape != (len(config['poses']), model.nq):
        raise ValueError('Each pose must provide exactly one angle per joint.')
    if speed.shape != (model.nq,) or acceleration.shape != (model.nq,):
        raise ValueError('Limit arrays must provide exactly one value per joint.')
    if not all(np.all(np.isfinite(x)) for x in (values, speed, acceleration)):
        raise ValueError('Pose angles and limits must be finite.')
    if np.any(speed <= 0) or np.any(acceleration <= 0):
        raise ValueError('Speed and acceleration parameters must be positive.')
    poses = np.radians(values)
    limited = model.jnt_limited.astype(bool)
    if np.any(poses[:, limited] < model.jnt_range[limited, 0] - 1e-6) or np.any(
            poses[:, limited] > model.jnt_range[limited, 1] + 1e-6):
        raise ValueError(f'{robot}: a configured pose is outside the model joint ranges.')
    if any(np.allclose(poses[i], poses[j]) for i in range(len(poses)) for j in range(i)):
        raise ValueError('Pose angles must be distinct; duplicated states inflate diversity.')
    env = Environment(robot, model, mujoco.MjData(model), config, poses,
                      np.radians(speed), np.radians(acceleration))
    env.put(poses[0])
    return env


def make_dance(env, bpm=90., seed=7):
    if env.config.get('profile') == 'expressive':
        from .expressive import PhraseDance
        return PhraseDance(env, bpm, seed)
    return Dance(env, bpm, seed)


def graph(env, bpm):
    if not np.isfinite(bpm) or bpm <= 0:
        raise ValueError('BPM must be finite and positive.')
    adjacency = env.times() <= 60 / bpm + 1e-12
    np.fill_diagonal(adjacency, False)  # Holding a pose is not counted as a dance transition.
    radius = float(np.max(np.abs(np.linalg.eigvals(adjacency.astype(float)))))
    capacity = float(np.log(max(1., radius)))
    return adjacency, capacity


class Dance:
    def __init__(self, env, bpm=90., seed=7):
        self.env, self.bpm = env, bpm
        self.adjacency, self.capacity = graph(env, bpm)
        # These single-path, symmetric timing rules produce an undirected graph.
        # Start in the largest connected component with at least one edge.
        unseen = set(range(len(env.poses)))
        components = []
        while unseen:
            frontier = [min(unseen)]; component = set()
            while frontier:
                node = frontier.pop()
                if node in component:
                    continue
                component.add(node)
                frontier.extend(i for i in np.flatnonzero(self.adjacency[node]) if i not in component)
            unseen -= component
            if len(component) > 1:
                components.append(sorted(component))
        if not components:
            raise ValueError(f'{env.robot}: no sustainable moving sequence at {bpm:g} BPM. Lower BPM or edit poses/limits.')
        self.component = max(components, key=len)
        self.current = self.component[0]
        self.sequence = [self.current]
        self.visits = np.zeros(len(env.poses), dtype=int)
        self.visits[self.current] = 1
        self.edges = np.zeros(self.adjacency.shape, dtype=int)
        self.rng = np.random.default_rng(seed)

    def extend(self):
        current = self.sequence[-1]
        options = np.flatnonzero(self.adjacency[current])
        weights = 1 / (1 + self.edges[current, options]) / np.sqrt(1 + self.visits[options])
        target = int(self.rng.choice(options, p=weights / weights.sum()))
        self.edges[current, target] += 1
        self.visits[target] += 1
        self.sequence.append(target)

    def at(self, t):
        beat = max(0., t * self.bpm / 60)
        index = int(np.floor(beat))
        # A target is recorded as visited only when the next whole beat is reached.
        while len(self.sequence) <= index:
            self.extend()
        if beat - index < 1e-12:
            return self.env.poses[self.sequence[index]].copy()
        while len(self.sequence) <= index + 1:
            self.extend()
        u = beat - index
        blend = u**3 * (10 + u * (-15 + 6*u))
        a, b = self.env.poses[self.sequence[index:index+2]]
        return (1-blend)*a + blend*b

    def report(self, seconds):
        reached = int(np.floor(seconds * self.bpm / 60 + 1e-9))
        while len(self.sequence) <= reached:
            self.extend()
        sequence = self.sequence[:reached+1]
        pairs = set(zip(sequence[:-1], sequence[1:]))
        return {'robot': self.env.robot, 'mode': 'kinematic_fixed_base_no_contacts',
                'bpm': self.bpm, 'seconds': seconds, 'pose_count': len(self.env.poses),
                'graph_edges': int(self.adjacency.sum()), 'freestyle_capacity': self.capacity,
                'reachable_pose_count': len(self.component), 'visited_poses': len(set(sequence)),
                'visited_directed_edges': len(pairs), 'completed_transitions': len(sequence)-1,
                'sequence': [self.env.config['poses'][i]['name'] for i in sequence],
                'timing_model': 'sufficient quintic path time; not global minimum time'}
