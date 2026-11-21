"""Expanded finite-pose vocabularies and musical 1/2/4-beat transitions."""
import json
from bisect import bisect_right
import numpy as np
from .assets import ROOT


def write_expressive_poses(robot, model):
    destination = ROOT/'sim_poses/expressive'/f'{robot}.json'
    if destination.exists():
        return
    names = [model.joint(i).name for i in range(model.njnt)]
    base = dict.fromkeys(names, 0.)
    if robot == 'kinova':
        rows = [
            ('neutral', [-25,-50,-85,58,-90,117]),
            ('sweep_left', [-90,-25,-70,85,-55,60]),
            ('sweep_right', [45,-25,-70,-15,-55,165]),
            ('high_reach', [-20,-85,-30,25,-40,90]),
            ('low_curl', [-25,-10,-100,95,-105,140]),
            ('wide_left', [-80,-60,-45,10,-35,20]),
            ('wide_right', [35,-65,-45,100,-35,170]),
            ('tight_curl', [-15,-30,-125,115,-90,45]),
            ('long_reach', [-25,-85,-20,-35,-35,155]),
            ('wrist_fan_left', [-55,-45,-70,-65,-30,-10]),
            ('wrist_fan_right', [15,-45,-70,145,-100,210]),
            ('point_left', [-95,-40,-60,55,-65,105]),
            ('point_right', [50,-40,-60,55,-65,105]),
            ('overhead', [-10,-100,-10,50,-20,100]),
            ('diagonal', [-65,-75,-35,120,-90,150]),
            ('finale', [10,-70,-35,-15,-45,65])]
    elif robot == 'misty':
        rows = [
            ('neutral', [55,55,0,0,0]), ('wave_left', [-20,80,55,25,-10]),
            ('wave_right', [80,-20,-55,-25,-10]), ('hands_up', [-25,-25,0,0,-30]),
            ('deep_bow', [85,85,0,0,23]), ('tilt_left', [5,70,30,32,-15]),
            ('tilt_right', [70,5,-30,-32,-15]), ('look_left', [40,30,65,-15,10]),
            ('look_right', [30,40,-65,15,10]), ('alternate_left', [-20,85,0,-20,-20]),
            ('alternate_right', [85,-20,0,20,-20]), ('peek', [-25,-25,45,20,12]),
            ('shrug', [10,10,-45,-20,-25]), ('reach_front', [0,0,0,0,15]),
            ('diagonal', [15,75,50,-30,20]), ('finale', [-25,-25,-10,15,-30])]
    elif robot == 'go2':
        for leg in ('FL', 'FR', 'RL', 'RR'):
            base.update({leg+'_hip_joint': 0., leg+'_thigh_joint': 52., leg+'_calf_joint': -103.})
        def legs(**values):
            result = {}
            for leg, angles in values.items():
                result.update(zip([leg+'_hip_joint', leg+'_thigh_joint', leg+'_calf_joint'], angles))
            return result
        edits = [
            ('neutral', {}), ('front_left_up', legs(FL=[15,0,-145])),
            ('front_right_up', legs(FR=[-15,0,-145])), ('rear_left_up', legs(RL=[20,80,-145])),
            ('rear_right_up', legs(RR=[-20,80,-145])),
            ('front_bow', legs(FL=[0,90,-130], FR=[0,90,-130], RL=[0,35,-75], RR=[0,35,-75])),
            ('rear_bow', legs(FL=[0,35,-70], FR=[0,35,-70], RL=[0,90,-140], RR=[0,90,-140])),
            ('wide_stance', legs(FL=[30,60,-105], FR=[-30,60,-105], RL=[25,60,-105], RR=[-25,60,-105])),
            ('tuck', legs(**{leg:[0,20,-150] for leg in ('FL','FR','RL','RR')})),
            ('extend', legs(**{leg:[0,70,-55] for leg in ('FL','FR','RL','RR')})),
            ('diagonal_a', legs(FL=[20,0,-140], RR=[-20,0,-140], FR=[0,65,-100], RL=[0,65,-100])),
            ('diagonal_b', legs(FR=[-20,0,-140], RL=[20,0,-140], FL=[0,65,-100], RR=[0,65,-100])),
            ('rock_left', legs(FL=[25,75,-120], RL=[25,75,-120], FR=[-10,30,-100], RR=[-10,30,-100])),
            ('rock_right', legs(FR=[-25,75,-120], RR=[-25,75,-120], FL=[10,30,-100], RL=[10,30,-100])),
            ('paw_salute', legs(FL=[35,-15,-145], FR=[-10,65,-95], RL=[15,70,-110], RR=[-15,40,-100])),
            ('finale', legs(FL=[35,45,-95], FR=[-35,45,-95], RL=[30,45,-95], RR=[-30,45,-95]))]
        rows = [(label, [edit.get(name, base[name]) for name in names]) for label, edit in edits]
    else:
        for side in ('left', 'right'):
            base.update({side+'_hip_pitch_joint': -8, side+'_knee_joint': 16,
                         side+'_ankle_pitch_joint': -8, side+'_shoulder_pitch_joint': 5,
                         side+'_shoulder_roll_joint': 12 if side == 'left' else -12,
                         side+'_elbow_joint': 40})
        def side(which, **angles):
            return {which+'_'+joint+'_joint': angle for joint, angle in angles.items()}
        def both(**angles):
            return {**side('left', **angles), **side('right', **angles)}
        edits = [
            ('neutral', {}),
            ('reach_left', {**side('left', shoulder_pitch=-95, shoulder_roll=35, elbow=15,
                                  wrist_pitch=-20, hip_pitch=-25, knee=40, ankle_pitch=10),
                            'waist_yaw_joint':20, 'waist_roll_joint':-8}),
            ('reach_right', {**side('right', shoulder_pitch=-95, shoulder_roll=-35, elbow=15,
                                   wrist_pitch=-20, hip_pitch=-25, knee=40, ankle_pitch=10),
                             'waist_yaw_joint':-20, 'waist_roll_joint':8}),
            ('star', {**both(shoulder_pitch=-20, elbow=15, hip_pitch=-12, knee=25),
                      'left_shoulder_roll_joint':95, 'right_shoulder_roll_joint':-95,
                      'left_hip_roll_joint':10, 'right_hip_roll_joint':-10, 'waist_pitch_joint':-5}),
            ('hands_up', {**both(shoulder_pitch=-135, elbow=25, hip_pitch=-15, knee=25),
                          'left_shoulder_roll_joint':15, 'right_shoulder_roll_joint':-15}),
            ('deep_bow', {**both(hip_pitch=-25, knee=55, ankle_pitch=-25, shoulder_pitch=35, elbow=10),
                          'waist_pitch_joint':25}),
            ('disco_left', {**side('left', shoulder_pitch=-125, shoulder_roll=55, elbow=15, wrist_yaw=45),
                            **side('right', shoulder_pitch=35, shoulder_roll=-30, elbow=100, wrist_yaw=-45),
                            'waist_yaw_joint':30, 'waist_roll_joint':-15}),
            ('disco_right', {**side('right', shoulder_pitch=-125, shoulder_roll=-55, elbow=15, wrist_yaw=-45),
                             **side('left', shoulder_pitch=35, shoulder_roll=30, elbow=100, wrist_yaw=45),
                             'waist_yaw_joint':-30, 'waist_roll_joint':15}),
            ('twist_left', {**side('left', shoulder_pitch=-45, shoulder_roll=45, elbow=95, wrist_roll=55,
                                  hip_yaw=-12, knee=45),
                            **side('right', shoulder_pitch=20, shoulder_roll=-45, elbow=30, wrist_roll=-55,
                                   hip_yaw=8, knee=18), 'waist_yaw_joint':-45, 'waist_roll_joint':-15}),
            ('twist_right', {**side('right', shoulder_pitch=-45, shoulder_roll=-45, elbow=95, wrist_roll=-55,
                                   hip_yaw=12, knee=45),
                             **side('left', shoulder_pitch=20, shoulder_roll=45, elbow=30, wrist_roll=55,
                                    hip_yaw=-8, knee=18), 'waist_yaw_joint':45, 'waist_roll_joint':15}),
            ('step_left', {**side('left', hip_pitch=-45, knee=85, ankle_pitch=-25, shoulder_pitch=35, elbow=35),
                           **side('right', shoulder_pitch=-70, elbow=35)}),
            ('step_right', {**side('right', hip_pitch=-45, knee=85, ankle_pitch=-25, shoulder_pitch=35, elbow=35),
                            **side('left', shoulder_pitch=-70, elbow=35)}),
            ('crouch_reach', {**both(hip_pitch=-35, knee=85, ankle_pitch=-35, shoulder_pitch=-70, elbow=15),
                              'left_shoulder_roll_joint':20, 'right_shoulder_roll_joint':-20,
                              'waist_pitch_joint':12}),
            ('salute', {**side('left', shoulder_pitch=-95, shoulder_roll=25, elbow=105, wrist_roll=70),
                        **side('right', shoulder_roll=-20, elbow=35), 'waist_yaw_joint':15}),
            ('cross', {**both(shoulder_pitch=-60, elbow=100, wrist_pitch=25),
                       'left_shoulder_roll_joint':-30, 'right_shoulder_roll_joint':30,
                       'left_shoulder_yaw_joint':40, 'right_shoulder_yaw_joint':-40}),
            ('finale', {**both(shoulder_pitch=-70, elbow=10, hip_pitch=-15, knee=30),
                        'left_shoulder_roll_joint':105, 'right_shoulder_roll_joint':-105,
                        'left_wrist_yaw_joint':50, 'right_wrist_yaw_joint':-50,
                        'left_hip_roll_joint':12, 'right_hip_roll_joint':-12, 'waist_pitch_joint':-10})]
        rows = [(label, [edit.get(name, base[name]) for name in names]) for label, edit in edits]
    config = {'robot': robot, 'profile': 'expressive', 'mode': 'kinematic_fixed_base_no_contacts',
              'joint_names': names, 'speed_deg_s': [120.]*len(names),
              'acceleration_deg_s2': [600.]*len(names),
              'limits_note': 'Illustrative simulation parameters; not physical hardware ratings.',
              'allowed_move_beats': [1, 2, 4],
              'poses': [{'name': label, 'degrees': values} for label, values in rows]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2), encoding='utf-8')


class PhraseDance:
    """Select large, varied finite poses and land on integer music beats."""
    def __init__(self, env, bpm=117.0254648, seed=10):
        if not np.isfinite(bpm) or bpm <= 0:
            raise ValueError('BPM must be finite and positive.')
        self.env, self.bpm = env, bpm
        self.period = 60/bpm
        required = env.times()/self.period
        self.durations = np.zeros(required.shape, dtype=int)
        for beats in (4, 2, 1):
            self.durations[required <= beats+1e-10] = beats
        np.fill_diagonal(self.durations, 0)
        self.adjacency = self.durations > 0
        self.component = self._largest_component()
        if len(self.component) < 2:
            raise ValueError(f'{env.robot}: no sustainable sequence in four beats at {bpm:g} BPM.')
        self.capacity = self._capacity()
        self.sequence = [self.component[0]]
        self.arrival_beats = [0]
        self.visits = np.zeros(len(env.poses), dtype=int)
        self.visits[self.sequence[0]] = 1
        self.edges = np.zeros(self.adjacency.shape, dtype=int)
        self.rng = np.random.default_rng(seed)

    def _largest_component(self):
        unseen, components = set(range(len(self.env.poses))), []
        while unseen:
            component, frontier = set(), [min(unseen)]
            while frontier:
                node = frontier.pop()
                if node in component:
                    continue
                component.add(node)
                frontier.extend(i for i in np.flatnonzero(self.adjacency[node]) if i not in component)
            unseen -= component
            components.append(sorted(component))
        return max(components, key=len)

    def _capacity(self):
        # Variable-duration edges require a time-weighted entropy calculation.
        # rho(A_ij * exp(-h * duration_ij)) = 1 gives capacity per music beat.
        low, high = 0., float(np.log(len(self.env.poses)))
        for _ in range(60):
            middle = (low+high)/2
            weighted = self.adjacency*np.exp(-middle*self.durations)
            radius = np.max(np.abs(np.linalg.eigvals(weighted)))
            if radius > 1:
                low = middle
            else:
                high = middle
        return (low+high)/2

    def extend(self):
        current = self.sequence[-1]
        options = np.flatnonzero(self.adjacency[current])
        fresh = np.array([i for i in options if i not in self.sequence[-3:]], dtype=int)
        if len(fresh):
            options = fresh
        options = options[self.visits[options] == self.visits[options].min()]
        distance = np.linalg.norm(self.env.poses[options]-self.env.poses[current], axis=1)
        weights = (distance+.1)/(1+self.edges[current, options]) / np.sqrt(self.durations[current, options])
        target = int(self.rng.choice(options, p=weights/weights.sum()))
        self.arrival_beats.append(self.arrival_beats[-1]+int(self.durations[current, target]))
        self.sequence.append(target)
        self.visits[target] += 1
        self.edges[current, target] += 1

    def video_state(self, elapsed):
        beat = max(0., elapsed/self.period)
        while self.arrival_beats[-1] <= beat+1e-10:
            self.extend()
        index = max(0, bisect_right(self.arrival_beats, beat+1e-10)-1)
        return elapsed, index, self.sequence[index], self.sequence[index+1], self.sequence[:index+1]

    def at(self, elapsed):
        _, index, current, target, _ = self.video_state(elapsed)
        u = np.clip((max(0., elapsed)/self.period-self.arrival_beats[index]) /
                    (self.arrival_beats[index+1]-self.arrival_beats[index]), 0, 1)
        blend = u**3*(10+u*(-15+6*u))
        return (1-blend)*self.env.poses[current]+blend*self.env.poses[target]

    def report(self, seconds):
        _, index, _, _, completed = self.video_state(seconds)
        pairs = set(zip(completed[:-1], completed[1:]))
        return {'robot': self.env.robot, 'profile': 'expressive', 'mode': 'kinematic_fixed_base_no_contacts',
                'bpm': self.bpm, 'seconds': seconds, 'pose_count': len(self.env.poses),
                'graph_edges': int(self.adjacency.sum()), 'freestyle_capacity': self.capacity,
                'capacity_units': 'per music beat, accounting for variable transition durations',
                'reachable_pose_count': len(self.component), 'visited_poses': len(set(completed)),
                'visited_directed_edges': len(pairs), 'completed_transitions': index,
                'arrival_beats': self.arrival_beats[:index+1],
                'sequence': [self.env.config['poses'][i]['name'] for i in completed],
                'timing_model': 'quintic transitions over 1, 2, or 4 music beats; no global minimum-time claim'}
