"""Virtual-only MuJoCo environments. This entry point never connects to hardware."""
import argparse
import json
from pathlib import Path
import struct
import time
import zlib
import numpy as np
import mujoco
from rita_sim.assets import setup, MODELS, ROOT, ROBOT_IDS
from rita_sim.core import load, Dance, make_dance, TITLES, graph

OUTPUT = ROOT / 'sim_output'


def save_png(path, pixels):
    """Write RGB arrays with the standard library, avoiding an image dependency."""
    pixels = np.asarray(pixels, dtype=np.uint8)
    height, width, _ = pixels.shape
    def chunk(tag, data):
        return struct.pack('!I', len(data)) + tag + data + struct.pack('!I', zlib.crc32(tag+data) & 0xffffffff)
    raw = b''.join(b'\x00' + row.tobytes() for row in pixels)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
    Path(path).write_bytes(png + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def selection(args):
    return ROBOT_IDS if args.robot == 'all' else (args.robot,)


def check(args):
    reports = []
    for robot in selection(args):
        env = load(robot, args.profile)
        dance = make_dance(env, args.bpm, args.seed)
        # Exercise every declared pose and a dense beat-synchronous trajectory.
        for pose in env.poses:
            env.put(pose)
            if not np.all(np.isfinite(env.data.xpos)):
                raise RuntimeError(f'{robot}: nonfinite forward kinematics')
        began = time.perf_counter()
        for t in np.linspace(0, args.seconds, max(2, int(args.seconds*120)+1)):
            env.put(dance.at(t), t)
        report = dance.report(args.seconds)
        report.update(joints=env.model.njnt, contacts=int(env.data.ncon),
                      check_wall_seconds=time.perf_counter()-began)
        reports.append(report)
        print(f"OK {TITLES[robot]} | {report['joints']} joints | "
              f"{report['visited_poses']}/{report['pose_count']} poses | "
              f"h={report['freestyle_capacity']:.3f} | {report['contacts']} contacts")
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/'check.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


def snapshot(args):
    OUTPUT.mkdir(exist_ok=True)
    pictures = []
    for robot in selection(args):
        env = load(robot, args.profile)
        pose = args.pose
        if not 0 <= pose < len(env.poses):
            raise ValueError('--pose is outside the configured pose library.')
        env.put(env.poses[pose])
        with mujoco.Renderer(env.model, height=480, width=640) as renderer:
            option = mujoco.MjvOption()
            option.geomgroup[3] = 0
            renderer.update_scene(env.data, camera=env.camera(), scene_option=option)
            pixels = renderer.render().copy()
        destination = OUTPUT/f'{robot}.png'
        save_png(destination, pixels)
        pictures.append(pixels)
        print(destination)
    if len(pictures) == 4:
        grid = np.concatenate((np.concatenate(pictures[:2], axis=1),
                               np.concatenate(pictures[2:], axis=1)), axis=0)
        save_png(OUTPUT/'overview.png', grid)
        print('Overview order: Kinova, G1 / Misty simplified, Go2')


def dance(args):
    if args.robot == 'all':
        raise ValueError('Choose one robot for the viewer, or use check/snapshot --robot all.')
    env = load(args.robot, args.profile)
    controller = make_dance(env, args.bpm, args.seed)
    if args.headless:
        for t in np.linspace(0, args.seconds, max(2, int(args.seconds*120)+1)):
            env.put(controller.at(t), t)
        elapsed = args.seconds
    else:
        import mujoco.viewer
        paused = [False]
        def on_key(key):
            if key == 32:
                paused[0] = not paused[0]
        print(f'{TITLES[args.robot]} | {args.bpm:g} BPM | {len(env.poses)} editable finite poses')
        print('SPACE: pause/resume | close the viewer: finish | mouse: orbit/zoom')
        print('Mode: kinematic, fixed base, contacts disabled.')
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key,
                                         show_left_ui=False, show_right_ui=False) as viewer:
            viewer.cam.lookat[:] = env.camera().lookat
            viewer.cam.distance = env.camera().distance
            viewer.cam.azimuth = env.camera().azimuth
            viewer.cam.elevation = env.camera().elevation
            viewer.opt.geomgroup[3] = 0
            elapsed = 0.
            last = time.perf_counter()
            while viewer.is_running() and elapsed < args.seconds:
                began = time.perf_counter()
                if not paused[0]:
                    elapsed = min(args.seconds, elapsed + began-last)
                last = began
                with viewer.lock():
                    env.put(controller.at(elapsed), elapsed)
                viewer.sync()
                time.sleep(max(0., 1/60 - (time.perf_counter()-began)))
    report = controller.report(elapsed)
    OUTPUT.mkdir(exist_ok=True)
    destination = OUTPUT/f'{args.robot}_last_run.json'
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def sweep(args):
    OUTPUT.mkdir(exist_ok=True)
    rows = []
    for robot in selection(args):
        env = load(robot, args.profile)
        for bpm in range(40, 241):
            if args.profile == 'expressive':
                try:
                    controller = make_dance(env, bpm, args.seed)
                    edges, capacity = int(controller.adjacency.sum()), controller.capacity
                except ValueError:
                    edges, capacity = 0, 0.
            else:
                adjacency, capacity = graph(env, bpm)
                edges = int(adjacency.sum())
            rows.append({'robot': robot, 'profile': args.profile, 'bpm': bpm,
                         'edges': edges, 'capacity': capacity})
    (OUTPUT/'bpm_sweep.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(OUTPUT/'bpm_sweep.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['setup', 'list', 'check', 'snapshot', 'dance', 'sweep'])
    parser.add_argument('--robot', choices=[*ROBOT_IDS, 'all'], default=None)
    parser.add_argument('--bpm', type=float, default=90.)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--profile', choices=['default', 'expressive'], default='default')
    parser.add_argument('--pose', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    args.robot = args.robot or ('kinova' if args.command == 'dance' else 'all')
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be finite and positive')
    try:
        if args.command == 'setup':
            setup()
            for robot in ROBOT_IDS:
                load(robot)
        elif args.command == 'list':
            for robot in ROBOT_IDS:
                ready = (MODELS/robot/'model.xml').exists()
                print(f'{robot:8} {TITLES[robot]:30} {"ready" if ready else "run setup"}')
        else:
            globals()[args.command](args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Simulation error: {exc}\n')


if __name__ == '__main__':
    main()
