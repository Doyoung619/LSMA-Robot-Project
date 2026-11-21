"""Render a music-synchronized four-robot MP4 without intermediate frame files."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rita_sim.assets import ROOT, ROBOT_IDS
from rita_sim.core import load, Dance, make_dance

WIDTH, HEIGHT = 1920, 1080
BACKGROUND = (9, 14, 23)
PANEL = (13, 20, 31)
INK = (233, 240, 248)
MUTED = (139, 156, 179)
ACCENTS = [(75, 218, 226), (165, 149, 255), (255, 190, 104), (116, 220, 165)]
LABELS = {'kinova': ('KINOVA GEN3', '6-DOF ARM'), 'g1': ('UNITREE G1', '29-JOINT HUMANOID'),
          'misty': ('MISTY', 'SIMPLIFIED / 5 JOINTS'), 'go2': ('UNITREE GO2', '12-JOINT QUADRUPED')}


def font(size, bold=False):
    for directory in (Path('C:/Windows/Fonts'), Path('/usr/share/fonts/truetype/dejavu')):
        name = ('segoeuib.ttf' if bold else 'segoeui.ttf') if directory.drive else (
            'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
        path = directory/name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


FONTS = {name: font(size, bold) for name, size, bold in (
    ('title', 43, True), ('bpm', 38, True), ('robot', 25, True), ('small', 18, False),
    ('label', 16, True), ('tiny', 14, False), ('number', 24, True))}


def beat_state(controller, music_time, first_beat):
    elapsed = max(0., music_time-first_beat)
    if hasattr(controller, 'video_state'):
        state = controller.video_state(elapsed)
        return (state[0], state[1], state[2], state[3] if music_time >= first_beat else state[2], state[4])
    controller.at(elapsed)
    index = int(math.floor(elapsed*controller.bpm/60 + 1e-10))
    while len(controller.sequence) <= index+1:
        controller.extend()
    current = controller.sequence[index]
    target = controller.sequence[index+1] if music_time >= first_beat else current
    completed = controller.sequence[:index+1]
    return elapsed, index, current, target, completed


def graph_panel(controller, state, accent):
    _, _, current, target, completed = state
    tile = Image.new('RGB', (190, 330), PANEL)
    draw = ImageDraw.Draw(tile)
    draw.text((10, 12), 'DANCE GRAPH', font=FONTS['label'], fill=MUTED)
    draw.text((10, 39), f'h = {controller.capacity:.3f}', font=FONTS['number'], fill=accent)
    count = len(controller.env.poses)
    points = [(95 + 65*math.cos(-math.pi/2+2*math.pi*i/count),
               153 + 65*math.sin(-math.pi/2+2*math.pi*i/count)) for i in range(count)]
    for i in range(count):
        for j in range(i):
            if controller.adjacency[i, j]:
                draw.line([points[i], points[j]], fill=(40, 53, 70), width=1)
    used_edges = set(zip(completed[:-1], completed[1:]))
    for i, j in used_edges:
        draw.line([points[i], points[j]], fill=tuple(int(c*.60) for c in accent), width=2)
    if current != target:
        draw.line([points[current], points[target]], fill=INK, width=3)
    for i, (x, y) in enumerate(points):
        radius = 11 if i == current else 9
        color = INK if i == current else accent if i in completed else (52, 64, 81)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        draw.text((x, y), str(i+1), font=FONTS['tiny'], fill=BACKGROUND, anchor='mm')
    draw.text((10, 244), f'POSES   {len(set(completed))}/{count}', font=FONTS['label'], fill=INK)
    draw.text((10, 273), f'EDGES   {len(used_edges)}/{int(controller.adjacency.sum())}',
              font=FONTS['label'], fill=MUTED)
    caption = '1 / 2 / 4 BEAT MOVES' if hasattr(controller, 'durations') else 'Less-used moves favored'
    draw.text((10, 304), caption, font=FONTS['tiny'], fill=MUTED)
    return tile


def render(args):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise ValueError('ffmpeg and ffprobe must be available on PATH.')
    info = json.loads(args.analysis.read_text(encoding='utf-8'))
    bpm, period, first = info['bpm'], info['period_seconds'], info['first_beat_seconds']
    if not all(np.isfinite(x) for x in (bpm, period, first, args.start, args.seconds)):
        raise ValueError('Music timing must be finite.')
    if abs(60/bpm-period) > 1e-8 or period <= 0:
        raise ValueError('Inconsistent BPM and beat period.')
    if args.start < 0 or args.seconds <= 0 or args.start+args.seconds > info['duration_seconds']:
        raise ValueError('Choose a positive clip duration within the source music.')
    if args.fps < 1 or args.fps > 120:
        raise ValueError('Choose an FPS between 1 and 120.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = round(args.seconds*args.fps)
    duration = frames/args.fps
    environments = [load(robot, args.profile) for robot in ROBOT_IDS]
    controllers = [make_dance(env, bpm, args.seed+i) for i, env in enumerate(environments)]
    for controller in controllers:
        controller.at(max(0., args.start-first))
    locations = [(32, 112), (970, 112), (32, 558), (970, 558)]
    base = Image.new('RGB', (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(base)
    subtitle = 'RITA  /  FOUR ROBOTS, ONE BEAT' + ('  /  EXPANDED MOTION' if args.profile == 'expressive' else '')
    draw.text((34, 16), subtitle, font=FONTS['label'], fill=MUTED)
    draw.text((31, 41), 'BILLIE JEAN', font=FONTS['title'], fill=INK)
    draw.text((WIDTH-34, 44), f'{bpm:.3f} BPM', font=FONTS['bpm'], fill=ACCENTS[0], anchor='ra')
    draw.text((WIDTH-35, 90), f'MEASURED FROM AUDIO  /  {period*1000:.3f} ms PER BEAT',
              font=FONTS['tiny'], fill=MUTED, anchor='ra')
    for (x, y), env, accent in zip(locations, environments, ACCENTS):
        draw.rounded_rectangle((x, y, x+918, y+426), radius=12, fill=PANEL,
                               outline=(42, 54, 71), width=1)
        draw.rounded_rectangle((x+14, y+17, x+19, y+37), radius=2, fill=accent)
        draw.text((x+30, y+10), LABELS[env.robot][0], font=FONTS['robot'], fill=INK)
        draw.text((x+897, y+20), LABELS[env.robot][1], font=FONTS['tiny'], fill=MUTED, anchor='ra')
    draw.text((32, 1043), 'MuJoCo  /  KINEMATIC FIXED-BASE STUDY  /  CONTACTS OFF',
              font=FONTS['tiny'], fill=MUTED)
    draw.text((1888, 1043), 'MUSIC CLOCK + BEAT PHASE', font=FONTS['tiny'], fill=MUTED, anchor='ra')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y',
               '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{WIDTH}x{HEIGHT}',
               '-framerate', str(args.fps), '-i', 'pipe:0',
               '-ss', str(args.start), '-i', str(args.music),
               '-map', '0:v:0', '-map', '1:a:0', '-t', str(duration),
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '21', '-pix_fmt', 'yuv420p',
               '-c:a', 'aac', '-b:a', '192k',
               '-af', f'afade=t=out:st={max(0,duration-.7)}:d=0.7',
               '-movflags', '+faststart', '-metadata', 'title=Rita - Billie Jean - Four Robot Dance',
               str(args.output)]
    logfile = args.output.with_suffix('.encoder.log')
    began = time.perf_counter()
    representative_frames = {0, min(frames-1, round(10*args.fps)), frames-1}
    with ExitStack() as stack:
        options = []
        for env in environments:
            env.model.vis.global_.offwidth = 700
            env.model.vis.global_.offheight = 330
            env.model.vis.quality.offsamples = 2
            env.model.vis.quality.shadowsize = 1024
            env.model.light_castshadow[:] = 0
            env.model.light_castshadow[0] = 1
            # Softer light retains detail in the white meshes.
            env.model.light_diffuse[:] *= .62
            env.model.light_specular[:] *= .25
            env.model.vis.headlight.diffuse[:] = [.38]*3
            env.model.vis.headlight.ambient[:] = [.24]*3
            option = mujoco.MjvOption()
            option.geomgroup[3] = 0
            options.append(option)
        renderers = [stack.enter_context(mujoco.Renderer(env.model, height=330, width=700))
                     for env in environments]
        graph_cache = [None]*4
        graph_indices = [None]*4
        error_stream = stack.enter_context(logfile.open('w', encoding='utf-8'))
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_stream)
        try:
            for frame_index in range(frames):
                video_time = frame_index/args.fps
                music_time = args.start+video_time
                frame = base.copy()
                overlay = ImageDraw.Draw(frame)
                for i, (env, controller, renderer, option, (x, y), accent) in enumerate(zip(
                        environments, controllers, renderers, options, locations, ACCENTS)):
                    state = beat_state(controller, music_time, first)
                    elapsed, index, current, target, completed = state
                    env.put(controller.at(elapsed), music_time)
                    camera = env.camera()
                    camera.distance *= .88
                    renderer.update_scene(env.data, camera=camera, scene_option=option)
                    picture = renderer.render()
                    frame.paste(Image.fromarray(picture), (x+12, y+52))
                    if graph_indices[i] != (index, target):
                        graph_cache[i] = graph_panel(controller, state, accent)
                        graph_indices[i] = (index, target)
                    frame.paste(graph_cache[i], (x+718, y+52))
                    names = env.config['poses']
                    title = f"{names[current]['name'].replace('_',' ').upper()}  >  {names[target]['name'].replace('_',' ').upper()}"
                    overlay.text((x+18, y+394), title, font=FONTS['label'], fill=accent)
                    suffix = f'TRANSITION {index:03}'
                    if hasattr(controller, 'durations'):
                        suffix += f'  /  {controller.durations[current, target] if current != target else 0} BEATS'
                    overlay.text((x+690, y+394), suffix, font=FONTS['tiny'], fill=MUTED, anchor='ra')
                phase = (music_time-first)/period
                beat_index = max(0, int(math.floor(phase))+1)
                beat_fraction = phase-math.floor(phase)
                pulse = math.exp(-beat_fraction*9) if music_time >= first else 0
                circle_color = tuple(int(MUTED[k]*(1-pulse)+ACCENTS[0][k]*pulse) for k in range(3))
                overlay.ellipse((35, 1003, 51, 1019), fill=circle_color)
                overlay.text((65, 998), f'BEAT {beat_index:03}', font=FONTS['label'], fill=INK)
                for tick in range(32):
                    left = 208+tick*43
                    color = ACCENTS[0] if beat_index and tick == (beat_index-1)%32 else (45, 60, 78)
                    overlay.rounded_rectangle((left, 1008, left+31, 1015), radius=3, fill=color)
                clock = f'{int(music_time)//60:02}:{music_time%60:05.2f}'
                overlay.text((1888, 996), clock, font=FONTS['number'], fill=INK, anchor='ra')
                # A short final visual fade matches the audio fade without changing beat timing.
                if duration-video_time < .7:
                    frame = Image.blend(Image.new('RGB', frame.size, BACKGROUND), frame,
                                        max(0., (duration-video_time)/.7))
                if frame_index in representative_frames:
                    frame.save(args.output.with_name(args.output.stem+f'_frame_{frame_index:05}.jpg'), quality=92)
                encoder.stdin.write(frame.tobytes())
                if frame_index % (args.fps*5) == 0 or frame_index == frames-1:
                    spent = time.perf_counter()-began
                    rate = (frame_index+1)/spent
                    remaining = (frames-frame_index-1)/max(rate, 1e-6)
                    print(f'Render {frame_index+1}/{frames} | {rate:.1f} fps | ETA {remaining:.0f}s', flush=True)
            encoder.stdin.close()
            status = encoder.wait(timeout=120)
            if status:
                raise RuntimeError(f'ffmpeg exited with {status}; see {logfile}')
        except BaseException:
            if encoder.stdin and not encoder.stdin.closed:
                try:
                    encoder.stdin.close()
                except OSError:
                    pass
            if encoder.poll() is None:
                encoder.terminate()
            encoder.wait(timeout=15)
            raise
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                            '-of', 'json', str(args.output)], capture_output=True, text=True, check=True)
    metadata = json.loads(probe.stdout)
    video = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in metadata['streams'] if s['codec_type'] == 'audio')
    if abs(float(metadata['format']['duration'])-duration) > .1 or int(video.get('nb_frames', frames)) != frames:
        raise RuntimeError('Encoded frame count or duration differs from the planned video.')
    first_visible = max(0, math.ceil((args.start-first)/period))
    beats = np.arange(first_visible, math.floor((args.start+duration-first)/period)+1)*period+first
    errors = np.abs(np.rint((beats-args.start)*args.fps)/args.fps-(beats-args.start))
    report = {'output': str(args.output), 'width': WIDTH, 'height': HEIGHT, 'fps': args.fps,
              'profile': args.profile,
              'frames': frames, 'duration_seconds': duration, 'music_start_seconds': args.start,
              'music_sha256': hashlib.sha256(args.music.read_bytes()).hexdigest(),
              'analysis': info, 'beat_phase_seconds': first,
              'max_nearest_frame_to_beat_error_ms': float(errors.max()*1000) if len(errors) else 0,
              'audio_codec': audio['codec_name'], 'video_codec': video['codec_name'],
              'render_wall_seconds': time.perf_counter()-began,
              'robots': [controller.report(max(0., args.start+duration-first)) for controller in controllers]}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Video complete: {args.output} ({args.output.stat().st_size/1024**2:.1f} MiB)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--music', type=Path, default=ROOT/'Billie Jean.mp3')
    parser.add_argument('--analysis', type=Path, default=ROOT/'sim_output/video/billie_jean_analysis.json')
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--profile', choices=['default', 'expressive'], default='default')
    parser.add_argument('--output', type=Path, default=ROOT/'sim_output/video/Rita_Billie_Jean_4_Robots.mp4')
    render(parser.parse_args())


if __name__ == '__main__':
    main()
