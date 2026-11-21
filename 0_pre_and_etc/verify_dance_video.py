"""Check encoded streams, source-audio alignment, and visible motion per panel."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy import signal
from PIL import Image, ImageDraw

from rita_sim.assets import ROOT, ROBOT_IDS


def pcm(path, start, duration):
    result = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                             '-ss', str(start), '-i', str(path), '-t', str(duration),
                             '-vn', '-ac', '1', '-ar', '8000', '-f', 'f32le', 'pipe:1'],
                            capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype='<f4').astype(float)


def picture(path, t):
    result = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                             '-ss', str(t), '-i', str(path), '-frames:v', '1',
                             '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                            capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(1080, 1920, 3)


def verify(path, music):
    report_path = path.with_suffix('.json')
    if not report_path.exists():
        report_path = ROOT / 'sim_output/video' / report_path.name
    report = json.loads(report_path.read_text(encoding='utf-8'))
    metadata = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_streams',
                                          '-show_format', '-of', 'json', str(path)],
                                         capture_output=True, text=True, check=True).stdout)
    video = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in metadata['streams'] if s['codec_type'] == 'audio')
    assert video['width'] == 1920 and video['height'] == 1080
    assert video['pix_fmt'] == 'yuv420p'
    assert int(video['nb_frames']) == report['frames']
    assert abs(float(video['start_time'])-float(audio['start_time'])) < 1/report['fps']
    assert report['max_nearest_frame_to_beat_error_ms'] <= 500/report['fps']+1e-6
    duration = float(metadata['format']['duration'])
    check_length = min(8., duration-1.)
    reference = pcm(music, report['music_start_seconds'], check_length)
    encoded = pcm(path, 0., check_length)
    n = min(len(reference), len(encoded))
    reference, encoded = reference[:n], encoded[:n]
    correlations = signal.correlate(encoded, reference, mode='full', method='fft')
    lags = signal.correlation_lags(len(encoded), len(reference))
    nearby = np.abs(lags) <= 800
    lag = int(lags[nearby][np.argmax(correlations[nearby])])
    if lag > 0:
        a, b = encoded[lag:], reference[:-lag]
    elif lag < 0:
        a, b = encoded[:lag], reference[-lag:]
    else:
        a, b = encoded, reference
    similarity = float(np.dot(a, b)/np.sqrt(np.dot(a, a)*np.dot(b, b)))
    assert abs(lag) <= 8, f'Encoded audio offset exceeds 1 ms: {lag/8:g} ms'
    assert similarity > .98, f'Audio does not closely match its source: {similarity}'
    times = sorted(set([0., min(3., duration*.2), min(10., duration*.5), duration*.8]))
    pictures = [picture(path, t) for t in times]
    motion = {}
    for robot, (x, y) in zip(ROBOT_IDS, [(44, 164), (982, 164), (44, 610), (982, 610)]):
        crops = [p[y:y+330, x:x+700].astype(float) for p in pictures]
        changes = [float(np.mean(np.abs(crops[0]-crop))) for crop in crops[1:]]
        motion[robot] = max(changes)
        assert motion[robot] > .05, f'{robot}: no visible motion in robot viewport'
    contact = Image.new('RGB', (1920, 1080))
    for i, pixels in enumerate(pictures[:4]):
        contact.paste(Image.fromarray(pixels).resize((960, 540), Image.Resampling.LANCZOS),
                      ((i%2)*960, (i//2)*540))
    contact.save(report_path.with_name(path.stem+'_qa_contact.jpg'), quality=92)
    result = {'passed': True, 'duration_seconds': duration, 'frames': int(video['nb_frames']),
              'audio_offset_ms': lag/8, 'source_audio_correlation': similarity,
              'robot_viewport_mean_pixel_change': motion,
              'max_nearest_frame_to_beat_error_ms': report['max_nearest_frame_to_beat_error_ms'],
              'sample_times': times}
    report_path.with_name(path.stem+'_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--music', type=Path, default=ROOT/'Billie Jean.mp3')
    args = parser.parse_args()
    verify(args.video, args.music)
