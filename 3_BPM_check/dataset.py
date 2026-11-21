"""Deterministic synthetic, GTZAN-mini, and existing project audio excerpts.

The GTZAN URLs and MD5s come from mirdata's published mini index.  Only
selected tracks are downloaded, rather than the entire audio archive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import urllib.request

import numpy as np
from scipy import signal
import soundfile as sf

INDEX_URL = "https://zenodo.org/records/14004436/files/gtzan_genre_1.0_mini_index.json?download=1"
INDEX_MD5 = "ac97f5a783d7843cf92ed8875d85af3d"
MIRDATA_SOURCE = "https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/gtzan_genre.py"
AUDIO_REPO = "https://raw.githubusercontent.com/TempoBeatDownbeat/gtzan_mini/"
ANNOTATION_REPO = "https://raw.githubusercontent.com/TempoBeatDownbeat/gtzan_tempo_beat/"
AUDIO_REVISION = "a61439e86c13037011fde8e0f0743ec55c50bce3"
ANNOTATION_REVISION = "fd9cafdebb6426fee41c3718791be61306323dd7"
SAMPLE_RATE = 44100


def _hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url, path, md5=None):
    path = Path(path)
    if path.exists() and (md5 is None or _hash(path, "md5") == md5):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Rita-BPM-Benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as target:
        while chunk := response.read(1024 * 1024):
            target.write(chunk)
    if md5 is not None and _hash(temporary, "md5") != md5:
        raise ValueError(f"MD5 mismatch for {url}")
    temporary.replace(path)
    return path


def _stereo_44100(audio, rate):
    if audio.ndim == 1:
        audio = audio[:, None]
    if rate != SAMPLE_RATE:
        factor = math.gcd(int(rate), SAMPLE_RATE)
        audio = signal.resample_poly(audio, SAMPLE_RATE // factor, int(rate) // factor, axis=0)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = np.repeat(audio.mean(axis=1, keepdims=True), 2, axis=1)
    return np.asarray(audio, dtype=np.float32)


def _write_crops(root, track_id, tier, audio, beats, bpm, durations, metadata):
    records = []
    for requested in durations:
        excerpt = audio[:round(float(requested) * SAMPLE_RATE)]
        actual = len(excerpt) / SAMPLE_RATE
        clip_id = f"{track_id}_{requested:g}s"
        relative = Path("data") / "clips" / tier / (clip_id + ".wav")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sf.info(target).frames != len(excerpt):
            sf.write(target, excerpt, SAMPLE_RATE, subtype="PCM_16")
        local_beats = None if beats is None else np.asarray(beats)[
            (np.asarray(beats) >= 0) & (np.asarray(beats) < actual)
        ]
        intervals = [] if local_beats is None else np.diff(local_beats)
        records.append({
            "clip_id": clip_id,
            "tier": tier,
            "track_id": track_id,
            "audio_path": relative.as_posix(),
            "duration_seconds": actual,
            "requested_duration_seconds": float(requested),
            "start_seconds": 0.0,
            "reference_bpm": bpm,
            "reference_beats": None if local_beats is None else local_beats.tolist(),
            "local_annotation_median_bpm": float(60 / np.median(intervals)) if len(intervals) else None,
            "sample_rate": SAMPLE_RATE,
            "channels": 2,
            "pcm_sha256": _hash(target),
            **metadata,
        })
    return records


def _synthetic(bpm, pattern, seconds, seed):
    """Quarter-note beats are the reference; eighth-note hats are subdivisions."""
    rng = np.random.default_rng(seed)
    audio = np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float32)
    beats = np.arange(0.5, seconds, 60.0 / bpm)

    def add(t0, waveform, gain=1.0):
        first = round(t0 * SAMPLE_RATE)
        count = min(len(waveform), len(audio) - first)
        if first >= 0 and count > 0:
            audio[first:first + count] += gain * waveform[:count]

    click_t = np.arange(round(.035 * SAMPLE_RATE)) / SAMPLE_RATE
    click = np.sin(2 * np.pi * 1400 * click_t) * np.exp(-click_t / .006)
    for i, beat in enumerate(beats):
        if pattern == "metronome":
            add(beat, click, .8 if i % 4 == 0 else .6)
            continue
        # A deterministic drum/bass phrase with syncopated eighth notes.
        t = np.arange(round(.22 * SAMPLE_RATE)) / SAMPLE_RATE
        kick = np.sin(2 * np.pi * (48 * t + 5 * (1 - np.exp(-30 * t)))) * np.exp(-t / .065)
        if i % 4 in (0, 2):
            add(beat, kick, .65)
        if i % 4 in (1, 3):
            snare = rng.normal(0, 1, len(t)) * np.exp(-t / .035)
            snare = signal.lfilter([1, -.95], [1], snare)
            add(beat, snare, .15)
        for offset in (0, .5):
            ht = np.arange(round(.055 * SAMPLE_RATE)) / SAMPLE_RATE
            hat = rng.normal(0, 1, len(ht)) * np.exp(-ht / .012)
            hat = signal.lfilter([1, -.95], [1], hat)
            add(beat + offset * 60 / bpm, hat, .055)
        nt = np.arange(round(min(.36, .7 * 60 / bpm) * SAMPLE_RATE)) / SAMPLE_RATE
        frequency = [65.406, 77.782, 87.307, 58.270][(i // 4) % 4]
        bass = (np.sin(2 * np.pi * frequency * nt) + .2 * np.sin(4 * np.pi * frequency * nt))
        bass *= (1 - np.exp(-nt / .004)) * np.exp(-nt / .16)
        add(beat, bass, .28)
        if i % 4 == 2:
            add(beat + .75 * 60 / bpm, click, .16)
    audio /= max(1.0, float(np.max(np.abs(audio))) / .9)
    return np.repeat(audio[:, None], 2, axis=1), beats


def _prepare_real(root, durations, seed, real_count):
    cache = root / "data" / "gtzan_mini"
    index_file = _fetch(INDEX_URL, cache / "mirdata_mini_index.json", INDEX_MD5)
    tracks = json.loads(index_file.read_text(encoding="utf-8"))["tracks"]
    provenance_file = cache / "provenance.json"
    if provenance_file.exists():
        revisions = json.loads(provenance_file.read_text(encoding="utf-8"))
    else:
        revisions = {"audio_revision": AUDIO_REVISION,
                     "annotation_revision": ANNOTATION_REVISION}
        provenance_file.write_text(json.dumps(revisions, indent=2), encoding="utf-8")
    # Round-robin across shuffled genre lists: two per genre for 20 tracks.
    rng = random.Random(seed)
    genres = sorted({key.split(".")[0] for key in tracks})
    lists = []
    for genre in genres:
        ids = sorted(key for key in tracks if key.startswith(genre + "."))
        rng.shuffle(ids)
        lists.append(ids)
    selected = [xs[index] for index in range(max(map(len, lists))) for xs in lists if index < len(xs)]
    selected = selected[:min(real_count, len(selected))]

    def prepare(track_id):
        item = tracks[track_id]
        source_paths = {}
        source_urls = {}
        for kind, (path, checksum) in item.items():
            repository = AUDIO_REPO if kind == "audio" else ANNOTATION_REPO
            revision = revisions["audio_revision" if kind == "audio" else "annotation_revision"]
            relative = path.split("/", 1)[1]
            url = repository + revision + "/" + relative
            local = cache / kind / Path(relative).name
            source_paths[kind] = _fetch(url, local, checksum)
            source_urls[kind] = url
        audio, rate = sf.read(source_paths["audio"], dtype="float32", always_2d=True)
        audio = _stereo_44100(audio, rate)
        beats = np.loadtxt(source_paths["beats"], ndmin=2)[:, 0]
        bpm = float(np.loadtxt(source_paths["tempo"]))
        if not np.isfinite(bpm) or bpm <= 0 or len(beats) < 2:
            raise ValueError(f"Invalid published annotations for {track_id}")
        return _write_crops(root, track_id, "real", audio, beats, bpm, durations, {
            "dataset": "GTZAN-mini",
            "genre": track_id.split(".")[0],
            "source_url": source_urls["audio"],
            "annotation_source": source_urls["tempo"],
            "beat_annotation_source": source_urls["beats"],
            "reference_definition": "Published global tempo annotation; unchanged for every prefix duration",
            "license": "GTZAN audio license unspecified by mirdata; local evaluation only; audio excluded from Git",
            "source_audio_md5": item["audio"][1],
            "source_audio_sha256": _hash(source_paths["audio"]),
        })

    # Downloads are I/O-bound; this function never performs model inference.
    records = []
    errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(track, pool.submit(prepare, track)) for track in selected]
        for track, future in futures:
            try:
                records.extend(future.result())
                print(f"Prepared GTZAN {track}", flush=True)
            except Exception as exc:
                errors.append({"track_id": track, "error": f"{type(exc).__name__}: {exc}"})
    selection = {
        "dataset": "GTZAN-mini", "seed": seed, "requested_tracks": real_count,
        "selected_track_ids": selected, "successful_track_count": len({r["track_id"] for r in records}),
        "selection": "Seeded shuffle within each genre, then round-robin across ten genres",
        "index_source": INDEX_URL, "index_md5": INDEX_MD5, "mirdata_loader_source": MIRDATA_SOURCE,
        **revisions, "download_errors": errors,
    }
    (root / "data" / "dataset_provenance.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return records


def prepare_dataset(root, durations=(8, 15, 30), seed=2026, real_count=20, audio_path=None):
    root = Path(root).resolve()
    (root / "data").mkdir(parents=True, exist_ok=True)
    durations = tuple(float(x) for x in durations)
    if not durations or min(durations) <= 0:
        raise ValueError("Durations must be positive")
    records = []
    for bpm in (60, 90, 117, 120, 150, 180):
        for pattern in ("metronome", "musical"):
            audio, beats = _synthetic(bpm, pattern, max(durations), seed + bpm)
            records.extend(_write_crops(root, f"synthetic_{pattern}_{bpm}", "synthetic", audio,
                beats, float(bpm), durations, {
                    "dataset": "generated", "pattern": pattern,
                    "source_url": None, "annotation_source": "Analytic quarter-note onset schedule from generator",
                    "reference_definition": "Exact configured quarter-note BPM; eighth notes are subdivisions",
                    "license": "Generated by benchmark code", "generator_seed": seed + bpm,
                }))
    if real_count > 0:
        records.extend(_prepare_real(root, durations, seed, real_count))
    song = Path(audio_path).resolve() if audio_path else root.parent / "0_pre_and_etc" / "Billie Jean.mp3"
    if song.exists():
        decoded = root / "data" / "billie_jean_decoded.wav"
        if not decoded.exists() or decoded.stat().st_mtime < song.stat().st_mtime:
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(song),
                            "-ar", str(SAMPLE_RATE), "-ac", "2", str(decoded)], check=True)
        audio, rate = sf.read(decoded, dtype="float32", always_2d=True)
        records.extend(_write_crops(root, "billie_jean" if audio_path is None else song.stem,
            "demo", _stereo_44100(audio, rate), None, None, durations, {
                "dataset": "existing project audio", "source_url": str(song),
                "annotation_source": None, "reference_definition": "No verified ground truth; qualitative demo only",
                "license": "Existing local file; excluded from Git", "source_audio_sha256": _hash(song),
            }))
    (root / "data" / "dataset_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--durations", type=float, nargs="+", default=[8, 15, 30])
    parser.add_argument("--audio-path", type=Path)
    args = parser.parse_args()
    result = prepare_dataset(Path(__file__).resolve().parent, args.durations, args.seed, args.real_count, args.audio_path)
    print(json.dumps({"clips": len(result), "tracks": len({r['track_id'] for r in result}),
                      "real_tracks": len({r['track_id'] for r in result if r['tier'] == 'real'})}, indent=2))
