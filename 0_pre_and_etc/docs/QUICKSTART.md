# Quickstart

The Git root is the `experiment` directory in the original workspace. A fresh
`git clone https://github.com/Doyoung619/LSMA-Robot-Project.git` puts that content
directly inside `LSMA-Robot-Project/`. Commands below assume that cloned root.
When still at the parent Rita-Robot workspace, prefix paths with `experiment/`.

Use Python 3.11+ and ffmpeg on PATH. A virtual environment is recommended:
`python -m venv .venv`, then activate it using your shell's normal command.
Do not copy the old Windows virtual environment to another computer; create a
new environment and install the project. GPU, model weights and ROS are not
required for mock runs.

```powershell
python -m pip install -e .
python 0_pre_and_etc/run.py analyze
python 0_pre_and_etc/run.py run --mode finite --backend mock --seed 7
python 0_pre_and_etc/run.py run --mode infinite --backend mock --seed 7
```

Real playback uses the PortAudio DAC timestamp. Planning runs on a separate
worker. Playback speed remains unchanged. The executed demo track
`0_pre_and_etc/Billie Jean.mp3` is included in Git and discovered automatically
after clone/pull. `--audio` accepts an arbitrary local path for other music.
The original whole-track estimate is about 117.0255 BPM; this is measured, not a
song-name constant. `--bpm-override 117 --beat-offset .336` applies manual
correction; `--beat-offset` is the absolute first-beat music time, in seconds.
`tempo_diagnostic.png` shows half/double-tempo alternatives, not confidence.
General dense music currently uses a global tempo fit; sparse transient trains
also support varying intervals. No downbeat detector is claimed.

Headless, silent execution is deliberately explicit:

```powershell
python 0_pre_and_etc/run.py fixture --output 0_pre_and_etc/outputs/click.wav --bpm 240 --seconds 660
python 0_pre_and_etc/run.py analyze --audio 0_pre_and_etc/outputs/click.wav
python 0_pre_and_etc/run.py run --mode finite --backend mock --synthetic --headless --seconds 660 --seed 7
python 0_pre_and_etc/run.py run --mode infinite --backend mock --synthetic --headless --seconds 660 --seed 7
python 0_pre_and_etc/run.py benchmark --suite quick
python 0_pre_and_etc/run.py benchmark --suite long --seeds 7,19
python 0_pre_and_etc/run.py test
```

Run `--seconds` bounds the requested interval, completed at a certified phrase
boundary; it can run one phrase longer. The audio tail too short for a complete
phrase is reported rather than silently extrapolated. `--output DIR` chooses
the run destination; otherwise the program prints its timestamped `outputs/`
directory. `summary.json` contains config/provenance, continuous reference
checks, descriptor/rhythm/activity metrics and separately labeled feedback.
`reference.csv`, `summary.png`, and benchmark `comparison.json/csv/png` support
inspection. No synthetic or mock result is reported as physical measurement.

`--config FILE.json` overrides the mode's defaults; example configs are beside
each planner. `--ablation no-tempo-adaptation`, `no-coverage-novelty`, and
`single-rhythm` are supported. Failed constrained searches are reported rather
than forcing tiny motion. Optional failure injection: `--command-delay .025`,
`--tracking-delay .05`, `--feedback-dropout .03`, `--disconnect-at 12` (mock only).

Hardware setup is in [HARDWARE](HARDWARE.md). `doctor --backend misty --read-only`
and `doctor --backend kinova --read-only` inspect config/dependencies without
moving or contacting equipment. `--probe` makes a read-only network probe.
Actual motion/calibration requires `--profile name.local.json --execute`.
