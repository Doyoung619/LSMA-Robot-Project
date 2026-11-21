# Rita: finite and continuous robot dance

CPU-only, non-neural music analysis and two different motion planners. The default
backend is mock. Music is analyzed before playback; **infinite motion is generated
during playback**, with short reserved horizons. Neither system changes music speed.

```text
0_pre_and_etc/    shared audio, contracts, certification, adapters, runner, tests, docs;
                  preserved earlier simulation, robot demo and local artifacts
1_finite_version/ frozen 48-motion library (32–64), certified macro graph, fair service
2_infinite_version/ continuous candidate generation, novelty, receding horizon
3_BPM_check/      separate BPM-detector benchmark and checkpoint download setup
4_Benchmark1/     Reachy dance baseline vs existing planners, offline comparison
5_Benchmark2/     RoboPerform native G1 reproduction and verified asset setup
result/           local videos/results, retained and intentionally gitignored
```

After cloning this repository, run these four commands from its root (Python 3.11+
and ffmpeg required). The demo track `0_pre_and_etc/Billie Jean.mp3` is included
and discovered automatically. Use `--audio PATH` to select another local file.

```powershell
python -m pip install -e .
python 0_pre_and_etc/run.py analyze
python 0_pre_and_etc/run.py run --mode finite --backend mock --seed 7
python 0_pre_and_etc/run.py run --mode infinite --backend mock --seed 7
```

In the larger **Rita-Robot workspace**, this Git repository lives inside
`experiment/`: use `python -m pip install -e experiment` and prefix the commands'
`0_pre_and_etc/` paths with `experiment/`. No manual PYTHONPATH is needed.
Use `--headless` for accelerated silent mock runs. Without music:

```powershell
python 0_pre_and_etc/run.py run --mode infinite --backend mock --synthetic --headless --seconds 660
python 0_pre_and_etc/run.py benchmark --suite quick
python 0_pre_and_etc/run.py test
```

[Quickstart](0_pre_and_etc/docs/QUICKSTART.md) ·
[Algorithms](0_pre_and_etc/docs/ALGORITHMS.md) ·
[Theory and limits](0_pre_and_etc/docs/THEORY.md) ·
[Hardware setup](0_pre_and_etc/docs/HARDWARE.md) ·
[Experiments](0_pre_and_etc/docs/EXPERIMENTS.md) ·
[Migration](0_pre_and_etc/docs/MIGRATION.md)

Benchmark setup and download instructions: [3 — BPM](3_BPM_check/PORTABILITY.md),
[4 — dance generation](4_Benchmark1/README.md),
[5 — RoboPerform G1](5_Benchmark2/README.md). Each has its own environment; external
checkpoints are downloaded and hash-verified locally rather than committed.

Misty HTTP/WebSocket and Gen3 L53 6-DOF ROS2 trajectory adapters are implemented,
**hardware-unverified**. They require a reviewed local profile and explicit
`--execute`. They do not claim mesh collision avoidance or exact Misty spline
tracking. The earlier Kortex velocity demo remains preserved as legacy code;
it is not the new trajectory execution path. Hardware measurements are N/A in
mock reports. Certificates apply to reference curves under declared limits,
not physical tracking or emergency-stop performance.
