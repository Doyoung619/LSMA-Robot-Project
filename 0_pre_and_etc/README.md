# Shared pipeline and preserved experiments

Run `python run.py --help`, or see [QUICKSTART](docs/QUICKSTART.md).

`rita_core/` owns explicit rad/second contracts, spectral-flux analysis, DAC
clock, rhythm grammar, continuous Bezier certification, mock/Misty/Gen3 adapters,
execution and common evaluation. The finite and infinite algorithm packages live
in their own numbered sibling directories. `tests_v2/` is the new contract suite.

Earlier `rita_robot/`, `rita_sim/`, launchers, audio, models and local environments
were moved intact here from `etc/`. [LEGACY_README](LEGACY_README.md) and
[SIMULATION](SIMULATION.md) describe that earlier work. Those claims and earlier
physical logs are not evidence that the new adapters have been run on hardware.
The legacy `requirements.txt` includes a vendor SDK and is not needed for the new
mock workflow; install the repository's `pyproject.toml` instead.

All generated runs go under `outputs/`, including summary JSON, reference CSV,
figures and decoded audio cache. These and local environments, user music,
hardware settings and earlier large results are gitignored.
