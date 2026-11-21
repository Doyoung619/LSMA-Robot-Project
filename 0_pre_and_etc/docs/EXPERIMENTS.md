# Reproducible experiments

All comparisons use the same audio grid, profile, seed and requested duration.
`benchmark --suite quick` compares a fixed feasible loop, feasible random
selection, finite service and continuous infinite, plus both planners' three
ablations: no tempo adaptation, no coverage/novelty, single rhythm. A search
failure stays a failed row rather than an invalid motion. Baselines are mock or
offline measurements; they have no fair-service guarantee.

From the cloned repository root:

```powershell
python 0_pre_and_etc/run.py test
python 0_pre_and_etc/run.py benchmark --suite quick --seeds 7
python 0_pre_and_etc/run.py benchmark --suite long --seeds 7,19
python 0_pre_and_etc/run.py benchmark --suite quick --audio "C:/music/Billie Jean.mp3" --seconds 250
```

Quick defaults to60 seconds; long to660 virtual seconds. Every planning call's
actual CPU wall latency is measured, even with accelerated audio time. Long
separates normal, tightened limits, delayed feedback, missing feedback and
disconnect conditions. Failure injection intentionally stops runs; such rows
are not classified as completed ten-minute demonstrations. Outputs include
comparison JSON/CSV/PNG and per-run q/dq/ddq plots, pose-event sync, low-speed
duration, coverage and repetition metrics. The executed demo MP3 is included in Git;
additional user audio, decoded caches, trajectories and full logs remain local.
Small validation evidence lives in `evidence/`
and the mode-specific validation summaries.

Metrics: reference accents are exact designated pose passages at rhythm targets;
feedback uses nearest local joint-pose passage within a temporal window, counts
misses and respects subdivisions/offbeats. Extra-event detection is N/A without
an independent full event detector. Reference control-hull bounds and C2 joins
are continuous-time checks, separate from filtered sampled feedback estimates.
The shared geometry descriptor excludes raw hashes and separates rhythm.
Finite total/feasible-tested/SCC coverage denominators are all preserved; shared
geometry cells and recent-window distance are comparable across methods.
Short n-gram repetition and observed revisits use the shared geometry-cell
sequence for both modes, rather than unique infinite IDs; finite ID coverage
remains separately reported. A stationary feedback stream near an accent is
not an event: a localized pose-distance minimum with observable approach/exit
and at least 0.002 rad local excursion is required. Boundary/weak passages which
cannot be resolved by this detector count as missed, not perfect synchronization.
Runtime records planning median/p95/max, misses, fallback, novelty relaxation,
k changes and certificate resets. Config/profile/backend/audio hash/seed and
Git SHA/dirty status accompany every run. Mock feedback is simulated, while
physical robot metrics are N/A. Automatic Billie Jean beat alignment has no
ground-truth music beat annotations and is not labeled beat-detection accuracy.

The original local Billie Jean file was found and reanalyzed in this work:
SHA256 `918dfeba11ed298af7e3f84a43527d0627b3e085766697bed758c851553ff174`,
294.011066 seconds, 573 estimated beats, first beat0.335926425s,
117.025464822 BPM. The original detector retains its documented global tempo
and octave limitations; diagnostic plots expose58.51/117.03/234.05 alternatives.
Synthetic signal-based transient tests cover60/120/180/240 BPM, known phase,
tempo change, silence and insufficient-duration failure. Fixture truth JSON is
used by tests only, never by the analyzer.

Detailed measured results for the final code are recorded with their tested
scope in the committed evidence. Earlier legacy hardware logs are not new
hardware validation. Both current hardware adapters are **implemented,
hardware-unverified**; no actual robot was moved in this implementation.

## Final executed validation — 2026-11-21

The compact reports and comparison figures in [evidence/](evidence/README.md)
preserve the tested commit, dirty flag, audio hash, profile and configuration
of each source run. Actual planning wall times may include concurrent CPU
load; these measurements do not establish hard-real-time deadlines.

Final sampled derivatives use the v3 estimator: near-duplicate feedback
timestamps are coalesced (threshold `max(command_period * .001, 1e-7 s)`,
which is 25 µs for the default 0.025-second command period),
samples are resampled uniformly at the median actual interval, and
Savitzky–Golay window 9/poly 3 derivatives use physical time delta. Raw
feedback timestamps are saved separately in ignored `feedback.csv` outputs.
The older v2 derivative estimates were numerically invalid at near-duplicate
join timestamps and are not used as final evidence. Continuous reference
certificates, geometry, coverage and planning times were unaffected by that
measurement bug. The final actual-DAC demonstration was also rerun with
the corrected evaluator and retains its own tested SHA.

Actual Billie Jean runs requested 294.011 seconds using the same seed 7
and analyzed grid. Complete feasible phrases end before the remaining tail:

| Mode | Planned seconds | Normal segments | Unplanned tail (s) | Geometry cells | Rhythms | Planning p95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| finite | 287.117 | 35 | 6.558 | 4/8 | 12 | 0.0370 |
| infinite | 291.219 | 71 | 2.456 | 8/8 | 13 | 0.3510 |

These are mock-robot runs, not physical tracking results. The unplanned tail
is recorded explicitly; completion does not claim a dance for every final
audio sample. Missed localized motion passages and N/A extra-event detection
remain in the evidence rather than being reported as perfect synchronization.

Quick suite: 10 runs, 9 completed. `infinite/no-tempo-adaptation` stopped at 32.000 seconds with `InfinitePlanningError: Infinite planner: candidate_search_failed; adapter stop required`. The comparison plot marks stopped rows in red and labels their actual
duration. Baseline/proposed rows share the requested grid/profile/seed;
stopped ablations are not silently compared as complete runs.

Long suite, seeds 7/19: normal and tightened-limit cases each ran at least
660 virtual seconds; delayed-feedback cases are separate short conditions.
Overall 12 of 20 conditions completed and 8 injected disconnect/missing-feedback conditions intentionally stopped.

| Condition | Seed | Mode | Reference seconds | Geometry cells | Rhythms | Planning p95 (s) | Max low speed (s) |
|---|---:|---|---:|---:|---:|---:|---:|
| normal | 7 | finite | 664.000 | 4/8 | 13 | 0.0251 | 0.3428 |
| normal | 7 | infinite | 664.000 | 8/8 | 13 | 0.3574 | 0.0510 |
| tightened | 7 | finite | 664.000 | 4/8 | 6 | 0.0129 | 0.3475 |
| tightened | 7 | infinite | 672.000 | 8/8 | 13 | 0.3863 | 0.1600 |
| normal | 19 | finite | 664.000 | 4/8 | 13 | 0.0254 | 0.3428 |
| normal | 19 | infinite | 664.000 | 8/8 | 13 | 0.3526 | 0.0555 |
| tightened | 19 | finite | 664.000 | 4/8 | 6 | 0.0132 | 0.3480 |
| tightened | 19 | infinite | 672.000 | 8/8 | 13 | 0.3970 | 0.2830 |

For both seeds, normal finite execution visited **48/48 stored motions** and
all 13 rhythms. Under tightened limits it visited **15/48 motions (31.25%)**:
26 motions were found feasible, with 15 in the selected SCC. Visiting all
15 SCC motions is not reported as full-library coverage. Shared geometry
cells in the table are a separate coarse descriptor measure.

The full compact JSON retains reference violations, missed/extra-event status,
sampled estimator scope and all expected failure reasons for every condition.
A separate actual PortAudio/DAC playback demonstration covered 16.407
reference seconds with online planning and a mock robot. It is evidence of
music-output clock integration, not a physical robot experiment.

Independent clean-clone verification: **54 tests passed in 43.110 seconds** at `14766406b946dd1db47ab7e3e5942e211dc2654e`. The installed `rita-dance doctor --backend mock --read-only` entry point returned ready at the earlier `e0f6832` commit; its bootstrap was unchanged in the final commit. The new venv shared the system numerical packages and installed the declared
soundfile/sounddevice requirements; robot SDKs were absent. The unchanged
legacy robot and simulation suites separately passed 10 and 13 tests.
