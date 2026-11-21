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
duration, coverage and repetition metrics. User audio, decoded caches, trajectories
and full logs are not committed. Small validation evidence lives in `evidence/`
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
