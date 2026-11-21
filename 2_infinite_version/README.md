# Continuous online planner

`rita_infinite.InfinitePlanner` generates new continuous joint targets while the
executor plays the already analyzed music. It never looks up a finite motion
library. A finite rhythm grammar supplies event times; continuous amplitudes,
directions, per-joint phases, harmonic mixtures and waypoint values are sampled
again on each call.

Use the shared entry point from the repository root:

```powershell
python experiment/0_pre_and_etc/run.py run --mode infinite --backend mock --audio "experiment/0_pre_and_etc/Billie Jean.mp3" --seed 7
```

The package is installed/discovered by the shared bootstrap. Numeric directory
names are never imported as Python module names. No robot SDK is imported here.

## Search and motion

The defaults in `config.json` are 24 proposals (supported range 16–32), a
two-phrase horizon (range 1–4), 16 recent geometry descriptors, novelty threshold
0.025 and a 1.5-second wall-clock budget. The planner tries metrical levels
0, 1, 2, 3 and prefers the fastest level with at least three certified choices.
Returning to a faster level requires two successful planning boundaries.
Music is never slowed. Each phrase spans `4 * 2**k` actual analyzed beats.

Shared quintic segments interpolate freshly generated waypoint positions.
Internal waypoint velocities are generally nonzero; acceleration is continuous.
The exact reserved start `q/dq/ddq` is preserved, and the horizon terminates at
neutral with zero velocity and acceleration. Only the first phrase is committed.
`stop_to_stop=true` forces zero velocity at phrase boundaries for hardware which
cannot stream C2 chunks; internal waypoint velocities remain nonzero. The default
mock configuration preserves flow across phrase boundaries. Near the end of the
analyzed beat grid the actual horizon can shrink to one complete phrase, and is
logged separately from the requested horizon.
For hardware compilation, `terminal_hold_seconds` (default 0) can reserve an
exact flat suffix, e.g. 0.20 seconds, only when `stop_to_stop=true`. The final
moving polynomial is compressed before that hold while the phrase end stays on
the original beat. The transformed controls, derivatives, activity, accent
timing and joins are revalidated; a hold which leaves no feasible candidate is
reported as search failure rather than silently violating the limits.
The remainder is retained as a validated suffix. No currently executing prefix
is replaced. Small amplitude projections use scales 1, 0.85 and 0.70; the final
geometry must still pass minimum-excursion and maximum-low-speed requirements.

Every candidate first passes conservative Bernstein position/velocity/
acceleration bounds. A small diverse finalist pool (up to eight by default) then
passes the shared full-segment/activity/join validator; unchecked proposals are
not called witnesses or counted feasible. This avoids expensive root solves on
every proposal.
Novelty, feature cell and terminal conditions are checked on the final geometry.
This certifies the reference under the profile's stated checks; it does not prove
physical tracking, self-collision avoidance or arbitrary workspace safety.

Rhythm accents are explicitly defined as designated waypoint passages.
Accents marked as rests are omitted while other joints continue moving. This
does not turn every beat into an all-joint stop. The shared evaluator must compare
these reference events with targets, and measured events separately if hardware
feedback exists.

## Diversity without impossible claims

The shared geometry descriptor separates shape from rhythm. Candidate distance
is the minimum distance to the recent 16 accepted normal trajectories. If no
candidate passes the requested threshold, it is relaxed by factors
`1, .75, .5, .25, 0`, with the chosen threshold and relaxation count logged.
A threshold of zero is explicitly a relaxation, not evidence of novelty.

Feature cells combine the dominant normalized joint-excursion group, first
quarter direction and low/high normalized excursion. There are eight cells for
multi-joint profiles, four for one joint. Cell membership is measured after
projection. A cell enters the persistent service queue only after a trajectory
from the current boundary has actually passed validation. Its witness records
that exact boundary and profile fingerprint. Old witnesses do not establish
current reachability. The oldest queue target persists until served; failure to
rediscover it neither removes it nor shrinks the coverage denominator.

If the persistent target has a current validated witness, it is selected before
other cells (with logged novelty relaxation if needed). If it has no witness now,
another valid trajectory can execute while the pending target remains. This is
conditional fair service when current witnesses keep being found. Random
candidate search supplies no finite-time reachability/coverage guarantee.
Statistics separately show all cells, ever-witnessed cells, visited cells and
unknown cells. Bounded workspace cannot support indefinitely positive distance
from every past trajectory; no such claim is made.

## Failure and terminal handling

Planning budget exhaustion or injected candidate failure uses only a retained
suffix whose start time and `q/dq/ddq` match the requested boundary and whose
constraints still validate. It is marked `generation_kind=fallback_suffix`,
excluded from novelty/coverage success, and consumed once. If no valid suffix
remains, `InfinitePlanningError(stop_required=True)` asks the executor to stop.
An untrusted/displaced feedback boundary is never snapped onto an old plan.
No candidate found means search failure, not proven mathematical impossibility.

At a requested session end, the executor calls `terminal_suffix(start_beat,
state)` to drain the already certified remainder without generating new motion.
These segments are tagged `terminal_suffix`, `recovery=true`, and
`excluded_from_diversity=true`. Draining can extend execution beyond the requested
cutoff until the retained terminal rest is reached; it records no search failure,
timeout, novelty success or normal coverage success. An already resting boundary
with no suffix returns `None`; a moving boundary without a valid suffix raises an
explicit stop-required error.

Ablations use `metrical_levels=[0]` (no tempo adaptation),
`no_coverage_novelty=true` (no service-target bias or novelty score/threshold), and
`single_rhythm=true` (only the first shared rhythm). Distances and observed cell
counts remain measurable in the diversity ablation; they are not policy claims.

The terminal neutral stop is a maintained witness for the existing plan only.
Calling this receding-horizon planning does not establish unrestricted recursive
feasibility. Feedback error, profile changes, current boundary constraints or a
deadline can still invalidate continuation. Full future information used by a
search is the logged `future_audio_lookahead_seconds` of the chosen horizon;
analysis itself may use the complete offline music file.

## Executed validation

`0_pre_and_etc/tests_v2/test_infinite.py` contains behavioral checks for genuinely
new rounded coefficients, C2 joins, nonzero boundary flow, hardware stop-boundary
compilation, horizons 1/4, exact suffix consumption, actual budget timeout,
feedback mismatch refusal, novelty relaxation and witness queue persistence.

`validate_online.py` runs accelerated virtual-time reference checks without any
hardware or audio calls. For seed 7, a 120-BPM/1.2-rad/s/6-rad/s² run completed
612 seconds in 149 phrases. At a deliberately strict 0.75-second budget two
deadline misses consumed certified suffixes; no invalid join or reference was
executed. All eight cells and thirteen rhythms were visited. Planning median/p95
were 0.219/0.453 seconds; maximum joined low-speed duration was 0.0507 seconds.

At 180 BPM with limits 0.30 rad/s and 1.2 rad/s², a 1.5-second budget completed
618.67 virtual seconds in 58 phrases, with no failures. Planning median/p95 were
0.313/0.524 seconds, maximum joined low-speed duration 0.2929 seconds, minimum
excursion 0.1229 rad. The same tight profile under concurrent CPU load and a
0.75-second budget correctly stopped after consecutive deadline misses exhausted
the retained suffix. This motivates the default 1.5-second budget; it is not a
hard-real-time guarantee. These are mock/reference results, not robot tracking.

```powershell
python experiment/2_infinite_version/validate_online.py --seconds 610 --budget 1.5
python experiment/2_infinite_version/validate_online.py --seconds 610 --bpm 180 --speed .30 --acceleration 1.2 --budget 1.5
```
