# Algorithms

Notation: t is audio time in seconds; b_i are analyzed beat timestamps; q, v, a
are ordered joint position, velocity and acceleration in rad, rad/s, rad/s².
k∈{0,1,2,3} is metrical level; a phrase covers 4·2^k source beats. p is a finite
motion ID, r is one of thirteen shared four-beat rhythms, and x=(t,q,v,a) is a
reserved boundary. An accent is a designated joint-pose passage. Beat-origin
four-beat blocks are not detected musical downbeats.

`audio analysis → DAC clock → rhythm/tempo planning → finite/infinite → final
reference validation → adapter → feedback/logging` is shared. AudioAnalysis
records source SHA256, original sample rate, explicit beat times, onset, energy,
accent and analyzer settings. RobotProfile fixes model, joint order, units,
limits, workspace, command period, feedback freshness and latency policy.
DanceSegment contains absolute knots, quintic Bezier coefficients, q/dq/ddq
evaluation, rhythm/accent targets and validation. PlannerState separates current
and reserved state, queue, history and failures. Unspecified joint order or
degree/radian conversion is rejected; only adapters convert vendor units.

For dense music the existing whitened STFT spectral flux analyzer is reused:
11.025 kHz, FFT1024, hop128; 35–180/180–2000/2000–5000 Hz with weights
.55/.30/.15. Positive changes, local median subtraction, repeated-lag onset
autocorrelation and robust global beat-grid regression produce explicit times.
It retains half/double-tempo ambiguity and a global-tempo assumption. A generic
signal-based sparse-transient branch tracks isolated attacks directly (including
tempo-changing clicks), without loading fixture truth labels. Silence and short
inputs fail explicitly. Full-audio analysis is offline; it does not claim to
listen to unknown music in real time. Manual BPM/origin correction is available.

Finite freezes mapped geometry before playback. Its shipped 64-path pool selects
48 by default (32–64 configurable), checks meaningful projected duplicates, and
admits only rhythms with matching stored event slots. Stored chord midpoints
let accent strength change approach timing without generating new waypoints.
A macro-node is (p,r,k). Full reference and concatenated low-speed/C2 checks
certify only checked self/hub/lazy shortcut edges. Unknown edges are distinct
from rejected certificates. The largest enterable recurrent certified SCC is
chosen; total, feasible-tested and SCC motion counts have separate denominators.
The fastest k retaining configured diversity is preferred, with hysteresis on
return to faster motion. Target motion and rhythm IDs enter a shuffle bag. A
minimum-edge certified service route persists until its target is reached.
Accent match, n-gram history and smoothness only break equal-depth route ties.

Infinite never loads that motion library. Each online call samples 24 continuous
amplitude, phase, direction and harmonic-mixture proposals (16–32 supported),
then constructs horizons of 1–4 phrases (default2). It preserves reserved x,
usually nonzero internal and phrase-boundary velocity, and terminates the horizon
at neutral rest. Cheap control-hull filters precede full root/activity checks on
a small diverse finalist pool. Small amplitude projections are revalidated.
Only the first phrase commits; the remaining validated suffix is retained.
Oldest current-witness feature-cell targets persist. Final geometry must pass
recent-K16 novelty at the logged threshold, relaxed by 1,.75,.5,.25,0 if needed.
Candidate failure never proves a cell impossible or removes the all-cell
denominator. Timeout uses an exact matching validated suffix, otherwise stop.

Both modes use geometry descriptors of time-normalized path, joint excursion,
reversals and joint correlation, separately from event interval/strength rhythm
descriptors. Float jitter and raw hashes are not novelty. Whole-robot low-speed
duration and minimum excursion prevent arbitrarily slow or tiny success.
Hardware stop-to-stop mode can reserve a certified constant terminal hold;
the complete transformed reference, including combined tail/head time, is
rechecked before a no-queue controller can omit that hold from its action.

Baselines are a fixed certified macro loop and feasible random selection using
the same constraints. Ablations disable tempo adaptation, coverage/novelty, or
all but one rhythm. They are evaluated offline/mock; failures remain visible.
