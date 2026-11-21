# Implemented guarantees and limits

Notation follows ALGORITHMS.md: b_i are audio beats, x=(t,q,dq,ddq) is a boundary,
k is metrical level, n counts visit goals, D is directed diameter, and tau_max
is maximum action duration of a fixed selected certified macro SCC.

## Continuous reference certificate

Every piece has six Bezier controls B_0..B_5 and duration T. Position lies in
their convex hull. Velocity has controls 5(B_(i+1)-B_i)/T; acceleration has
controls 20(B_(i+2)-2B_(i+1)+B_i)/T². Componentwise control limits are sufficient
for the entire reference interval. They are conservative: a rejected hull is
not proof that no other interpolation is feasible. Equal q/dq/ddq at all knots
and dispatched boundaries give C2 reference continuity. Nonlinear time warping
is not used. Timing changes rebuild derivative controls and are revalidated.

Low-speed time is computed by numerical real roots of quartic dq_j=±threshold,
then intersecting all joints' low-speed intervals, including across pieces.
Adjacent segment tail/head durations are added. Isolated zero-velocity direction
reversals are allowed. Waypoint excursion is a certified lower bound on path
excursion. Numerical roots have floating tolerances; these are not formal
rounding-error proofs. Collision-free motion, dynamic balance, unknown
obstacles and physical tracking are not established by joint bounds. Profiles
may require unavailable collision checks, in which case validation fails.

## Finite service bound

The certificate concerns the explicit checked macro graph, not an unknown dense
motion-ID graph. Node (p,r,k) represents executing that entire timed action;
an edge means the destination can follow it with certified boundaries/activity.
Edge time is bounded by destination execution duration tau_max. For a fixed
recurrent SCC, every target motion/rhythm has at least one node. BFS chooses a
minimum-edge route and the route is retained. A revisit to a current target is
a positive-length execution, not a zero-time self-visit.

Any target can therefore be served in at most D+1 executed actions: reach a
target in ≤D steps, with one extra action conservatively accounting for current
target re-execution/initial service. A bag with n goals finishes within
B=n(D+1)tau_max. A goal can be first in one bag and last in the next, so its
revisit gap is ≤2B. One action can satisfy two goals; that only tightens the bound.
Checked lazy shortcut edges cannot increase the bound from the original sparse
SCC. Tests explicitly check bag/revisit inequalities and disconnected graphs.

Conditions are fixed beat intervals/profile/latency, successful execution,
retained service paths and no interruption. Timing/profile/latency fingerprint
change resets graph and coverage certificate at a block boundary. Previous
coverage is not transferred as a guarantee. Inaccessible IDs outside the SCC
remain in the full-library denominator. No claim forbids every finite pattern
from ever recurring in an infinite execution.

## Infinite novelty, service and suffix

An accepted normal trajectory has a measured descriptor distance from the
recent window at its logged effective threshold. Threshold zero is explicitly
not positive novelty. Bounded feature space cannot support a permanently
positive distance from all past trajectories. Shared feature coverage and
fixed-window novelty are different metrics.

A feature cell is currently reachable only with a validated witness from the
current reserved boundary/profile. An old witness is not current proof.
Persistent oldest-target service is conditional on finding current witnesses;
random candidate search supplies no finite-time discovery bound. Unknown cells
are not silently removed. Candidate failure and narrow-box proven excursion
impossibility are distinct statuses.

Each accepted horizon ends at a validated rest state. Only its first phrase
commits. A planning failure can consume the untouched suffix once if its time,
q/dq/ddq and profile still match; otherwise the executor requests stop. Large
feedback errors never cause a snap onto planned state. This is a finite retained
witness, not unrestricted recursive feasibility for arbitrary future states,
constraint changes, solver failures or changing graphs.

## Clocks, execution and measurement

PortAudio outputBufferDacTime establishes the first audible sample epoch.
Current stream time minus that epoch gives audio time, separately mapped to
monotonic time with a measured mapping uncertainty. Callback invocation time,
queued samples and accumulated sleep are not the playback clock. Planning and
network dispatch never occur in the callback. The mock virtual clock is labeled
as accelerated simulation. Wall planning latency is measured independently.

Scheduled controller time is the execution target; dispatch lead is not a
second latency subtraction. No expired-command bursts are sent. Reference and
feedback event errors are reported separately, with missing feedback/unsafe
tracking leading to stop. Physical measurements are N/A for mock runs. Feedback
derivatives are filtered numerical estimates with period/filter recorded, not
continuous physical certificates. Software stop does not replace a hardware
emergency stop. Misty pose interpolation is not assumed equal to the spline.

Official references: [PortAudio stream timing](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html),
[non-neural DP beat tracking reference](https://librosa.org/doc/latest/_modules/librosa/beat.html),
and the vendor documents linked in HARDWARE.md. No neural detector or downloaded
model is required; librosa is not a mandatory dependency.
