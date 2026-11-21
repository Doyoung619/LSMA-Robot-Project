# Finite: frozen geometry and certified service paths

`rita_finite/library.json` contains 64 authored normalized joint-space paths.
The default configuration selects the first 48; `library_size` supports 32–64.
Every path's ordered waypoints, neutral C2 boundary, accent slots and compatible
rhythms are stored before playback. `build_library.py` is an **offline authoring
tool** and is never called by the planner. The library hash is verified on load.

At construction the paths are mapped to the profile's ordered joints using one
fixed neutral/range scale. A shared geometry descriptor detects projected
duplicates; duplicate IDs are excluded from feasible options and reported.
Mapping does not add noise or new waypoints at runtime. Changing tempo or rhythm
only changes the timing of the same stored joint-space polygon; quintic progress
along every leg has zero endpoint velocity and acceleration. This finite version
therefore stops instantaneously at pose accents; continuous low-speed intervals,
including joined tail/head intervals, must still pass the activity certificate.
The infinite version supplies trajectories with nonzero internal/boundary flow.

Each leg also has a midpoint stored in the frozen JSON. Strong/weak accents
change the arrival **time** of that same midpoint (`.44 + .12*strength` of the
leg's duration), changing actual approach-speed profiles. Midpoint velocity is
parallel to the stored chord, so this does not create new geometric waypoints.
It also distinguishes straight/backbeat strength patterns in actual motion,
rather than changing only an ID or metadata.

Each macro-action is `(primitive_id, rhythm_id, metrical_level)`. A rhythm is
compatible only if its event count matches the stored pose slots. Strong/weak
accents are stored pose arrivals/turning points; a rest suppresses that accent
while the other pose transitions continue. No extra geometry is inserted to
accommodate an incompatible rhythm. All 13 shared rhythms are represented by
the default library. Blocks use **beat-origin phase**, not a detected downbeat.

The planner validates actual analyzed beat timestamps at levels `k=0..3`, with
duration `4*2**k` music beats. It chooses the fastest level retaining the configured
motion/rhythm diversity. A move to a faster level needs three consecutive block
decisions by default; losing feasibility moves to a slower feasible level
immediately. BPM and playback speed stay unchanged. Failure to find a stored
certified candidate is reported as `candidate_search_failed`, separately from
the simple proven impossibility that every joint's entire position box is
narrower than the required excursion.

The default diversity floor is 32 motion IDs and 12 rhythms, clipped to the
available unique library/grammar sizes. This avoids accepting a fast mode that
has only a handful of executable movements. `terminal_hold_seconds` defaults to
zero. A controller without streaming/queues may request a certified constant
tail inside the phrase; this preserves the nominal end beat and must pass the
same activity and joined-tail checks. A too-short final piece rejects that
candidate and does not prevent trying the slower levels.

The graph starts with explicitly checked self and hub edges only. Its nodes
include rhythm and level, not just motion IDs. Each edge certificate composes
the two continuous reference certificates, equal C2 neutral boundary and the
sum of tail/head low-speed durations. Relevant direct shortcuts are validated
lazily before routing. Unchecked edges remain `unknown`; a conservative
certificate rejection is not a proof that a different interpolation could not
work. No full dense graph or collision-free claim is made.

The largest enterable recurrent SCC is selected. Independent counts describe
the whole library, tested certified feasible motions, and chosen SCC. Motion and
rhythm visitation goals form one shuffle bag. A target's minimum-edge certified
route is retained until arrival; music accent agreement and recent n-gram
penalties only resolve ties of equal path length. An unavoidable repeat never
blocks progress. Completed targets can satisfy both motion and rhythm goals.

`FinitePlanner.stats()` reports the denominators, pending target/path, completed
bags, preparation and planning latency, k changes and certificate resets.
Coverage describes scheduled reference segments; actual hardware success must
be established separately by the runner and feedback. Explicit entry/recovery
segments preserve supplied q/dq/ddq and are excluded from dance coverage.

The context hash includes relative half-beat timestamps, complete profile
fingerprint (including latency), library/mapping and rhythm configuration.
Changes invalidate the graph and restart its coverage certificate at a block
boundary. No fixed-graph coverage guarantee is transferred across such resets.

See [THEORY.md](THEORY.md) for the conditional bound and the common quickstart
under `../0_pre_and_etc/docs/QUICKSTART.md` for execution.
