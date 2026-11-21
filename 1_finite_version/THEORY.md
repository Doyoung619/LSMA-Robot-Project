# Finite reference certificates and conditional coverage

A macro-action is a fixed primitive, compatible rhythm and level k. Its vertex
is meaningful only in its timing/profile/latency context. Each piece is quintic
Bezier. Position and derivative control hulls provide conservative continuous
q/dq/ddq bounds; polynomial roots bound intervals where all joints are below the
configured meaningful-speed threshold. Stored waypoint range lower-bounds
excursion. Joining equal q/dq/ddq boundaries plus a bounded tail+head low-speed
interval certifies the concatenation's stated reference properties. Joint bounds
alone do not certify collision avoidance or actual tracking.

Let G be the fixed, explicitly certified sparse **base graph** in the chosen SCC.
Let n be the number of distinct motion goals plus distinct rhythm goals there;
let D be its directed diameter in edge count; and let tau_max bound the duration
of executing any vertex/action. Only enterable recurrent SCCs are used.

A bag contains each goal once. At each service decision, the planner retains a
minimum-edge certified path until reaching the goal. Moving from the current
action to another goal costs at most D executed actions. If the current action
itself is the goal, the implementation requires a positive-length revisit: a
self edge or an outgoing edge followed by a path back. That costs at most D+1.
The initial entry costs one action. Thus every serviced goal costs no more than
(D+1)*tau_max, and removing goals incidentally visited can only reduce work.

    B = n*(D+1)*tau_max
    bag completion <= B
    consecutive goal revisit gap <= 2*B

The revisit bound follows because every goal is visited in consecutive complete
bags; the latest gap spans at most the remainder of one bag plus the next bag.
It includes a conservative positive-cycle allowance. The fixed base graph is
not replaced by unknown edges. Lazily certified shortcut edges can be added
inside the same context: the old base paths remain available, so no service
route becomes longer than the base bound. The reported certificate retains
the base graph's D and tau_max rather than claiming the unexplored dense graph.

The assumptions are successful execution, unchanged analyzed relative timing,
profile/latency and selected SCC, persistent routes, and no external pause,
recovery or deadline interruption. Preparation, communication failure, recovery
time and context changes are outside this bound. Timing/profile changes reset
the certificate and shuffle bag explicitly. On varying beat grids the planner
revalidates actual action timestamps and resets whenever its timing fingerprint
changes; it does not promise this bound across arbitrarily changing graphs.

Reference plans alone cannot establish that a physical robot visited a goal.
Scheduled coverage and feedback-confirmed execution must remain distinct.
Finite state implies repetitions eventually occur; the guarantee is bounded
service of reachable goals, not an eternally nonrepeating sequence. Goals outside
the selected reachable SCC retain their whole-library denominator and are never
reported as achieved by a smaller SCC.
