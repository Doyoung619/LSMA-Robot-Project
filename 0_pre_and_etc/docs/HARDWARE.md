# Misty / Kinova connection

Status: **implemented, hardware-unverified**. Development and automated tests did
not connect to, move, calibrate, or home a real robot. Mock results are simulated.

Commands below are relative to the Git checkout root, currently `experiment/`.
From the containing Rita-Robot workspace, prepend `experiment/` to these paths.
`doctor` is read-only by default. It does not clear faults or move joints.

## Shared contract

All planner, profile and feedback values use radians, radians/second,
radians/second² and seconds. Ordered `joint_names` are mandatory. The adapter
performs any external degree conversion. No implicit unit, model or joint-order
inference is performed. A profile contains limits, neutral pose, command period,
feedback freshness, latency policy, endpoint configuration and workspace scope.

Copy the appropriate example under `0_pre_and_etc/rita_core/profiles/` to a
gitignored `*.local.json` or `*.local.yaml` file. JSON needs no YAML package.
Set the endpoint in that one file. Inspect the actual robot's model, firmware,
limits, command mode, joint ordering and reachable workspace before setting
`local_verified: true` and `workspace.verified: true`. Example joint boxes are
**not** collision certificates. Credentials belong in environment variables or
gitignored local configuration. The profiles do not contain credentials.

The new execution path permits physical motion only with a verified local
profile and explicit `--execute`. Supplying `--execute` alone is insufficient.
The historical `rita.py` entry point remains archived for reproducibility; its
legacy operational commands are not the new guarded execution interface.

```bash
python 0_pre_and_etc/run.py doctor --backend misty --profile misty.local.json --read-only
python 0_pre_and_etc/run.py doctor --backend kinova --profile kinova.local.json --read-only
```

Use the doctor's explicit `--probe` option to inspect a configured device
after the offline configuration check. No probe has been performed during this
implementation.

Adapters expose `connect`, `get_capabilities`, `read_state`, `execute_segment`,
`stop`, `close`. Check capabilities before selecting execution policy. Unsupported
queue or interpolation behavior is rejected instead of emulated as success.

## Misty II

Install the optional WebSocket dependency with `pip install -e '.[misty]'`.
No SDK is required for the standard HTTP requests. Set `endpoint.host` to the
robot's IP; HTTP and WebSocket ports default to 80. Wheels are excluded.

The exact internal joint order is:

```text
head_pitch, head_roll, head_yaw, left_arm, right_arm
```

The adapter compiles a validated reference to rate-limited target poses. Head
requests use `POST /api/head`, explicit degree units and **Duration only**.
Arm requests use `POST /api/arms/set` with positions and configured velocity
percentages. There is no invented arm duration field. A percent is a firmware
speed setting, not rad/s. The relevant API semantics are documented in the
[official Misty API](https://docs.mistyrobotics.com/misty-ii/web-api/api-reference/).

HTTP success means acceptance, not measured completion. Misty's internal pose
interpolator does not promise exact spline tracking. The reference certificate
therefore does not certify the actual motor trajectory. Monitor actuator
feedback and tune conservative command rates, speed percentages and tracking
thresholds for the specific robot.

Feedback uses WebSocket `/pubsub` and one `ActuatorPosition` subscription per
head/arm joint. The adapter reads advertised WebSocket properties and chooses
`sensorName`, `sensorId` or `actuatorId` filters. Current and deprecated field
casing, as well as per-sensor event names, are supported. If firmware exposes a
different filter schema, explicitly provide `endpoint.feedback_filter_property`
and all five `endpoint.feedback_sensor_values`; unknown schemas fail clearly.
The sensor stream reports degrees, converted to radians by the adapter. See the
[official actuator event documentation](https://docs.mistyrobotics.com/misty-ii/robot/sensor-data/).

Feedback timestamps use host monotonic **receipt** time; remote `created`
timestamps are retained without assuming synchronized clocks. Velocity is an
unfiltered first difference of asynchronous position samples, labeled as such.
This is not a certified acceleration measurement. Reception latency and sensor
sampling uncertainty remain part of real motion-event timing uncertainty.

Misty has no verified controller future queue or C2 chunk handoff here.
`queue=false`, `future_execution=false`, `streaming_c2=false`. Only zero-velocity,
zero-acceleration boundary chunks are accepted. A local dispatch worker waits for
absolute host deadlines outside the audio callback. It does not issue overdue
commands in a burst. Missing feedback, excessive tracking error or network
failure cancels pending host dispatch and requests `POST /api/halt`.
`/api/drive/stop` is not used as an all-motor stop.

## Kinova Gen3 L53, six joints

The archived repository identifies **MODEL_ID_L53, six DOF, spherical wrist**.
Its official URDF, local feedback guards, Kortex degree conversion and previous
logs remain under `0_pre_and_etc/rita_robot` and `data/robot`. The new path does
not change firmware limits or run a Python sleep loop as a high-frequency raw
actuator servo.

The implemented trajectory backend uses the official
[ros2_kortex](https://github.com/Kinovarobotics/ros2_kortex) driver and ROS2
`FollowJointTrajectory`. Install/source the matching driver's supported ROS2
environment on the robot PC; no OS or ROS upgrade is performed by this project.
ROS packages are optional imports, so mock runs do not need ROS or Kortex.

Set `endpoint.namespace`, `endpoint.action` and `endpoint.joint_states_topic`
once in the local profile. The adapter accepts exactly `joint_1` through
`joint_6` in that order and SI units. A prefixed, reordered or extra-joint state
stream fails with the expected and received order. Configure a dedicated correct
stream rather than silently guessing a mapping. The profile's example neutral
pose comes from the archived snapshot; it is not an automatic home target.

The action contains every piecewise-quintic knot with q/dq/ddq. The installed
controller must use its documented quintic spline interpolation when all three
fields are supplied. Verify controller parameters and set
`endpoint.controller_parameters_verified` only after that check. See the
[ROS2 controller trajectory representation](https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/trajectory.html).
The cited development documentation does not certify the version installed on
the robot PC; check that version's docs and behavior.

There is one active action, no verified future chunk queue, and no normal-path
preemption. Chunks must start/end at zero velocity and acceleration. This is an
explicit execution constraint, separate from the mock planner's optional flowing
boundaries. `FollowJointTrajectory` acceptance is recorded separately from action
result and joint-state feedback. Cancellation calls the action cancellation API;
physical standstill must still be checked from feedback. The controller's local
cancel/hold/deceleration policy must be verified. See the
[ROS2 controller action semantics](https://control.ros.org/rolling/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html).

For controllers without a verified future queue, planners may reserve a
certified terminal hold as part of the nominal segment. Both hardware compilers
trim only an **exact constant C2 suffix** whose six Bézier controls all equal the
final pose; they do not infer a hold from sampled low speed. The controller then
holds the same pose while the runtime waits for completion and submits the next
future timestamp. Receipts distinguish `controller_action_end_monotonic` from
`nominal_segment_end_monotonic`. The hold must cover dispatch lead plus response
margin, and the common activity checker must still certify the combined low-speed
tail/head duration. Without that window, continuous chaining is unavailable;
report a scheduling failure rather than silently stretching the audio timeline.

ROS header timestamps are computed from a sampled mapping between the host's
monotonic clock and `node.get_clock()`. `use_sim_time=true` is rejected because
the DAC clock needs an explicitly established simulation-clock mapping.
`scheduled_monotonic` denotes intended execution time; dispatch lead is a
separate requirement. The adapter does not subtract both physical latency and
dispatch lead from a timestamped controller's start time.

Gen3 Lite and seven-joint Gen3 are not automatically substituted for this profile.
Legacy JACO/JACO2/MICO use a different driver family,
[kinova-ros](https://github.com/Kinovarobotics/kinova-ros), and are **not supported**
by this adapter. Their dedicated profile/driver must be verified separately.
The earlier SDK integration is the
[official Kortex Gen3/Gen3 Lite API family](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L),
not a legacy Kinova API.

## Calibration and final execution

Calibration must be requested explicitly using a locally verified low-excursion
reference and actual feedback. Estimate onset/position-crossing lag from actual
joint motion; HTTP RTT is not motor latency. Keep audio output time, command
send/accept time and feedback-observed event time separate. Record the feedback
period, receipt-clock uncertainty and derivative estimator. Update only future,
unexecuted commands after a calibration change. Do not alter an executing prefix.

The CLI provides calibration recording/estimation; inspect `run.py calibrate
--help` before the locally requested motion. An explicit hardware run uses:

```bash
python 0_pre_and_etc/run.py run --mode finite --backend misty --profile misty.local.json --audio /actual/path/Billie_Jean.mp3 --execute
python 0_pre_and_etc/run.py run --mode finite --backend kinova --profile kinova.local.json --audio /actual/path/Billie_Jean.mp3 --execute
```

For infinite on a controller without verified flowing chunk support, select the
planner's stop-boundary setting. A short follow-up connection task should inspect
the actual controller and configure its supported execution constraints rather
than rewrite the planner or claim mock streaming matches that hardware.

Stop/Ctrl-C, communication loss, stale feedback or uncertain state cancels queued
host work and calls the robot-specific stop. There is no automatic home, fault
clear or restart. A software stop is not a substitute for the physical emergency
stop. Physical motion and latency metrics remain **N/A** until actually measured.

The existing `0_pre_and_etc/STOP_RITA` latch is preserved. Its presence blocks
hardware startup and is checked during dispatch/playback. The runtime never
clears it. Clear it deliberately only after local inspection and an explicit
decision to permit a new hardware run. Mock runs ignore this hardware latch.

The Kinova **example configuration** explicitly permits at most 0.65 seconds of
whole-robot low speed to accommodate the slow reference plus a controller
dispatch hold. This is not a runtime relaxation or hardware measurement; mock's
default remains 0.35 seconds. A reference-only check with a 0.28-second terminal
hold found feasible candidates and a maximum initial-segment low-speed interval
of about 0.5773 seconds under that example. Recheck the mapped library, joins and
actual hardware behavior after local calibration rather than treating this
example as a collision or long-run certificate.
