# Compact validation evidence

These files preserve each source report's Git SHA/dirty state, audio hash,
profile and configuration snapshot. `source_json` records the ignored local
output path and SHA-256 of its full JSON. Paths are relative to the experiment
repository; machine-specific decoded-cache paths are omitted. Source reports
contain complete details and can be regenerated with the documented CLI.

- `quick-comparison.*` and `quick-runs.json`: four baseline/proposed methods plus
  both planners' three ablations. Red `STOPPED` labels include actual duration.
- `long-comparison.*` and `long-runs.json`: seeds 7/19, normal/tightened long runs
  and separate disconnect/missing/delayed-feedback conditions. The PNG stacks
  two seed-specific 10-condition panels vertically so all labels remain readable;
  the JSON/CSV contain the combined 20 rows.
- `billie-jean-runs.json`: final full-available-grid actual-audio runs. Their
  unplanned tail is explicit; completion does not mean every final audio sample
  has a complete dance phrase.
- `dac-playback.json`: actual PortAudio/DAC music output and online planning,
  with a mock robot and its own tested commit.
- `tempo-change-and-contracts.json`: tempo-change grid and profile-contract
  checks, with separate source/provenance for each run.
- `validation-overview.json`: scope, clean-clone checks and explicit limitations.

Raw audio, trajectories, dispatch receipts and per-event match arrays are not
included. Missed counts, unknown extra events (`null` with explanation), and
physical-robot N/A (`null`) are retained. Reference certificates do not certify
collision avoidance or measured robot tracking. Planning timings were measured
under possible CPU contention and are not hard-real-time guarantees.
