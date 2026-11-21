"""Offline, deterministic authoring tool. NEVER called by the runtime planner."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np


FAMILIES = ("opposed_arc", "traveling_wave", "double_sweep", "cross_body",
            "fan_open", "counter_rotation", "three_pulse", "diagonal_return",
            "paired_weave", "rise_and_fold", "alternating_reach", "spiral_return",
            "lateral_bow", "expanding_wave", "mirror_exchange", "ribbon_cross")


def build(rhythms):
    """Store complete normalized waypoint geometry and boundaries before music."""
    counts = sorted({len(r.intervals) for r in rhythms})
    primitives = []
    for i in range(64):
        count = counts[i % len(counts)]
        family = i % len(FAMILIES)
        variant = i//len(FAMILIES)
        u = np.linspace(0, 1, count+1)
        # Large, fixed changes in direction, temporal shape and joint phase.
        # These are authored trajectories, not runtime jitter around an ID.
        rng = np.random.default_rng(31007+i)
        directions = rng.choice([-1., 1.], size=7)
        phase = 2*np.pi*((np.arange(7)*(1+family % 3)+variant) % 7)/7
        first = np.sin(np.pi*u)[:, None]
        travel = np.sin((1+family % 3)*np.pi*u[:, None]+phase)-np.sin(phase)
        coupled = np.sin((1+(family//3) % 3)*np.pi*u[:, None])
        shape = first*(.52*travel+.48*coupled*directions)
        shape += .28*first*(2*u[:,None]-1)*np.roll(directions, variant)
        # The independent fixed pose order prevents phase aliases for 4-event
        # primitives. Its magnitude is structural, not sub-threshold jitter.
        shape[1:-1] += .24*rng.uniform(-1,1,(count-1,7))
        shape[0] = shape[-1] = 0
        shape /= max(np.max(np.abs(shape)), 1e-8)
        # Author actual reaches, including entry/exit legs. Very short chords
        # would become long all-joint low-speed intervals after beat scaling.
        # This is a fixed, substantial pose choice, never runtime amplitude
        # shrinkage or a numerical perturbation counted as a new movement.
        for point in range(1,count):
            if np.max(np.abs(shape[point,:3]-shape[point-1,:3]))<.55:
                joint = point%3
                shape[point,joint] = -.7 if shape[point-1,joint]>=0 else .7
        if np.max(np.abs(shape[-2,:3]))<.55:
            joint = (count-1)%3
            shape[-2,joint] = -.7 if shape[-3,joint]>=0 else .7
        allowed = [r.id for r in rhythms if len(r.intervals) == count]
        primitives.append({
            "id": f"motion_{i+1:02d}", "name": f"{FAMILIES[family]}_{variant+1}",
            "family": FAMILIES[family], "event_count": count,
            "normalized_knots": np.linspace(0,1,count+1).tolist(),
            "normalized_waypoints": np.round(shape,8).tolist(),
            "normalized_midpoints": np.round((shape[:-1]+shape[1:])/2,8).tolist(),
            "boundary": {"start_q":[0.]*7, "start_dq":[0.]*7,"start_ddq":[0.]*7,
                         "end_q":[0.]*7,"end_dq":[0.]*7,"end_ddq":[0.]*7},
            "accent_positions": np.linspace(0,1,count+1)[:-1].tolist(),
            "allowed_rhythms": allowed,
            "constraint_metadata": {"requires_profile_mapping_and_continuous_certificate":True,
                                    "geometry": "fixed ordered joint-space polygon with quintic progress per leg",
                                    "event_knot_dq_ddq": "zero",
                                    "midpoint_progress_speed": "1.875 times the leg mean; parallel to fixed chord",
                                    "strength_timing": "fixed midpoint at .44+.12*next_accent_strength of each leg duration"}})
    payload = {"schema_version":1,"authored_count":64,"default_count":48,
               "mapping":"first dof columns, repeated cyclically only above 7 dof; profile-neutral plus fixed per-joint range scale",
               "authoring_seed":31007,"primitives":primitives}
    payload["geometry_sha256"] = hashlib.sha256(json.dumps(primitives,sort_keys=True).encode()).hexdigest()
    return payload


if __name__ == '__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'0_pre_and_etc'))
    from rita_core.rhythm import RHYTHMS
    target = Path(__file__).with_name('library.json')
    target.write_text(json.dumps(build(RHYTHMS),indent=2)+'\n',encoding='utf-8')
    print(target)
