"""Freeze a development-selected detector before obtaining the held-out audio."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from dataset import _prepare_real
    out = ROOT/"improvements"
    development = json.loads((out/"results/development_split.json").read_text(encoding="utf-8"))
    config = out/"frozen_config.json"
    protocol = out/"frozen_protocol.json"
    if not protocol.exists():
        shutil.copyfile(out/"results/selected_development_config.json",config)
        record={"frozen_at_utc":datetime.now(timezone.utc).isoformat(),
                "development_track_ids":development["track_ids"],"development_n_tracks":len(development["track_ids"]),
                "selection_rule":development["selection_rule"],"configuration_count":development["configuration_count"],
                "configuration_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),
                "algorithm_sha256":hashlib.sha256((ROOT/"adapters/classical_v2.py").read_bytes()).hexdigest(),
                "holdout_definition":"All official GTZAN-mini track IDs not present in the original 50-track benchmark",
                "primary_accuracy":"Acc1 raw BPM within2%; Acc2 separately reported; failures count as incorrect",
                "primary_latency":"CPU1thread, matched file-to-result,5warmups20repetitions; secondarypreloaded5repetitions",
                "claim_rule":"Both higherAcc1 and lower matched median latency than every compared NN required for joint-superiority claim",
                "rule":"No algorithm/configuration changes based on holdout outcomes"}
        protocol.write_text(json.dumps(record,indent=2),encoding="utf-8")
    record=json.loads(protocol.read_text(encoding="utf-8"))
    assert hashlib.sha256(config.read_bytes()).hexdigest()==record["configuration_sha256"],"Frozen configuration changed"
    assert hashlib.sha256((ROOT/"adapters/classical_v2.py").read_bytes()).hexdigest()==record["algorithm_sha256"],"Frozen algorithm changed"
    development_ids=set(record["development_track_ids"])
    provenance=ROOT/"data/dataset_provenance.json"
    previous=provenance.read_bytes() if provenance.exists() else None
    records=_prepare_real(ROOT,(8.,15.,30.),2026,100)
    (out/"expanded_data_provenance.json").write_bytes(provenance.read_bytes())
    if previous is not None:
        provenance.write_bytes(previous)
    holdout=[c for c in records if c["track_id"] not in development_ids]
    ids=sorted({c["track_id"] for c in holdout})
    assert not set(ids)&development_ids
    assert len(ids)==50 and len(holdout)==150,(len(ids),len(holdout))
    (out/"holdout_manifest.json").write_text(json.dumps(holdout,indent=2),encoding="utf-8")
    (out/"holdout_split.json").write_text(json.dumps({"development_track_ids":sorted(development_ids),
        "holdout_track_ids":ids,"overlap_count":0,"n_holdout_clips":len(holdout),
        "protocol_sha256":hashlib.sha256(protocol.read_bytes()).hexdigest()},indent=2),encoding="utf-8")
    print(json.dumps({"development_tracks":len(development_ids),"holdout_tracks":len(ids),"holdout_clips":len(holdout),"overlap":0}),flush=True)


if __name__ == "__main__":
    main()
