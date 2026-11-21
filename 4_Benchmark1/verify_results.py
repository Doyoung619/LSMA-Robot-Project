"""Verify saved metrics against raw measurements without model inference."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def read(path):
    return list(csv.DictReader(path.open(encoding='utf-8')))


def main():
    results=ROOT/'results';rows=read(results/'same_robot_metrics.csv');raw=read(results/'latency_raw.csv');summary=read(results/'latency_summary.csv')
    assert len(rows)==36 and all(r['status']=='generated' for r in rows),'Main quality runs missing or failed'
    assert json.loads((results/'timing_conditions.json').read_text())['idle_confirmed'] is True,'Primary timing has no idle confirmation'
    for method in ('our_finite','our_infinite','reachy_adapted'):
        trials=[r for r in raw if r['method']==method and r['kind']=='measured']
        warm=[r for r in raw if r['method']==method and r['kind']=='warmup']
        assert len(trials)==5 and len(warm)==2 and all(r['status']=='generated' for r in trials+warm)
        values=np.array([float(r['total_generation_seconds']) for r in trials])
        aggregate=next(r for r in summary if r['method']==method)
        assert abs(float(aggregate['median_seconds'])-np.median(values))<1e-10
        assert abs(float(aggregate['min_seconds'])-values.min())<1e-10
        assert abs(float(aggregate['max_seconds'])-values.max())<1e-10
        assert aggregate['p95_seconds']=='','Five samples must not imply stable latency P95'
        assert sum(r['stage']=='stage2' and r['method']==method for r in rows)==(3 if method=='reachy_adapted' else 9)
        assert float(next(r for r in rows if r['stage']=='long' and r['method']==method)['duration_seconds'])>=156
    # Other baseline source files must remain byte-identical to the recorded calls.
    manifest=json.loads((results/'source_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['our_sources']:
        content=(ROOT.parent/entry['path']).read_bytes()
        raw=hashlib.sha256(content).hexdigest();lf=hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest()
        assert raw==entry['sha256'] or lf==entry.get('sha256_lf'),'Planner source changed beyond portable line endings'
    full=read(ROOT/'results_full_cycle/same_robot_metrics.csv')
    assert len(full)==3 and all(float(r['duration_seconds'])>=760.1 and r['status']=='generated' for r in full)
    native=read(ROOT/'results_native_full_cycle/native_reproduction.csv')
    assert len(native)==1 and float(native[0]['duration_seconds'])>=2280
    stationary_contract=__import__('metrics').evaluate
    from our_adapters import robot_profile
    from rita_core.audio import synthetic_analysis
    p=robot_profile();times=np.arange(0,12,.1);config=json.loads((ROOT/'configs/benchmark.json').read_text())['metrics']
    metric,_=stationary_contract(times,np.tile(p.neutral,(len(times),1)),synthetic_analysis(duration=30),p,config)
    assert metric['alignment_valid'] is False and metric['permitted_grid_median_error_ms'] is None
    proof={'passed':True,'main_quality_runs':36,'full_waveform_quality_runs':3,'native_full_waveform_runs':1,
           'primary_methods':3,'warmups_per_method':2,'measured_repeats_per_method':5,'latency_p95_claimed':False,
           'existing_planner_source_hashes_match':True,'deterministic_reachy_not_counted_as_multiple_quality_seeds':True,
           'stationary_alignment_not_perfect':True,'hardware_executed':False,
           'source_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'run_benchmark.py',ROOT/'metrics.py',ROOT/'our_adapters.py',ROOT/'reachy_adapter.py',ROOT/'configs/benchmark.json')},
           'raw_csv_sha256':hashlib.sha256((results/'latency_raw.csv').read_bytes()).hexdigest()}
    (results/'verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
