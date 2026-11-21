"""Refresh portable provenance and reports without rerunning/tuning generators."""
import importlib.metadata
import json
from pathlib import Path
from run_benchmark import ROOT,REPO,write_json,source_text_hash,summarize


def main():
    for folder in (ROOT/'results',ROOT/'results_full_cycle',ROOT/'results_native_full_cycle'):
        path=folder/'source_manifest.json'
        manifest=json.loads(path.read_text(encoding='utf-8'))
        for entry in manifest['our_sources']:
            entry['sha256_lf']=source_text_hash(REPO/entry['path'])
            entry['text_hash_scope']='UTF-8 text; CRLF normalized to LF for portable Git checkout verification; raw original sha256 retained'
        manifest['input_conditions']={'real_music_stages':['stage1','stage2','long'],
            'synthetic_stages':['stress90','stress120','stress180','synthetic_full_cycle'],
            'synthetic_rows_do_not_use_audio_file':True,
            'active_scope':'known synthetic beat grid only' if folder.name!='results' else 'same real-audio crops plus separately labeled synthetic timing stress'}
        manifest['quality_processing_time_scope']='Exploratory wall time; background experiments may exist. Primary repeat latency is in results/latency_raw.csv with idle confirmation.'
        write_json(path,manifest)
        summarize(folder)
    packages=('numpy','scipy','soundfile','sounddevice','matplotlib','Pillow','PyYAML')
    (ROOT/'requirements-lock.txt').write_text('-e ..\n'+'\n'.join(f'{name}=={importlib.metadata.version(name)}' for name in packages)+'\n',encoding='utf-8')
    print('Portable source hashes, scope, reports, and declared dependency lock refreshed; raw measurements unchanged.')


if __name__=='__main__':main()
