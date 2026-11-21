"""Pin official source and download only selected dataset artifacts, with hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
SOURCE_URL="https://github.com/gentlefress/RoboPerform.git"
SOURCE_REVISION="e6e2a0e4cdb20f081c1a3d46d648d7a202816779"
DATASET_ID="SanQing1/RoboPerform"
DATASET_REVISION="3fdfcd75480450ec2f8f884723b5724cb3a8051b"
ASSETS={
 "student_80500.pt":(563075994,"93145065d01bc6a8b3c8567a7edfb36a15eb1621c3d02d60c399a0a07c360dfd"),
 "music.pt":(306873850,"15a6ff08bb3dc7bd83d3666f333b1501a3bdfa2a29f0203677829aab907e4bae"),
 "dance.pt":(129106,"402406733767b0967df016c995cf388616d0a4f697c3a5fc4a6967fe6b6da6b2"),
 "Roboperform_Data.pkl":(578989934,"fffae4d1e0c9caec9ec5d36295bf09a1042df88b799fd39b3f8c857f24e02a58"),
 "model_80500.engine":(188901556,"28c31d45df37d062aa30a0a2082ce011dad004f85334275d5961965d1c6bb7b6"),
 "model_88500.engine":(189040700,"76b7535f1cbf24022ebfe27b76ed28b8e2154b2786d1bd7e1c2e7579ad45b01f"),
 "teacher_74000.pt":(19347818,"a5840c94361e71e72dc7736ee3c4d5a2cb62b52683eab31b5ef789ba9804bd52"),
}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()

def source():
    checkout=ROOT/"external/RoboPerform"
    checkout.parent.mkdir(parents=True,exist_ok=True)
    if not (checkout/".git").exists():
        subprocess.run(["git","clone","--no-checkout",SOURCE_URL,str(checkout)],check=True)
    current=subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()
    if current!=SOURCE_REVISION:
        subprocess.run(["git","-C",str(checkout),"fetch","origin",SOURCE_REVISION],check=True)
    subprocess.run(["git","-C",str(checkout),"checkout","--detach",SOURCE_REVISION],check=True)
    return checkout

def download(name):
    expected_size,expected_hash=ASSETS[name]
    folder=ROOT/"weights";folder.mkdir(parents=True,exist_ok=True)
    path=folder/name
    started=time.perf_counter()
    if not path.exists():
        url=f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{name}?download=true"
        partial=path.with_suffix(path.suffix+".partial")
        request=urllib.request.Request(url,headers={"User-Agent":"Rita-Robot-reproducibility/1.0"})
        with urllib.request.urlopen(request,timeout=120) as response,partial.open("wb") as f:
            count=0;previous=0
            while block:=response.read(8*1024*1024):
                f.write(block);count+=len(block)
                if count-previous>=100*1024*1024:
                    print(f"{name}: {count/1e6:.1f}/{expected_size/1e6:.1f} MB",flush=True);previous=count
        if partial.stat().st_size!=expected_size:raise ValueError(f"{name}: size mismatch")
        partial.replace(path)
    digest=sha256(path)
    if path.stat().st_size!=expected_size or digest!=expected_hash:
        raise ValueError(f"{name}: artifact size/hash mismatch; refusing checkpoint")
    return {"filename":name,"bytes":path.stat().st_size,"sha256":digest,"verified":True,
            "download_or_verify_seconds":time.perf_counter()-started,
            "url":f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{name}"}

def inspect_source(checkout):
    files=list(checkout.rglob("*.py"))
    relevant=[]
    for p in files:
        text=p.read_text(encoding="utf-8",errors="replace")
        if any(token in text for token in ("get_music(","music_latent", "student_obs_norm", "audio_encoder")):
            relevant.append(str(p.relative_to(checkout)).replace("\\","/"))
    return {"official_url":SOURCE_URL,"revision":SOURCE_REVISION,
            "license":{"readme_claim":"BSD-3-Clause","root_license_file_present":(checkout/"LICENSE").exists(),
                       "dataset_license_metadata":None,"note":"Weights/data remain local. Upstream README links a missing LICENSE file at this revision; do not infer a weight redistribution license."},
            "relevant_files":relevant,
            "custom_audio":{"supported_in_released_deployment":False,
                "finding":"Released get_latent.py indexes music.pt by motion_new_name and time index; no documented released raw-audio encoder+adapter checkpoint path was found.",
                "label":"Released latent-conditioned G1 demo; custom-audio inference unavailable."},
            "modifications":"No architecture, normalization, observation or latent replacement. Optional CPU device-placement patch is recorded separately when requested."}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets",choices=["none","student","deployment","engine"],default="deployment")
    p.add_argument("--refresh-metadata",action="store_true")
    args=p.parse_args();results=ROOT/"results";results.mkdir(exist_ok=True)
    checkout=source();manifest=inspect_source(checkout)
    names={"none":[],"student":["student_80500.pt"],"deployment":["student_80500.pt","music.pt","dance.pt","Roboperform_Data.pkl"],"engine":["model_80500.engine","music.pt","dance.pt","Roboperform_Data.pkl"]}[args.assets]
    manifest.update(dataset_id=DATASET_ID,repo_type="dataset",dataset_revision=DATASET_REVISION,
                    hosted_file_inventory=[{"filename":k,"bytes":v[0],"sha256":v[1]} for k,v in ASSETS.items()],
                    selected_assets=names,selected_total_bytes=sum(ASSETS[n][0] for n in names),artifacts=[])
    if args.refresh_metadata:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/{DATASET_REVISION}",timeout=30) as r:
            meta=json.load(r)
        manifest["latest_pinned_host_inventory"]=[{"path":x["path"],"size":x["size"],"sha256":x.get("lfs",{}).get("oid")} for x in meta]
    path=results/"source_manifest.json"
    try:
        for name in names:
            row=download(name);manifest["artifacts"].append(row)
            path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            print(f"VERIFIED {name}: {row['bytes']} bytes",flush=True)
    finally:path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"manifest":str(path),"asset_bytes":sum(x["bytes"] for x in manifest["artifacts"])}),flush=True)

if __name__=="__main__":main()
