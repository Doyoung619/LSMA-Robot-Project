"""Verify and reproduce the released latent-conditioned native G1 policy only."""
from __future__ import annotations
import argparse
import csv
import difflib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

from prepare import ROOT,ASSETS,SOURCE_REVISION,DATASET_REVISION,sha256,source,download

for _folder in (ROOT.parent/"0_pre_and_etc",ROOT.parent/"6_container"):
    if str(_folder) not in sys.path:sys.path.insert(0,str(_folder))


def save_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def environment():
    versions={}
    for name in ("torch","numpy","mujoco","joblib","torchdiffeq","einops","Pillow"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    total=sum(p.stat().st_size for p in Path(sys.prefix).rglob("*") if p.is_file()) if sys.prefix!=sys.base_prefix else None
    gpu=None
    if shutil.which("nvidia-smi"):
        r=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"],capture_output=True,text=True)
        gpu=r.stdout.strip() if r.returncode==0 else r.stderr.strip()
    ram=None
    if sys.platform=='win32':
        import ctypes
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ('total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','available_extended')]
        mem=Memory();mem.length=ctypes.sizeof(mem);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem));ram=int(mem.total_physical)
    elif hasattr(os,'sysconf'):
        try:ram=int(os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'))
        except (ValueError,OSError):pass
    return {"os":platform.platform(),"python":platform.python_version(),"cpu":platform.processor(),"logical_cpu_count":os.cpu_count(),"ram_total_bytes":ram,
            "packages":versions,"nvidia_smi":gpu,"virtualenv_installed_bytes":total,
            "framework_size_note":"Environment installed file total is separate from downloaded model/data bytes; it is not parameter count or peak RAM.",
            "hardware_executed":False}


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def policy_source(checkout,allow_cpu_port,results):
    original=checkout/"deploy/save_TRT"
    if not allow_cpu_port:return original,[]
    target=ROOT/"external/policy_cpu_device_port"
    shutil.copytree(original,target,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
    rel="diffusions/diffusion/gaussian_diffusion.py"
    path=target/rel;before=path.read_text(encoding="utf-8")
    after=before.replace("img = th.randn(*shape).cuda()","img = th.randn(*shape, device=device)")
    after=after.replace("t = th.tensor([i] * shape[0]).cuda()","t = th.tensor([i] * shape[0], device=img.device)")
    if before.count("t = th.tensor([i] * shape[0]).cuda()")!=2:raise RuntimeError("Pinned sampler changed; device patch refuses an unknown source")
    path.write_text(after,encoding="utf-8")
    patch="".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile="upstream/"+rel,tofile="cpu_device_port/"+rel))
    (results/"cpu_device_port.patch").write_text(patch,encoding="utf-8")
    return target,[{"path":rel,"scope":"device placement only; no architecture, weights, normalization, observation or sampling-step change",
                    "original_sha256":sha256(original/rel),"ported_sha256":sha256(path),"patch":"cpu_device_port.patch"}]


def load_policy(checkout,device,allow_cpu_port,results):
    import torch
    started=time.perf_counter()
    folder,changes=policy_source(checkout,allow_cpu_port,results)
    sys.path.insert(0,str(folder))
    exporter=load_module("roboperform_official_exporter",folder/"save_tensorrt.py")
    policy,nobs,nactions=exporter.build_policy(device)
    build_seconds=time.perf_counter()-started
    start=time.perf_counter()
    checkpoint=torch.load(ROOT/"weights/student_80500.pt",map_location=device,weights_only=True)
    load_seconds=time.perf_counter()-start
    if not isinstance(checkpoint,dict):raise RuntimeError("Checkpoint is not the documented mapping")
    if "student_model_state_dict" not in checkpoint or "student_obs_norm_state_dict" not in checkpoint:
        raise RuntimeError(f"Missing released student/normalization state; keys={list(checkpoint)}")
    state=checkpoint["student_model_state_dict"]
    prefixes=("latent_encoder.","student_actor_backbone.")
    unexpected=[k for k in state if not k.startswith(prefixes)]
    if unexpected:raise RuntimeError(f"Unrecognized student state keys: {unexpected[:12]}")
    latent={k.removeprefix("latent_encoder."):v for k,v in state.items() if k.startswith("latent_encoder.")}
    backbone={k.removeprefix("student_actor_backbone."):v for k,v in state.items() if k.startswith("student_actor_backbone.")}
    norm={k.removeprefix("student_obs_norm_state_dict."):v for k,v in checkpoint["student_obs_norm_state_dict"].items()}
    policy.student_distill_net.latent_encoder.load_state_dict(latent,strict=True)
    policy.student_distill_net.student_actor_backbone.load_state_dict(backbone,strict=True)
    policy.student_obs_normalizer.load_state_dict(norm,strict=True)
    policy.eval()
    info={"architecture_source":"deploy/save_TRT/save_tensorrt.py + diffusion_policy_23_dofs.py",
          "num_observations":nobs,"num_actions":nactions,"strict_state_dict":True,
          "missing_keys":[],"unexpected_keys":[],"checkpoint_top_level_keys":list(checkpoint),
          "parameter_count":sum(p.numel() for p in policy.parameters()),
          "model_construction_seconds":build_seconds,"checkpoint_loading_seconds":load_seconds,"modifications":changes}
    return policy,info


def load_latents(checkout,device,results,allow_cpu_port=False):
    import torch
    original=checkout/"deploy/get_latent.py"
    before=original.read_text(encoding="utf-8")
    after=before
    for name in ("Roboperform_Data.pkl","dance.pt","music.pt"):
        after=after.replace("/path/to/RoboPerform/"+name,str((ROOT/"weights"/name).resolve()).replace("\\","/"))
    if allow_cpu_port:
        after=after.replace("torch.load(motion_latent_path)","torch.load(motion_latent_path, map_location='cpu')")
        after=after.replace("torch.load(music_latent_path)","torch.load(music_latent_path, map_location='cpu')")
    target=ROOT/"external/runtime_get_latent.py";target.write_text(after,encoding="utf-8")
    (results/"latent_paths.patch").write_text("".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile="upstream/get_latent.py",tofile="configured/get_latent.py")),encoding="utf-8")
    started=time.perf_counter();module=load_module("roboperform_configured_latents",target);latents=module.get_latent()
    info={"loading_seconds":time.perf_counter()-started,"num_motions":int(latents.num_motions),
          "music_shape":list(latents.music.shape),"dance_shape":list(latents.motion_latents.shape),
          "input":"released cached music and motion latents; raw-audio frontend unavailable",
          "modification":"Three asset paths configured; explicit CPU map_location only when the device-port flag is enabled"}
    return latents,info


def container_profile(model,upstream,checkout,period):
    """Describe native G1 joints without fabricating acceleration constraints."""
    import numpy as np
    import xml.etree.ElementTree as ET
    from rita_core.contracts import RobotProfile
    names=tuple(upstream.joint_xml)
    joints=[model.joint(name) for name in names]
    if any(int(j.qposadr[0])!=7+i or int(j.dofadr[0])!=6+i for i,j in enumerate(joints)):
        raise RuntimeError("G1 container joint order differs from model qpos/qvel order")
    if any(not bool(j.limited[0]) for j in joints):
        raise RuntimeError("G1 common metrics require explicit position limits for each joint")
    urdf=checkout/"deploy/unitree_description/urdf/g1/main.urdf"
    limits={j.attrib["name"]:j.find("limit") for j in ET.parse(urdf).getroot().findall("joint")}
    velocity=np.array([float(limits[name].attrib["velocity"]) for name in names])
    ranges=np.array([j.range.copy() for j in joints])
    return RobotProfile("g1-native-mujoco-UNVERIFIED","unitree_g1_mujoco",names,
        np.asarray(upstream.joint_pos_array).copy(),ranges[:,0],ranges[:,1],velocity,np.ones(len(names)),
        command_mode="simulation",command_period=period,
        workspace={"position_limit_source":"pinned upstream MuJoCo joint ranges",
                   "velocity_limit_source":"pinned upstream urdf/g1/main.urdf",
                   "unavailable_limits":["acceleration"],
                   "acceleration_limit_note":"No published acceleration bounds in this model; profile amax is an unused structural placeholder and acceleration feasibility is N/A."})


def simulate(checkout,policy,latents,args,results,progress=None):
    import numpy as np
    import torch
    import mujoco
    from rita_container import RolloutRecorder
    sys.path.insert(0,str(checkout/"deploy"))
    upstream=load_module("roboperform_mujoco_constants",checkout/"deploy/deploy_mujoco/Roboperform_mujoco.py")
    xml_path=checkout/"deploy/unitree_description/mjcf/g1_liao.xml"
    # MuJoCo 3.3's Windows path loader rejects some non-ASCII usernames. Load
    # identical XML/mesh bytes via VFS; only its mesh search directory is rebound.
    xml_text=xml_path.read_text(encoding="utf-8").replace('meshdir="../meshes/g1"','meshdir=""')
    mesh_folder=checkout/"deploy/unitree_description/meshes/g1"
    assets={p.name:p.read_bytes() for p in mesh_folder.iterdir() if p.is_file()}
    m=mujoco.MjModel.from_xml_string(xml_text,assets=assets);d=mujoco.MjData(m)
    m.opt.timestep=upstream.simulation_dt
    if (m.nq,m.nv,m.nu)!=(36,35,29):raise RuntimeError(f"Unexpected G1 model dimensions {(m.nq,m.nv,m.nu)}")
    d.qpos[7:]=upstream.joint_pos_array.copy();mujoco.mj_forward(m,d)
    history=upstream.init_history_buffers_with_step0(d,m,upstream.joint_seq,upstream.joint_pos_array_seq,15)
    target=upstream.joint_pos_array.copy()
    motion_id=torch.tensor([args.motion_id],device=latents._device)
    available=min(int(latents.music.shape[1]),int(latents.motion_latents.shape[1]),499)
    if not 0<=args.motion_id<latents.num_motions:raise ValueError("motion_id outside released latent mapping")
    period=upstream.simulation_dt*upstream.control_decimation
    steps=min(int(args.seconds/period),available)
    if steps<1:raise ValueError("Requested rollout is shorter than one official control period")
    obs=np.zeros(1400,dtype=np.float32);qpos=[];qvel=[];targets=[];actions=[];times=[];inferences=[];simulation=[];base_heights=[]
    recorder=RolloutRecorder(results/"containers",profile=container_profile(m,upstream,checkout,period),
        analysis=None,metadata={"benchmark":"Benchmark2","method":"RoboPerform released latent-conditioned G1 demo",
            "measurement_scope":"closed_loop_simulation","backend":"mujoco","hardware_executed":False,
            "seed":args.seed,"motion_id":args.motion_id,"device":args.device,
            "source_revision":SOURCE_REVISION,"dataset_revision":DATASET_REVISION,
            "time_basis":"MuJoCo simulation seconds; no audio clock or beat labels",
            "audio_alignment":None,"custom_audio_ok":False,
            "trajectory_artifact":str(results/"trajectories/g1_rollout.npz")})
    if progress is not None:
        progress["container_run_id"]=recorder.run_id;progress["container_path"]=str(recorder.run_dir)
    start=time.perf_counter();control_index=0;first_forward=None
    # Recorder work is outside policy timing. The context finalizes partial
    # observations and failed status if inference or simulation raises.
    with recorder,torch.inference_mode():
        recorder.record_sample("simulation",float(d.time),d.qpos[7:].copy(),dq=d.qvel[6:].copy(),
            metadata={"floating_base_position_m":d.qpos[:3].tolist(),"floating_base_quaternion_wxyz":d.qpos[3:7].tolist(),
                      "floating_base_velocity":d.qvel[:6].tolist(),"initial_state":True})
        recorder.record_sample("reference",float(d.time),target.copy(),metadata={"target_kind":"PD joint target"})
        for counter in range(1,steps*upstream.control_decimation+1):
            began=time.perf_counter();mujoco.mj_step(m,d)
            tau=upstream.pd_control(target,d.qpos[7:],upstream.stiffness_array,np.zeros_like(upstream.damping_array),d.qvel[6:],upstream.damping_array)
            d.ctrl[:]=tau;simulation.append(time.perf_counter()-began)
            if counter%upstream.control_decimation:continue
            music=latents.get_music(motion_id,control_index);motion=latents.get_motion_latent(motion_id,control_index)
            qseq=np.array([d.qpos[7:36][upstream.joint_xml.index(j)] for j in upstream.joint_seq])
            vseq=np.array([d.qvel[6:35][upstream.joint_xml.index(j)] for j in upstream.joint_seq])
            for name in ("base_ang_vel_his","joint_pos_his","joint_vel_his"):history[name][:-1]=history[name][1:]
            history["base_ang_vel_his"][-1]=d.qvel[3:6]
            history["joint_pos_his"][-1]=(qseq-upstream.joint_pos_array_seq)[:23]
            history["joint_vel_his"][-1]=vseq[:23]
            obs[:256]=music.cpu().numpy().reshape(-1);obs[256:320]=motion.cpu().numpy().reshape(-1)
            offset=320
            for name in ("base_ang_vel_his","joint_pos_his","joint_vel_his","action_his"):
                values=history[name].reshape(-1);obs[offset:offset+len(values)]=values;offset+=len(values)
            if offset!=1400 or not np.all(np.isfinite(obs)):raise RuntimeError("Invalid official history observation")
            tensor=torch.from_numpy(obs[None,:]).to(args.device)
            if args.device.startswith("cuda"):torch.cuda.synchronize()
            tick=time.perf_counter();action=policy(tensor)
            if args.device.startswith("cuda"):torch.cuda.synchronize()
            inference=time.perf_counter()-tick
            recorder.record_event("policy_inference",music_time=float(d.time),control_index=control_index,
                                  wall_seconds=inference,device=args.device,
                                  timing_scope="policy forward with CUDA synchronization; logging excluded")
            if action.shape!=(1,23) or not torch.isfinite(action).all():raise RuntimeError("Invalid student action output")
            if first_forward is None:
                first_forward=inference
                if progress is not None:progress["forward_ok"]=True
            inferences.append(inference)
            raw=upstream.expand_23d_to_29d(action).cpu().numpy().reshape(-1)
            history["action_his"][:-1]=history["action_his"][1:];history["action_his"][-1]=raw[:23]
            target_seq=raw*upstream.action_scale+upstream.joint_pos_array_seq
            target=np.array([target_seq[upstream.joint_seq.index(j)] for j in upstream.joint_xml])
            qpos.append(d.qpos.copy());qvel.append(d.qvel.copy());targets.append(target.copy());actions.append(raw);times.append(float(d.time));base_heights.append(float(d.qpos[2]))
            if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):raise RuntimeError("Nonfinite simulated state")
            recorder.record_sample("simulation",float(d.time),d.qpos[7:].copy(),dq=d.qvel[6:].copy(),
                metadata={"control_index":control_index,"floating_base_position_m":d.qpos[:3].tolist(),
                          "floating_base_quaternion_wxyz":d.qpos[3:7].tolist(),
                          "floating_base_velocity":d.qvel[:6].tolist(),"base_height_m":float(d.qpos[2])})
            recorder.record_sample("reference",float(d.time),target.copy(),
                metadata={"control_index":control_index,"target_kind":"PD joint target",
                          "action_29d":raw.tolist(),"action_joint_order":list(upstream.joint_seq)})
            control_index+=1
            if control_index%50==0:print(f"G1 rollout {control_index}/{steps}; t={d.time:.2f}s",flush=True)
    trajectory=results/"trajectories";trajectory.mkdir(exist_ok=True)
    np.savez_compressed(trajectory/"g1_rollout.npz",time=np.array(times),qpos=np.array(qpos),qvel=np.array(qvel),target_q=np.array(targets),action=np.array(actions))
    with (trajectory/"g1_rollout.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["simulation_seconds","base_height_m"]+["q_rad_"+j for j in upstream.joint_xml])
        for t,p in zip(times,qpos):w.writerow([t,p[2],*p[7:]])
    with (results/"latency_raw.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["control_index","policy_step_seconds","simulation_seconds"])
        for i,value in enumerate(inferences):w.writerow([i,value,times[i]])
    info={"control_steps":control_index,"control_period_seconds":period,"duration_seconds":times[-1],
          "released_latent_max_control_steps":available,"latent_looped":False,"initialization_history":"actual G1 state tiled per official deployment helper",
          "observation_order":["music 256","motion 64","base angular velocity history 15x3","q-neutral history 15x23","dq history 15x23","action history 15x23"],
          "first_valid_step_seconds":first_forward,"policy_step_seconds_median":float(np.median(inferences)),
          "policy_step_seconds_min":min(inferences),"policy_step_seconds_max":max(inferences),"policy_generation_seconds":float(sum(inferences)),
          "simulation_processing_seconds":float(sum(simulation)),"closed_loop_total_seconds":time.perf_counter()-start,
          "policy_RTF":float(sum(inferences)/times[-1]),"minimum_base_height_m":min(base_heights),
          "fall_proxy_base_below_0_35m":bool(min(base_heights)<.35),
          "input":"released latent-conditioned G1 demo; custom-audio inference unavailable","audio_alignment":None,
          "model_loading":"Identical upstream XML and mesh bytes via virtual file system; only meshdir rebound to avoid Windows non-ASCII path failure",
          "trajectory":"trajectories/g1_rollout.npz","state_source":"closed-loop MuJoCo state, not a commanded pose animation","hardware_executed":False}
    info.update(container_run_id=recorder.run_id,container_path=str(recorder.run_dir),
                container_time_basis="simulation seconds; music_time field does not imply an audio clock",
                unavailable_feasibility_limits=["acceleration"])
    if len(inferences)>=20:info["policy_step_seconds_p95"]=float(np.percentile(inferences,95))
    render_start=time.perf_counter()
    if not args.no_video:
        try:render_video(m,qpos,times,results)
        except Exception as exc:info["video_error"]=f"{type(exc).__name__}: {exc}"
    info["visualization_seconds"]=time.perf_counter()-render_start
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True);axes[0].plot(times,np.array(qpos)[:,7:],lw=.6);axes[0].set_ylabel("G1 q (rad)")
    axes[1].plot(times,base_heights);axes[1].axhline(.35,color="red",ls="--",label="fall proxy (not ground truth)");axes[1].legend();axes[1].set(ylabel="Base z (m)",xlabel="Simulation seconds")
    fig.suptitle("Released-latent G1 closed-loop MuJoCo state; custom audio unavailable");fig.tight_layout();fig.savefig(results/"g1_state.png",dpi=130);plt.close(fig)
    return info


def render_video(model,poses,times,results):
    import mujoco
    from PIL import Image,ImageDraw
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=480,width=640)
    camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.8];camera.distance=3.0;camera.azimuth=135;camera.elevation=-12
    out=results/"videos";out.mkdir(exist_ok=True)
    command=["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","640x480","-r","25","-i","pipe:0","-an","-c:v","libx264","-pix_fmt","yuv420p",str(out/"g1_released_latent.mp4")]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for i in range(0,len(poses),2):
            data.qpos[:]=poses[i];mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,640,47),fill=(8,12,20));draw.text((10,8),f"RoboPerform released-latent G1 / MuJoCo state / t={times[i]:.2f}s",fill="white")
            draw.text((10,26),"Custom audio unavailable / silent / device-only CPU port if specified",fill=(255,210,120))
            process.stdin.write(im.tobytes())
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError("Video encoder failed")
    finally:renderer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native",action="store_true",help="Execute the released-latent G1 policy only")
    p.add_argument("--headless",action="store_true",help="Offline renderer; no interactive viewer or hardware")
    p.add_argument("--inspect",action="store_true",help="Strict checkpoint load only; not dance reproduction")
    p.add_argument("--device",default="cpu");p.add_argument("--allow-cpu-device-port",action="store_true")
    p.add_argument("--motion-id",type=int,default=0);p.add_argument("--seconds",type=float,default=2)
    p.add_argument("--seed",type=int,default=7);p.add_argument("--threads",type=int,default=1)
    p.add_argument("--no-video",action="store_true");p.add_argument("--results",default="results")
    args=p.parse_args();results=Path(args.results);results=results if results.is_absolute() else ROOT/results;results.mkdir(parents=True,exist_ok=True)
    status={"method":"RoboPerform released latent-conditioned G1 demo","artifact_verified":False,"model_loaded_strict":False,
            "forward_ok":False,"rollout_ok":False,"custom_audio_ok":False,"hardware_executed":False,
            "custom_audio_reason":"No released raw-audio encoder/adapter inference entry point and checkpoint path found; music.pt is indexed by stored motion name/time.",
            "audio_alignment":None,"same_robot_comparison":"unavailable: native G1 embodiment only",
            "source_revision":SOURCE_REVISION,"dataset_revision":DATASET_REVISION,"arguments":vars(args),"errors":[]}
    env=environment();save_json(results/"environment.json",env)
    try:
        checkout=source()
        asset_names=["student_80500.pt","music.pt","dance.pt","Roboperform_Data.pkl"] if args.native else ["student_80500.pt"]
        artifacts=[]
        for name in asset_names:
            path=ROOT/"weights"/name;size,digest=ASSETS[name]
            if not path.exists() or path.stat().st_size!=size or sha256(path)!=digest:raise RuntimeError(f"Run prepare.py --assets deployment first: {name} is missing or invalid")
            artifacts.append({"name":name,"bytes":size,"sha256":digest})
        status["artifact_verified"]=True;status["artifacts"]=artifacts
        import torch
        torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
        if args.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("Requested CUDA is unavailable")
        env.update(torch_cuda_available=torch.cuda.is_available(),torch_cuda_version=torch.version.cuda,threads=args.threads)
        save_json(results/"environment.json",env)
        policy,model_info=load_policy(checkout,args.device,args.allow_cpu_device_port,results)
        status["model_loaded_strict"]=True;status["model"]=model_info
        if args.native:
            latents,latent_info=load_latents(checkout,args.device,results,args.allow_cpu_device_port);status["latents"]=latent_info
            info=simulate(checkout,policy,latents,args,results,status)
            status["forward_ok"]=True;status["rollout_ok"]=True;status["rollout"]=info
    except Exception as exc:
        status["errors"].append({"type":type(exc).__name__,"message":str(exc)})
        (results/"last_attempt_traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        print(traceback.format_exc(),flush=True)
    finally:
        if not status["errors"]:(results/"last_attempt_traceback.txt").unlink(missing_ok=True)
        save_json(results/"run_status.json",status)
        with (results/"native_reproduction.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["method","source_revision","artifact_verified","strict_model_loaded","forward_ok","rollout_ok","custom_audio_ok","input","duration_seconds","audio_alignment","hardware_executed"])
            w.writerow([status["method"],SOURCE_REVISION,status["artifact_verified"],status["model_loaded_strict"],status["forward_ok"],status["rollout_ok"],False,"released cached latents",status.get("rollout",{}).get("duration_seconds"),"N/A",False])
        print(json.dumps(status,indent=2,ensure_ascii=False),flush=True)
    if not status["model_loaded_strict"] or (args.native and not status["rollout_ok"]):raise SystemExit(2)

if __name__=="__main__":main()
