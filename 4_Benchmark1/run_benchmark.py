"""Offline dance-generation comparison, never a robot-motion entry point."""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_name]="1"
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
from our_adapters import robot_profile,create,generate,REPO
# Direct-script use works before an editable repository install, too.
if str(REPO/"6_container") not in sys.path:sys.path.insert(0,str(REPO/"6_container"))
from rita_container import RolloutRecorder
from metrics import evaluate,seed_distances
from reachy_adapter import load_upstream,sample_native,adapt_misty,METHOD_NATIVE,METHOD_ADAPTED
from rita_core.audio import analyze_audio,find_audio,synthetic_analysis,file_hash

METHODS=("our_finite","our_infinite","reachy_adapted")
LABELS={"our_finite":"Our finite (existing)","our_infinite":"Our infinite (existing)","reachy_adapted":METHOD_ADAPTED,"reachy_native":METHOD_NATIVE}


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def source_text_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,tuple,dict,np.ndarray)) else v for k,v in row.items()})


def environment():
    packages={}
    for name in ("numpy","scipy","soundfile","matplotlib","Pillow","torch","mujoco"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    def command(args):
        try:
            p=subprocess.run(args,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=20)
            return {"returncode":p.returncode,"stdout":p.stdout.strip(),"stderr":p.stderr.strip()}
        except (OSError,subprocess.TimeoutExpired) as exc:return {"available":False,"error":str(exc)}
    host_probe=command(["powershell","-NoProfile","-NonInteractive","-Command","$probeCpu=Get-CimInstance Win32_Processor; $probeMem=Get-CimInstance Win32_ComputerSystem; @{cpu_name=$probeCpu.Name;physical_cores=$probeCpu.NumberOfCores;logical_processors=$probeCpu.NumberOfLogicalProcessors;ram_bytes=$probeMem.TotalPhysicalMemory}|ConvertTo-Json -Compress"]) if os.name=="nt" else command(["uname","-a"])
    return {"platform":platform.platform(),"python":sys.version,"executable":sys.executable,"cpu":platform.processor(),"host_hardware_probe":host_probe,
            "logical_cpus":os.cpu_count(),"packages":packages,"device":"CPU","batch_size":1,"threads":1,
            "thread_environment":{n:os.environ.get(n) for n in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
            "nvidia_smi":command(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv,noheader"]),
            "git":command(["git","-C",str(REPO),"rev-parse","HEAD"]),"dirty":command(["git","-C",str(REPO),"status","--porcelain"]),
            "timing_scope":"Serial methods on CPU; no hardware commands. Timed work uses shared cached beats, not raw-audio end-to-end inference."}


def generate_method(method,analysis,profile,start_beat,seconds,seed):
    times=analysis.beat_time(start_beat)+np.arange(0,seconds,profile.command_period)
    if times[-1]>=analysis.beats[-1]:raise ValueError("Requested crop exceeds analyzed beat timeline")
    phase=np.interp(times,analysis.beats,np.arange(len(analysis.beats)))
    if method.startswith("our_"):
        planner,init=create(method,analysis,profile,seed)
        output=generate(planner,analysis,profile,start_beat,times)
    else:
        began=time.perf_counter();source=load_upstream();init=time.perf_counter()-began
        began=time.perf_counter();native=sample_native(times,phase);generation=time.perf_counter()-began
        output={"native":native,"generation_seconds":generation,"adaptation_seconds":0.,"limit_handling_seconds":0.,
                "metadata":{"source":source,"scheduler":native["scheduler"],"selection":"deterministic official cyclic preview, no randomness/novelty added",
                            "metrical_timeline":[{"music_start_seconds":float(times[0]),"metrical_level":0,"motion_beat_ratio":1.0}],
                            "hardware_executed":False,"simulation_state_available":False,"chunk_deadline_misses":None},
                "input":"shared cached estimated beat timestamps; native functions receive global cumulative beat phase"}
        if method=="reachy_adapted":
            began=time.perf_counter();mapped=adapt_misty(native,profile,anchor_first=True);output["adaptation_seconds"]=time.perf_counter()-began
            output["q"]=mapped["q"];output["metadata"]["mapping"]=mapped["mapping"]
    output.update(times_seconds=times,beat_phase=phase,init_seconds=init)
    output["total_generation_seconds"]=sum(output[k] for k in ("generation_seconds","adaptation_seconds","limit_handling_seconds"))
    return output


def competing_workers():
    """Observe other benchmark inference; never terminate another experiment."""
    if os.name=="nt":
        script="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match '3_BPM_check|5_Benchmark2' } | Select-Object -ExpandProperty ProcessId"
        p=subprocess.run(["powershell","-NoProfile","-NonInteractive","-Command",script],capture_output=True,text=True,timeout=20)
        if p.returncode:raise RuntimeError("Could not inspect competing benchmark workers before timing")
        return [int(s) for s in p.stdout.split() if s.isdigit()]
    result=[]
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            command=(directory/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if 'python' in command and any(name in command for name in ('3_BPM_check','5_Benchmark2')):result.append(int(directory.name))
        except OSError:pass
    return result


def wait_for_timing_window(timeout):
    began=time.perf_counter();checks=0
    while True:
        busy=competing_workers();checks+=1
        if not busy:return {"idle_confirmed":True,"wait_seconds":time.perf_counter()-began,"checks":checks,
                            "policy":"other 3/5 benchmark Python jobs must finish; no process was stopped; methods run serially"}
        if time.perf_counter()-began>=timeout:raise RuntimeError(f"Timing not measured: other experiments still active (PIDs {busy}); retry --timing-only after they finish")
        print(f"WAIT_FOR_IDLE competing={busy} waited={time.perf_counter()-began:.1f}s",flush=True)
        time.sleep(5)


def save_trajectory(output,run_id,result):
    folder=output/"trajectories";folder.mkdir(parents=True,exist_ok=True)
    data={"times_seconds":result["times_seconds"],"beat_phase":result["beat_phase"]}
    if "q" in result:data["q"]=result["q"]
    if "native" in result:
        for key in ("position_m","orientation_rad","antenna_rad"):
            data[key]=result["native"][key]
        data["primitive_ids"]=np.asarray(result["native"]["primitive_ids"])
    np.savez_compressed(folder/(run_id+".npz"),**data)
    fields=[("q_rad",data["q"])] if "q" in data else [("position_m",data["position_m"]),("orientation_rad",data["orientation_rad"]),("antenna_rad",data["antenna_rad"])]
    with (folder/(run_id+".csv")).open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["absolute_audio_seconds","global_beat_phase"]+[f"{label}_{j}" for label,values in fields for j in range(values.shape[1])])
        for i,t in enumerate(data["times_seconds"]):writer.writerow([t,data["beat_phase"][i],*[x for _,values in fields for x in values[i]]])
    write_json(folder/(run_id+".json"),{k:v for k,v in result.items() if k not in ("q","analytic_dq","analytic_ddq","native","times_seconds","beat_phase")})


def quality_run(method,analysis,profile,cfg,stage,start_beat,seconds,seed,output):
    run_id=f"{stage}__{method}__seed{seed}__beat{start_beat}"
    print(f"GENERATE {run_id} {seconds:.3f}s",flush=True)
    row={"run_id":run_id,"stage":stage,"method":method,"method_label":LABELS[method],"seed":seed,
         "start_beat":start_beat,"crop_start_seconds":analysis.beat_time(start_beat),"duration_seconds":seconds,
         "audio_sha256":analysis.audio_hash,"input":"shared estimated beat reference","status":"failed","hardware_executed":False}
    recorder=None
    try:
        if method!="reachy_native":
            recorder=RolloutRecorder(output/"containers",profile=profile,analysis=analysis,
                metadata={"benchmark":"Benchmark1","benchmark_run_id":run_id,"method":method,
                          "stage":stage,"seed":seed,"measurement_scope":"offline_commanded",
                          "hardware_executed":False,"input":"shared cached beat timestamps",
                          "trajectory_artifact":str(output/"trajectories"/(run_id+".npz"))},
                metric_config=cfg["metrics"])
            row.update(container_run_id=recorder.run_id,container_path=str(recorder.run_dir))
        result=generate_method(method,analysis,profile,start_beat,seconds,seed)
        if stage.startswith(('stress','synthetic')):
            result['input']='known synthetic beat timestamps; no raw-audio naturalness evaluation'
        for k in ("init_seconds","generation_seconds","adaptation_seconds","limit_handling_seconds","total_generation_seconds"):row[k]=result[k]
        row["rtf"]=result["total_generation_seconds"]/seconds
        save_trajectory(output,run_id,result)
        if recorder is not None:
            # Export is outside the measured generation region and preserves the
            # original, uncorrected q values, including any limit violations.
            for i,t in enumerate(result["times_seconds"]):
                recorder.record_sample("reference",float(t),result["q"][i],
                    dq=result["analytic_dq"][i] if "analytic_dq" in result else None,
                    metadata={"global_beat_phase":float(result["beat_phase"][i]),
                              **({"declared_accent_times":result.get("declared_accent_times",[])} if i==0 else {})})
            for target in result.get("declared_accent_times",[]):
                recorder.record_event("declared_accent",music_time=float(target))
            for key in ("init_seconds","generation_seconds","adaptation_seconds","limit_handling_seconds","total_generation_seconds"):
                recorder.record_event("generation_timing",stage=key,wall_seconds=float(result[key]),
                                      timing_scope="offline generation; container export excluded")
            recorder.record_event("generation_metadata",details=clean(result["metadata"]))
        if method!="reachy_native":
            metrics,details=evaluate(result["times_seconds"],result["q"],analysis,profile,cfg["metrics"],result.get("declared_accent_times"))
            row.update(metrics);row["status"]="generated"
            if method=="reachy_adapted":
                begin_phase=float(result["beat_phase"][0]);end_phase=float(result["beat_phase"][-1]);jumps=[]
                for switch in np.arange((math.floor(begin_phase/8)+1)*8,end_phase,8):
                    phases=np.array([switch-1e-8,switch+1e-8])
                    at=np.interp(phases,np.arange(len(analysis.beats)),analysis.beats)
                    pair=sample_native(at,phases)["orientation_rad"][:,[1,0,2]]
                    jumps.append(float(np.max(np.abs(pair[1]-pair[0]))))
                row["exact_switch_position_jump_max_rad"]=max(jumps,default=0.)
                row["exact_switches_checked"]=len(jumps)
            meta=result["metadata"]
            row["metrical_levels_used"]=sorted(set(item["metrical_level"] for item in meta.get("metrical_timeline",[])))
            row["motion_ratios_used"]=sorted(set(item["motion_beat_ratio"] for item in meta.get("metrical_timeline",[])))
            row["chunk_deadline_misses"]=meta.get("chunk_deadline_misses")
            write_json(output/"_scratch"/(run_id+"_metrics.json"),details)
            recorder.close(status="completed",native_status=row["status"])
            return row,result,details
        row.update(status="generated_native",embodiment="Reachy head pose/antennas",audio_alignment=None,kinematic_feasibility=None,
                   container_path=None,container_note="Native xyz/Euler/antenna channels remain in native trajectory artifacts; common joint containers cover the adapted same-robot q trajectories.")
        return row,result,{}
    except Exception as exc:
        row["status"]="failed"
        row["error"]=f"{type(exc).__name__}: {exc}"
        if recorder is not None:
            recorder.close(status="failed",error=row["error"])
        print(f"FAILED {run_id}: {row['error']}",flush=True)
        return row,None,{}


def summarize(output):
    rows=list(csv.DictReader((output/"same_robot_metrics.csv").open(encoding="utf-8"))) if (output/"same_robot_metrics.csv").exists() else []
    raw=list(csv.DictReader((output/"latency_raw.csv").open(encoding="utf-8"))) if (output/"latency_raw.csv").exists() else []
    timing=[]
    for method in METHODS:
        samples=[float(r["total_generation_seconds"]) for r in raw if r["method"]==method and r["kind"]=="measured" and r["status"]=="generated"]
        init=[float(r["init_seconds"]) for r in raw if r["method"]==method and r["kind"]=="measured" and r["status"]=="generated"]
        if samples:timing.append({"method":method,"method_label":LABELS[method],"repetitions":len(samples),"workload_seconds":12.,
                                  "median_seconds":float(np.median(samples)),"min_seconds":min(samples),"max_seconds":max(samples),
                                  "initialization_median_seconds":float(np.median(init)),"rtf_median":float(np.median(samples)/12),
                                  "p95_seconds":float(np.percentile(samples,95)) if len(samples)>=20 else None,
                                  "scope":"motion generation + documented adaptation + zero common correction; no visualization/download"})
    write_csv(output/"latency_summary.csv",timing)
    native=list(csv.DictReader((output/'native_reproduction.csv').open(encoding='utf-8'))) if (output/'native_reproduction.csv').exists() else []
    scope_rows=rows or native
    synthetic_only=bool(scope_rows) and all(r['stage'].startswith(('stress','synthetic')) for r in scope_rows)
    lines=["# 로봇 춤 생성 비교 결과","","이 실험은 BPM 검출기 비교가 아니라 기존 finite/infinite와 공개 Reachy primitive scheduler의 오프라인 궤적 비교입니다.",
           "입력: 알려진 synthetic 120 BPM beat grid. 실제 음악의 자연스러움/음원 검출 정확도 실험이 아닙니다." if synthetic_only else "음악: 동일 Billie Jean 파일의 여러 구간. Beat는 정답 annotation이 없는 shared estimated beat reference입니다.",
           "대상은 동일 Misty 머리 pitch/roll/yaw 3관절의 commanded trajectory이며, 실제 장치·물리 시뮬레이터 상태가 아닙니다.",
           "Reachy는 Apache-2.0 공개 엔지니어링 primitive 구현이며 사전학습 신경망이 아닙니다. 원형과 변환 결과는 분리했습니다.",
           "고정 변환: 명칭에 따른 Euler pitch/roll/yaw, scale=1, 첫 Euler offset을 빼 neutral에 정렬. 위치·안테나는 제외, smoothing/clipping 없음.",
           "실행 전 고정한 설정은 configs/benchmark.json, 외부 소스 revision/license는 source_manifest.json에 있습니다.","",
           "## Table A — 같은 로봇 / 같은 관절","","|stage|method|seed|duration s|status|recurrence|activity stationary|beat error ms (allowed offbeat grid)|position violation rate|",
           "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['stage']}|{r['method']}|{r['seed']}|{float(r['duration_seconds']):.2f}|{r['status']}|{r.get('recurrence_rate','N/A')}|{r.get('stationary_fraction','N/A')}|{r.get('permitted_grid_median_error_ms','N/A')}|{r.get('position_violation_rate','N/A')}|")
    lines += ["","## 처리 시간","","아래 시간은 동일 12초 출력, batch1, CPU 1 thread, 방법 순차 실행, 2 warm-up 후 5회 측정입니다. 5회로 안정적인 P95를 주장하지 않습니다.",
              "기존 finite의 그래프 준비/모듈 로드는 initialization에 따로 기록됩니다. 입력은 캐시된 beat이며 raw-audio end-to-end는 N/A입니다.",
              "|method|n|init median s|generation+adaptation median s|min s|max s|RTF|","|---|---:|---:|---:|---:|---:|---:|"]
    for r in timing:lines.append(f"|{r['method']}|{r['repetitions']}|{r['initialization_median_seconds']:.6f}|{r['median_seconds']:.6f}|{r['min_seconds']:.6f}|{r['max_seconds']:.6f}|{r['rtf_median']:.6f}|")
    if not timing:
        lines[lines.index("아래 시간은 동일 12초 출력, batch1, CPU 1 thread, 방법 순차 실행, 2 warm-up 후 5회 측정입니다. 5회로 안정적인 P95를 주장하지 않습니다.")]="이 보조 실행에서는 반복 latency를 측정하지 않았습니다. quality 행의 wall time을 독립 5회 timing 결과로 해석하지 않습니다."
    lines += ["","초기화 열은 process/library warm-up 이후 per-run planner 생성·캐시 조회입니다. Reachy의 값은 이미 로드한 순수 모듈의 조회이며 cold source import/검증 시간이 아닙니다.",
              "품질 행의 generation_seconds는 다른 작업의 부하가 포함될 수 있습니다. 속도 비교에는 timing_conditions.json의 idle 확인 뒤 수집한 latency_raw/summary만 사용합니다.",
              "chunk_deadline_misses는 offline compute time과 이전 구절 길이의 비교 proxy이며 실제 DAC/hardware deadline 측정이 아닙니다."]
    lines += ["","## 해석 범위와 실패","","- 관절 범위 normalization을 사용하며 실행 자체의 분산으로 나누지 않았습니다. 4-beat window, nonadjacent separation≥2 windows, RMS<0.04가 recurrence입니다.",
              "- 속도 에너지 peak detector는 전 방법에 동일하며 의미 있는 excursion/이벤트 수가 부족하면 alignment N/A입니다. 선언한 pose-accent와 일반 에너지 peak는 서로 다른 event 정의입니다.",
              "- q/dq/ddq 위반은 동일 0.1초 grid의 원본 commanded trajectory에서 측정한 kinematic 지표입니다. 연속시간 보증·균형·충돌·실제 추종을 뜻하지 않습니다.",
              "- Reachy 전환은 원래 cyclic preview의 hard switch이며 global phase를 초기화하지 않습니다. 전환 불연속/위반을 보정해 숨기지 않았습니다.",
              "- Reachy의 빠른 raw 출력이 곧 유효한 Misty 제어 출력이라는 뜻은 아닙니다. 이 고정 projection의 제한 위반과 큰 amplitude를 다양성·RTF와 함께 봐야 합니다.",
              "- deterministic Reachy는 각 음악 구간을 한 번만 품질 평가했습니다. timing 반복은 새 춤 표본이 아닙니다.",
              "- 여러 crop은 모두 같은 곡입니다. 이 소규모 결과로 보편적 우열이나 자연스러움을 결론 내리지 않습니다.","",
              "## Table B — RoboPerform native G1","","RoboPerform의 native 상태/제공 latent/원본 G1 rollout은 ../5_Benchmark2/results/native_reproduction.csv 및 README를 보세요.",
              "G1은 이 표의 Misty 점수와 합쳐 순위를 만들지 않았으며, custom-audio가 확인되지 않으면 Billie Jean 조건 생성으로 표시하지 않습니다.","",
              "## 재실행","","```powershell","python setup_external.py","python run_benchmark.py --quick --robot misty-head","python run_benchmark.py --summarize","```","",
              "## 발표용 영어 bullet (측정 범위 한정)",
              f"- Generated commanded head trajectories for {sum(r['status']=='generated' for r in rows)} successful same-robot runs using shared estimated beats.",
              "- Reported raw kinematic violations and recurrence without post-hoc amplitude, phase, or octave tuning.",
              "- Separated initialization from five-repeat cached-beat motion generation; native G1 reproduction is a different task.",""]
    if synthetic_only:
        lines += ['## 전체 파형 주기 보조 검증','',
                  '152 beats는 primitive 선택기의 순환입니다. 고정 원형 소스의 head 전체 파형은 760 beats, 모든 native 채널은 2280 beats 주기를 가집니다.',
                  'Misty-head 보조 실행은 1520 beats/760.1초, native 전체 보조 실행은 4560 beats/2280.1초를 생성합니다. 자연 음악의 제한된 길이를 반복 재생하지 않았습니다.',
                  'lag 760 beat 표시는 원본 함수에서 유도한 고정 주기이며, 결과를 좋게 만드는 사후 phase/octave fitting이 아닙니다. 무한 비반복 보장은 하지 않습니다.','']
    if native:
        lines += ['## Reachy native 원형 기록','','|method|duration seconds|status|input|','|---|---:|---|---|']
        for r in native:lines.append(f"|{r['method_label']}|{r['duration_seconds']}|{r['status']}|{r['input']}|")
    g1_path=REPO/'5_Benchmark2/results/run_status.json'
    if g1_path.exists():
        g1=json.loads(g1_path.read_text(encoding='utf-8'));rollout=g1.get('rollout',{})
        lines += ['','## Table B — native G1 관측 (Table A와 순위 혼합 없음)','',
                  '|method|input|rollout_ok|duration s|policy seconds|simulation seconds|custom audio|audio alignment|',
                  '|---|---|---|---:|---:|---:|---|---|',
                  f"|RoboPerform device-only CPU port|released music/dance latent|{g1.get('rollout_ok')}|{rollout.get('duration_seconds')}|{rollout.get('policy_generation_seconds')}|{rollout.get('simulation_processing_seconds')}|{g1.get('custom_audio_ok')}|N/A|",
                  '', '[student strict load·CPU port·실제 state feedback·실패 기록·100-step 원시 시간](../../5_Benchmark2/results/report.md)을 참고하세요.',
                  '이 표의 입력은 raw Billie Jean이 아니며, 2초 G1 정책과 12초 Misty-head generator의 시간을 서로 나눠 우열을 만들지 않습니다.','']
    (output/"report.md").write_text("\n".join(lines),encoding="utf-8")
    return timing


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true");parser.add_argument("--robot",choices=["misty-head"],default="misty-head")
    parser.add_argument("--audio");parser.add_argument("--output",default=str(ROOT/"results"));parser.add_argument("--method",choices=["all",*METHODS,"reachy_native"],default="all")
    parser.add_argument("--stage",choices=["all","stage1","stage2","long","stress","full-cycle"],default="all")
    parser.add_argument("--summarize",action="store_true");parser.add_argument("--skip-timing",action="store_true");parser.add_argument("--skip-visuals",action="store_true")
    parser.add_argument("--timing-only",action="store_true",help="Keep saved quality results and repeat the serial latency workload")
    parser.add_argument("--wait-idle-seconds",type=float,default=900,help="Wait for other 3/5 inference jobs before warm-up/timing; never terminate them")
    args=parser.parse_args(argv);output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if args.summarize:
        summarize(output)
        existing=list(csv.DictReader((output/'same_robot_metrics.csv').open(encoding='utf-8'))) if (output/'same_robot_metrics.csv').exists() else []
        if existing and all(r['stage']=='synthetic_full_cycle' for r in existing):
            from visualize import make_full_cycle_plot
            make_full_cycle_plot(output)
        elif existing:
            from visualize import make_plots
            make_plots(output)
        return
    cfg=json.loads((ROOT/"configs/benchmark.json").read_text(encoding="utf-8"));profile=robot_profile();source=find_audio(args.audio)
    write_json(output/"environment.json",environment());write_json(output/"config_snapshot.json",{**cfg,"profile":profile.to_dict(),"arguments":vars(args)})
    began=time.perf_counter();analysis=analyze_audio(source,ROOT/"data/cache");analysis_seconds=time.perf_counter()-began
    external=load_upstream();cycle_beats=external["preview_cycle_beats"]
    manifest={"audio":{"path_relative_to_repository":str(source.relative_to(REPO)) if source.is_relative_to(REPO) else str(source),"sha256":file_hash(source),"duration_seconds":analysis.duration},
              "input_conditions":{"real_music_stages":["stage1","stage2","long"],"synthetic_stages":["stress90","stress120","stress180","synthetic_full_cycle"],
                                  "synthetic_rows_do_not_use_audio_file":True,"active_request_stage":args.stage},
              "beat_reference":{"label":"shared estimated beat reference","bpm":analysis.bpm,"beats_seconds":analysis.beats.tolist(),"accent_strengths":analysis.accent.tolist(),"manual_ground_truth":False,
                                "analysis_or_cache_load_seconds":analysis_seconds,"cache_path":"4_Benchmark1/data/cache (gitignored)","method":analysis.settings["method"]},
              "reachy":external,"our_sources":[{"path":p,"sha256":file_hash(REPO/p),"sha256_lf":source_text_hash(REPO/p),"text_hash_scope":"UTF-8 source with CRLF normalized to LF; algorithm content unchanged"} for p in ("1_finite_version/rita_finite/planner.py","1_finite_version/rita_finite/library.json","2_infinite_version/rita_infinite/planner.py")],
              "modifications":"Thin adapters only; no existing planner algorithm changes. Pure Reachy namespace loader bypasses optional hardware wrapper only. Fixed Euler projection/initial anchor for Misty."}
    write_json(output/"source_manifest.json",manifest)
    methods=list(METHODS) if args.method=="all" else [args.method]
    rows=list(csv.DictReader((output/'same_robot_metrics.csv').open(encoding='utf-8'))) if args.timing_only and (output/'same_robot_metrics.csv').exists() else []
    native_rows=[];status={};lags={};seed_data=defaultdict(list)
    stages=[] if args.timing_only else (["stage1","stage2","long","stress"] if args.stage=="all" else [args.stage])
    specs=[]
    if "stage1" in stages:specs += [("stage1",0,12.,0,analysis)]
    if "stage2" in stages:specs += [("stage2",b,24.,s,analysis) for b in cfg["stage2"]["start_beat_indices"] for s in cfg["stage2"]["seeds"]]
    if "long" in stages:
        long_seconds=(math.ceil(max(120.,analysis.beat_time(2*cycle_beats)-analysis.beat_time(0))/profile.command_period)+1)*profile.command_period
        specs += [("long",0,long_seconds,0,analysis)]
    if "stress" in stages:
        specs += [(f"stress{bpm}",0,12.,0,synthetic_analysis(bpm,duration=180,phase=0)) for bpm in (90,120,180)]
    if "full-cycle" in stages:
        full=cfg['full_cycle'];period=full['native_all_channel_period_beats'] if args.method=='reachy_native' else full['head_waveform_period_beats']
        seconds=period*full['cycles']*60/full['bpm']+profile.command_period
        specs += [('synthetic_full_cycle',0,seconds,0,synthetic_analysis(full['bpm'],duration=seconds+90,phase=0))]
    for stage,beat,seconds,seed,a in specs:
        for method in methods:
            if method.startswith("reachy") and seed!=0:continue
            row,result,details=quality_run(method,a,profile,cfg,stage,beat,seconds,seed,output)
            if stage.startswith("stress") or stage.startswith('synthetic'):
                row["input"]="known synthetic beat timestamps (planner timing stress, not music naturalness)"
                row["music_reference"]="known synthetic beat grid"
            (native_rows if method=="reachy_native" else rows).append(row)
            status[row["run_id"]]={"status":row["status"],"error":row.get("error"),"hardware_executed":False}
            if result is not None and "q" in result:
                lags[row["run_id"]]=details["lag_similarity"]
                if stage=="stage2":seed_data[(method,beat)].append((seed,result["q"]))
            # Save completed methods immediately, independent of later failures.
            write_csv(output/"same_robot_metrics.csv",rows);write_csv(output/"native_reproduction.csv",native_rows)
            write_json(output/"run_status.json",status);write_json(output/"lag_similarity.json",lags)
    if args.method=="all" and "stage1" in stages:
        row,_,_=quality_run("reachy_native",analysis,profile,cfg,"stage1",0,12.,0,output);native_rows.append(row)
        write_csv(output/"native_reproduction.csv",native_rows)
        status[row["run_id"]]={"status":row["status"],"error":row.get("error"),"hardware_executed":False}
        write_json(output/"run_status.json",status)
    distances=[]
    for (method,beat),trajectories in seed_data.items():
        distances += [{"method":method,"start_beat":beat,**r} for r in seed_distances(trajectories,profile)]
    write_csv(output/"seed_distances.csv",distances)
    timing=[]
    if not args.skip_timing:
        write_json(output/'timing_conditions.json',wait_for_timing_window(args.wait_idle_seconds))
        for method in [m for m in methods if m in METHODS]:
            for rep in range(-cfg["latency"]["warmups"],cfg["latency"]["repetitions"]):
                print(f"TIME {method} repeat={rep}",flush=True)
                row={"method":method,"kind":"warmup" if rep<0 else "measured","repeat":rep,"duration_seconds":12.,"status":"failed"}
                try:
                    result=generate_method(method,analysis,profile,0,12.,0)
                    row.update({k:result[k] for k in ("init_seconds","generation_seconds","adaptation_seconds","limit_handling_seconds","total_generation_seconds")})
                    row.update(status="generated",rtf=result["total_generation_seconds"]/12.)
                except Exception as exc:row["error"]=f"{type(exc).__name__}: {exc}"
                timing.append(row);write_csv(output/"latency_raw.csv",timing)
    if not timing:
        write_csv(output/"latency_raw.csv",[{"method":m,"kind":"not_measured","repeat":0,"status":"not_measured","total_generation_seconds":None} for m in METHODS])
    summarize(output)
    if not args.skip_visuals:
        from visualize import make_plots,render_video
        if args.stage=='full-cycle' and args.method!='reachy_native':
            from visualize import make_full_cycle_plot
            make_full_cycle_plot(output)
        elif args.stage!='full-cycle':make_plots(output)
        if "stage1" in stages and args.method=="all":render_video(output,source,analysis.beat_time(0),12.)
    print(json.dumps({"output":str(output),"same_robot_runs":len(rows),"native_runs":len(native_rows),"generated":sum(r["status"]=="generated" for r in rows),"failed":sum(r["status"]=="failed" for r in rows)},indent=2))


if __name__=="__main__":main()
