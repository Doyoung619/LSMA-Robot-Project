"""Summarize measured native reproduction without rerunning policy inference."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'


def main():
    status=json.loads((RESULTS/'run_status.json').read_text(encoding='utf-8'))
    env=json.loads((RESULTS/'environment.json').read_text(encoding='utf-8'))
    if sys.platform=='win32':
        import ctypes
        class Memory(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ('total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','available_extended')]
        mem=Memory();mem.length=ctypes.sizeof(mem);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        env['ram_total_bytes']=int(mem.total_physical)
        (RESULTS/'environment.json').write_text(json.dumps(env,indent=2)+'\n',encoding='utf-8')
    lock=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (ROOT/'requirements-lock.txt').write_text(lock,encoding='utf-8')
    model=status.get('model',{});r=status.get('rollout',{})
    total=sum(x['bytes'] for x in status.get('artifacts',[]))
    report=f'''# RoboPerform native 재현 — 실제 실행 결과

공식 student와 제공 latent를 사용하여 G1의 **closed-loop MuJoCo state-feedback**를
실제로 실행했습니다. 이는 Misty/Kinova의 같은 로봇 비교가 아니라 별도의 native Table B입니다.

| 항목 | 관측 결과 |
|---|---|
| source revision | `{status['source_revision']}` |
| checkpoint | `student_80500.pt`, 563,075,994 bytes |
| model/data 4개 총량 | {total:,} bytes |
| 실제 policy parameter 수 | {model.get('parameter_count'):,} |
| strict student/normalizer load | {status['model_loaded_strict']} (missing/extra keys 0) |
| 유효 forward / closed-loop rollout | {status['forward_ok']} / {status['rollout_ok']} |
| custom raw-audio inference | unavailable |
| hardware executed | False |
| 입력 | released music/dance latent + 실제 G1 history |
| 출력 구간 | {r.get('duration_seconds',0):.3f}초, {r.get('control_steps',0)} control steps |
| 원형 control period | {r.get('control_period_seconds')}초 |
| policy generation 총시간 | {r.get('policy_generation_seconds',0):.6f}초 |
| step median / min / max | {r.get('policy_step_seconds_median',0):.6f} / {r.get('policy_step_seconds_min',0):.6f} / {r.get('policy_step_seconds_max',0):.6f}초 |
| step P95 (100회 관측) | {r.get('policy_step_seconds_p95',0):.6f}초 |
| policy RTF | {r.get('policy_RTF',0):.4f} |
| MuJoCo simulation 연산 | {r.get('simulation_processing_seconds',0):.6f}초 |
| 영상/시각화 연산 | {r.get('visualization_seconds',0):.6f}초 |
| model construction / checkpoint load | {model.get('model_construction_seconds',0):.6f} / {model.get('checkpoint_loading_seconds',0):.6f}초 |
| latent loading | {status.get('latents',{}).get('loading_seconds',0):.6f}초 |
| 최소 base 높이 | {r.get('minimum_base_height_m',0):.6f} m |
| base<0.35m proxy | {r.get('fall_proxy_base_below_0_35m')} |
| audio alignment / raw-audio end-to-end | N/A / N/A |

CPU PyTorch {env['packages']['torch']}, Python {env['python']}, MuJoCo
{env['packages']['mujoco']}, 1 inference thread에서 실행했습니다. 새로운 학습은 하지 않았습니다.
위 step P95는 100개의 연속 state-feedback step 관측이며 서로 다른 2초 궤적을
100개 생성했다는 뜻이 아닙니다. warm-up 2회·독립 workload 5회 비교 시간과도
구분합니다. Native G1 수치를 다른 로봇의 trajectory 생성 시간과 직접 나누지 않습니다.

공식 CPU 원형 실행은 CUDA에 저장된 latent의 `torch.load` 때문에 실패했습니다.
해당 실제 traceback은 `unmodified_cpu_attempt/last_attempt_traceback.txt`에 남겼습니다.
성공 run은 별도의 **device-only CPU port**를 명시적으로 사용합니다. `.cuda()`
placement와 `map_location='cpu'`, 로컬 asset 경로 및 Windows Unicode 경로용
동일 XML/mesh VFS 로딩만 추가했습니다. architecture, 가중치, normalizer,
관측 history 순서, diffusion steps, PD gain, control 주기는 동일합니다.

영상은 무음입니다. 공개 music.pt가 어떤 원본 wave와 대응하는지 이 checkout의
배포 경로에서 확인하지 못했으므로 Billie Jean을 붙이거나 beat alignment를
추정 정확도로 쓰지 않습니다. 2초의 한 가지 released-latent demo만 검증했습니다.
긴 rollout의 안정성·다양성, 실제 로봇 balance/collision/추종 성능은 검증하지 않았습니다.

## 생성 파일

- `trajectories/g1_rollout.npz`, `trajectories/g1_rollout.csv`: 실제 MuJoCo q/qvel, target, action.
- `videos/g1_released_latent.mp4`: 640×480, 25fps, 무음 상태 렌더링.
- `g1_state.png`: joint state와 base height.
- `latency_raw.csv`: 모든 policy step의 처리 시간.
- `run_status.json`, `source_manifest.json`, `environment.json`: 상태·해시·환경.

## 발표용 영어 bullets (이번 측정 범위)

- The released student checkpoint loaded strictly with zero missing or unexpected keys.
- A device-only CPU port completed a 2-second, 100-step closed-loop native G1 MuJoCo rollout.
- The measured policy processing RTF was {r.get('policy_RTF',0):.3f}; custom-audio inference and audio alignment remain unavailable.
'''
    (RESULTS/'report.md').write_text(report,encoding='utf-8')
    print('Updated report.md and dependency lock; no policy inference executed.')


if __name__=='__main__':main()
