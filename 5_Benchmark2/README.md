# Benchmark 2 — RoboPerform native G1 재현

이 폴더는 **춤 생성 논문 방법**인 [RoboPerform](https://github.com/gentlefress/RoboPerform)의
공개 student와 제공 latent를 검증하고, 원래 G1 모델에서 closed-loop MuJoCo 실행을
시도하는 독립 실험입니다. Misty/Kinova와 동일 로봇 비교표에 넣지 않습니다.
실제 로봇 명령은 보내지 않습니다.

고정 버전은 source `e6e2a0e4cdb20f081c1a3d46d648d7a202816779`,
Hugging Face dataset `SanQing1/RoboPerform` revision
`3fdfcd75480450ec2f8f884723b5724cb3a8051b`입니다. 소스, 가중치, 데이터,
SDK, 가상환경은 Git에서 제외합니다. clone한 컴퓨터에서 아래 명령으로 다시 받습니다.

## 다운로드·설치

Python 3.11, Git, FFmpeg가 필요합니다. Windows PowerShell에서 이 폴더로 이동한 뒤:

```powershell
.\setup_benchmark.ps1
.\.venv\Scripts\python.exe run_benchmark.py --inspect --headless
.\.venv\Scripts\python.exe run_benchmark.py --native --headless --device cpu --allow-cpu-device-port --seconds 2
.\.venv\Scripts\python.exe write_report.py
```

`prepare.py`는 HF **dataset**의 고정 revision URL을 사용합니다. 먼저 최신 pinned-host
파일 메타데이터와 예상 크기/해시를 대조하고, 다운로드 후 크기와 SHA-256을 모두
검증합니다. HTTP 오류 페이지, Git LFS 포인터 또는 다른 checkpoint를 성공으로
처리하지 않습니다. 전처리/다운로드는 policy latency 밖에서 수행합니다.

```powershell
# 전체 환경 없이 소스와 원본 student만 먼저 확인
python prepare.py --assets student --refresh-metadata
# 실제 공개 latent demo에 필요한 최소 집합
python prepare.py --assets deployment --refresh-metadata
```

| 필수 파일 | 정확한 다운로드 bytes | 역할 |
|---|---:|---|
| student_80500.pt | 563,075,994 | 공개 audio-conditioned student checkpoint |
| music.pt | 306,873,850 | 제공된 music latent 사전 |
| dance.pt | 129,106 | 제공된 motion latent |
| Roboperform_Data.pkl | 578,989,934 | motion ID와 latent 이름을 대응시키는 원본 데이터 |
| 합계 | **1,449,068,884** | 약 1.45 GB, 프레임워크 설치 용량 별도 |

teacher와 TensorRT engine은 이 PyTorch 경로에서 받지 않습니다. teacher 결과를
student처럼 표시하지 않습니다. `results/environment.json`의 환경 설치 bytes는
가중치 bytes·파라미터 수·최대 실행 RAM과 별개입니다.

Linux의 NVIDIA GPU 컴퓨터에서는 별도 venv에 [공식 PyTorch 설치표](https://pytorch.org/get-started/previous-versions/)
중 Torch 2.6.0에 맞는 CUDA wheel을 설치하고 `requirements.txt`를 설치합니다.
CUDA 12.4 wheel 예시는 다음과 같습니다. 드라이버 호환성을 해당 컴퓨터에서 먼저 확인합니다.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prepare.py --assets deployment --refresh-metadata
.venv/bin/python run_benchmark.py --native --headless --device cuda --seconds 9.98
```

## 입력과 재현 범위

공개 `deploy/get_latent.py`는 `motion_new_name`과 time index로 `music.pt`를
조회합니다. 현재 고정 소스에서 임의 wav/mp3를 같은 music latent로 변환하는
공개 encoder+adapter 가중치·정규화·temporal alignment 경로는 확인되지 않았습니다.
따라서 이 실험의 이름은 다음과 같습니다.

**Released latent-conditioned G1 demo; custom-audio inference unavailable.**

Billie Jean을 배경음으로 붙이지 않습니다. 영상은 무음이며 음악 alignment와
raw-audio-to-motion end-to-end 시간은 N/A입니다. 제공 latent의 실제 길이와
공식 배포 코드의 최대 499 control step 중 짧은 쪽만 실행합니다. 0.02초 주기에서
최대 약 9.98초이며, 짧은 latent를 반복해 긴 새 춤으로 표시하지 않습니다.

`--inspect`는 strict checkpoint load 검사일 뿐 춤 생성 성공이 아닙니다.
`run_status.json`은 `artifact_verified`, `model_loaded_strict`, `forward_ok`,
`rollout_ok`, `custom_audio_ok`, `hardware_executed`를 분리합니다.
학생 가중치와 observation normalizer는 **strict=True**로 로드합니다.
누락/초과 key나 shape mismatch가 있으면 중지합니다. 무작위 가중치나 다른
embedding으로 바꾸지 않습니다.

## CPU device-only port

공식 DDIM 코드에는 timestep tensor의 무조건적인 `.cuda()` 호출이 있습니다.
NVIDIA GPU가 없는 환경에서 원형 CPU 실행은 이 지점에서 실패할 수 있습니다.
`--allow-cpu-device-port`는 별도의 ignored 소스 복사본에 다음 변경만 적용합니다.

- 초기 noise의 device를 함수 인수에 따르게 함.
- timestep tensor의 device를 현재 sampling image와 같게 함.
- CUDA에 저장된 music/dance tensor를 `map_location='cpu'`로 같은 값의 CPU storage에 로드.

architecture, 가중치, normalizer, 1400차원 observation 순서, 23개 action,
확산 sampling step 수, CFG 인수, PD gain, control 주기는 바꾸지 않습니다.
차이는 `cpu_device_port.patch`와 원본/복사본 SHA-256에 저장합니다.
원본 pinned checkout은 수정하지 않습니다. 이 결과에는 device-only CPU port라는
표시가 포함됩니다. CPU/GPU latency는 섞지 않습니다.

한글 Windows 사용자 경로를 MuJoCo 3.3 파일 loader가 거부하는 경우를 피하려고,
같은 G1 XML과 mesh bytes를 virtual file system으로 전달합니다. mesh 검색 경로만
재연결하며 joint/geometry/mass/contact/PD 설정은 바꾸지 않습니다.

## 출력과 해석

기본 `results/` 아래에 다음을 저장합니다.

- `source_manifest.json`: URL, 고정 commit, HF 파일 크기/해시, 라이선스 확인 범위.
- `environment.json`: OS/Python/패키지, CPU/GPU, 환경 설치 용량.
- `run_status.json`, `last_attempt_traceback.txt`: 성공 단계와 실제 실패 이유.
- `native_reproduction.csv`: native Table B. Misty/Kinova Table A와 분리.
- `latency_raw.csv`: 각 유효 state-feedback policy step의 실제 연산 시간.
- `trajectories/g1_rollout.npz`, `.csv`: 실제 closed-loop MuJoCo state와 target/action.
- `g1_state.png`, `videos/g1_released_latent.mp4`: 상태 그림과 무음 G1 영상.

가중치·raw 궤적·영상은 Git에 넣지 않습니다. 작은 JSON/CSV/소스 patch와 결과
설명은 버전 관리할 수 있습니다. simulation 시간, policy inference 시간,
initialization/weight loading, visualization 시간을 분리합니다. 실제 sleep과
viewer는 사용하지 않습니다. RTF<1이어도 streaming 성공을 뜻하지 않습니다.
낙상은 base height<0.35m라는 진단 proxy로만 기록하며, 실제 로봇 낙상률이나
balance/collision 인증으로 해석하지 않습니다.

라이선스: upstream README는 BSD-3-Clause라고 적지만 이 고정 revision에는 root
LICENSE 파일이 없고 HF dataset license metadata도 비어 있습니다. 링크와 확인
사실을 manifest에 남겼습니다. 이 저장소는 원본 코드·가중치·dataset을 재배포하지
않고 공식 위치에서 내려받는 wrapper와 재현 기록을 제공합니다.

## Common rollout containers

Native simulation also streams a versioned container to
`<results>/containers/<run_id>/`, using the same schema as Misty and Kinova
rollouts. The `simulation` channel stores all 29 G1 joint positions/velocities
in the actual model order. The `reference` channel stores the PD joint target
for the following control interval, with the released 29-dimensional action
and its ordering in metadata. Floating-base position, quaternion, velocity,
base height and control index remain available as simulation metadata.

The profile uses the pinned MuJoCo position ranges and pinned URDF velocity
limits. The model does not provide acceleration limits; acceleration
feasibility is explicitly N/A. Policy inference events retain the original
forward timing, with logger and metric work outside the measured forward
region. Total closed-loop wall time includes logging overhead. Exceptions
finalize the container as failed and preserve observations already received.

Time is MuJoCo simulation time, even though the shared field is called
`music_time`. No AudioAnalysis is supplied for released latent demos, so audio
synchronization metrics remain null. G1 simulation and physical Misty/Kinova
measurements retain distinct sources and embodiment metadata.

From the repository root:

```powershell
python 6_container/read_runs.py 5_Benchmark2/results/containers --csv result/g1_rollouts.csv
python 6_container/read_runs.py 5_Benchmark2/results/containers --recompute
```

See [6_container](../6_container/README.md) for common readers and extensions.
