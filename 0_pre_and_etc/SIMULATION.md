# Rita virtual robot sandbox

> ?? ?? ? ? ??? ??? `experiment/0_pre_and_etc`?? ?????. ??? ??? `../result/0_pre-result/`? ???, ?? ?? ??? `../result/0_pre-result/Play_Dance_Video.cmd`? ?????. ? ??? ?? ??? ? ?? ??? ?????.

MuJoCo에서 네 종류의 로봇이 유한한 목표 자세들을 오가는 독립 실험 환경입니다.
`Play_Sim.cmd`를 더블 클릭하고 1~4 중 하나를 선택하면 120초 동안 실행합니다.
SPACE로 일시정지/재개하고, 창을 닫으면 종료합니다. 마우스로 카메라를 회전/확대할 수 있습니다.
현재 더블 클릭 실행은 큰 동작 16개와 117.025 BPM을 사용하는 `expressive` 프로필입니다.

| 선택 | 모델 | 관절 | 출처/정확도 |
|---|---|---:|---|
| 1 / `kinova` | Gen3 6-DOF spherical wrist with vision | 6 | 기존 프로젝트의 Kinova 공식 URDF + 해당 6축용 공식 STL |
| 2 / `g1` | Unitree G1 | 29 | MuJoCo Menagerie, 고정 베이스로 변환 |
| 3 / `misty` | Misty 단순화 모델 | 5 | 로컬 기본 도형, 머리 yaw/roll/pitch와 양팔; 크기·외형은 근사 |
| 4 / `go2` | Unitree Go2 | 12 | MuJoCo Menagerie, 고정 베이스로 변환 |

## 설치 및 실행

현재 PC에는 `.venv-sim`에 설치되어 있습니다. 기존 `.venv`와 실물 제어 코드는 별도로 유지합니다.
기존 Anaconda의 NumPy를 공유하고 MuJoCo 코어·GLFW·PyOpenGL만 새 환경에 설치합니다.
다른 컴퓨터에서는 Python 3.11 이상으로 다음 명령을 실행합니다.

```powershell
powershell -ExecutionPolicy Bypass -File .\Setup_Sim.ps1
```

```powershell
# 모델 준비 상태
.\.venv-sim\Scripts\python.exe sim.py list

# 창을 열어서 한 로봇 실행
.\.venv-sim\Scripts\python.exe sim.py dance --robot kinova --bpm 90 --seconds 120
.\.venv-sim\Scripts\python.exe sim.py dance --robot g1 --bpm 90 --seconds 120
.\.venv-sim\Scripts\python.exe sim.py dance --robot misty --bpm 90 --seconds 120
.\.venv-sim\Scripts\python.exe sim.py dance --robot go2 --bpm 90 --seconds 120

# 창 없이 네 모델 모두 30초 궤적 검증
.\.venv-sim\Scripts\python.exe sim.py check --seconds 30

# 렌더링 PNG와 4분할 미리보기 생성
.\.venv-sim\Scripts\python.exe sim.py snapshot --robot all --pose 3

# 40~240 BPM에서 그래프와 capacity 계산
.\.venv-sim\Scripts\python.exe sim.py sweep

# 동작/그래프 테스트
.\.venv-sim\Scripts\python.exe -m unittest discover -s sim_tests -v
```

## 이번 환경의 동작 범위

- 각 모델에 서로 다른 목표 관절 자세 8개를 넣었습니다. `sim_poses/<robot>.json`의 `poses[].degrees`를 수정하면 됩니다. 각 행은 `joint_names` 순서이며 단위는 도입니다. 자세 추가/삭제도 가능합니다.
- `speed_deg_s`, `acceleration_deg_s2`는 비교 실험용 사용자 설정입니다. 현재 기본값은 모든 관절에 120 deg/s, 600 deg/s²이며 실제 제품의 성능 수치가 아닙니다.
- 두 관절 자세를 연결하는 quintic smoothstep 경로에 필요한 시간을 계산하고, 한 박자(60/BPM) 안에 가능한 전환만 그래프에 넣습니다. 시작/도착 속도와 가속도는 0입니다.
- 전환시간은 이 경로의 속도·가속도 제한을 만족시키는 충분 시간입니다. 여러 경로에 대한 전역 최소시간 최적화 결과가 아닙니다.
- 자기 자신에 머무는 self-loop는 춤 전환으로 세지 않습니다. 다음 자세는 덜 사용한 이동과 덜 방문한 자세에 가중치를 줘 선택합니다. `--seed`로 재현할 수 있습니다. 모든 자세의 방문을 강제하는 보장은 없습니다.
- 그래프의 `log(max(1, spectral_radius))`와 실제 방문 자세/간선 수를 따로 기록합니다. 빠른 BPM 때문에 이동 가능한 cycle이 없어지면 명시적으로 중단합니다.
- 베이스 위치를 고정하고 매 프레임 관절 위치를 지정하는 **기구학 모드**입니다. 접촉은 껐으며 충돌·토크·균형·보행 제어를 검사하지 않습니다. MuJoCo의 `mj_forward`로 모델 자세와 렌더링을 갱신합니다. 실제 동역학 추종이나 자립 춤의 검증 결과로 해석하지 않습니다.
- 각 로봇의 자세 라이브러리는 환경 확인용으로 따로 정한 예시입니다. 같은 task-space vocabulary를 유지하는 논문의 DOF 비교 실험이나 공정한 로봇 간 성능 비교는 아직 구성하지 않았습니다.
- 음악 재생 대신 `--bpm`으로 박자 시계를 지정합니다. 기존 Billie Jean 데모와 로봇 네트워크에 연결하지 않습니다.

## 파일, 용량, 출처

- `sim.py`, `rita_sim/`: 시뮬 전용 실행·모델 준비·그래프 코드.
- `sim_poses/`: 사용자가 수정할 목표 자세와 실험용 속도 설정.
- `sim_models/`: 필요한 XML·STL/OBJ·라이선스만 다운로드. 모델 전체 저장소나 학습 가중치를 받지 않습니다.
- `sim_models/manifest.json`: 원본 URL, 고정 commit, 다운로드 크기, SHA-256.
- `sim_output/`: 스크린샷, 실행 보고서, BPM sweep 결과. 필요할 때 다시 만들 수 있습니다.
- 새 환경 설치 직후 실측: `.venv-sim` 약 62 MiB, 모델 약 51 MiB. 기존 시스템 NumPy의 용량은 이 추가 설치량에 포함되지 않습니다.

Menagerie revision: `370c43965323c0944b433b202aac5058b5f907a9`.
Kinova revision: `c3280d8a6a6b5d96590d51538e3f265710a6c77d`.

- [MuJoCo core 및 경량 렌더링 설치](https://mujoco.readthedocs.io/en/stable/python.html#installation)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Kinova 공식 6축 모델](https://github.com/Kinovarobotics/ros_kortex/tree/noetic-devel/kortex_description/arms/gen3/6dof)
- [Misty 머리·팔 좌표 및 API 범위](https://docs.mistyrobotics.com/misty-ii/web-api/api-reference/)

G1/Go2의 원본 라이선스와 Kinova 라이선스는 각 모델 디렉터리에 보관합니다.
Misty는 공식 모델이 아닌 이 프로젝트의 단순화 모델이며, 관절 범위는 위 API의 degree 표를 따릅니다.

## Billie Jean 동기화 영상

`sim_output/video/Rita_Billie_Jean_4_Robots.mp4`는 원본 음악이 들어간 60초, 1920×1080, 30fps 영상입니다.
네 로봇의 움직임·현재 전환·Dance Graph·방문 다양성을 한 화면에서 볼 수 있습니다.
기존 원본 MP3를 다시 분석한 결과는 `sim_output/video/billie_jean_analysis.json`에 보관합니다.
검출된 고정 템포는 117.0254648 BPM, 첫 박자 위상은 원본 오디오 기준 0.3359264초입니다.
이는 오디오에서 추정한 beat grid이며 사람이 지정한 downbeat 라벨은 아닙니다.

다음 명령으로 같은 영상을 다시 만들 수 있습니다. 영상용 Pillow는 현재 PC에 이미 있습니다.
다른 PC에서 필요하면 `requirements-sim-video.txt`를 설치하세요. FFmpeg/FFprobe도 PATH에 있어야 합니다.

```powershell
.\.venv-sim\Scripts\python.exe render_dance_video.py --seconds 60
# 다른 구간도 원본 음악의 같은 beat grid에 맞춰 렌더링
.\.venv-sim\Scripts\python.exe render_dance_video.py --start 45 --seconds 30 --output sim_output/video/clip_45s.mp4
```

각 프레임의 원본 음악 시각에서 박자 위상을 빼고 동작을 평가합니다. 박자마다 도착 자세를 맞추며,
30fps에 따른 가까운 프레임의 시간 양자화 오차는 최대 약 16.7ms입니다.
임시 프레임을 수천 장 저장하지 않고 FFmpeg로 바로 보내 H.264/AAC MP4를 생성합니다.
영상과 같은 이름의 JSON에는 원본 MP3 SHA-256, 분석값, 프레임 수, 박자/프레임 오차,
로봇별 방문 이력을 기록합니다. 음악 원본과 동작 속도는 그대로 유지하고 끝 0.7초만 페이드합니다.

## 동작이 큰 버전 (v2)

`../result/0_pre-result/Rita_Billie_Jean_Expressive_v2.mp4`는 같은 음악·BPM의 확장 버전입니다.
`../result/0_pre-result/Play_Dance_Video.cmd`를 더블 클릭하면 이 음악 포함 MP4를 기본 동영상 플레이어로 엽니다.
`Play_Sim.cmd`는 음악 없이 가상 로봇을 실시간으로 실행하는 별도 기능입니다.
기존 8개 자세는 `default` 프로필로 보존하고, `sim_poses/expressive/`에 로봇별 16개 자세를 추가했습니다.
Kinova는 넓은 팔 스윕과 펼치기/접기, G1은 양팔·허리·다리를 쓰는 자세,
Misty는 큰 머리 회전과 양팔 동작, Go2는 네 다리 들기·웅크리기·벌리기를 포함합니다.

속도·가속도 설정은 기존과 같습니다. 큰 자세 전환은 필요한 시간에 따라 1·2·4박자로 수행하여
음악의 정수 박자에 도착합니다. 덜 방문한 자세를 우선하고 직전 몇 자세로의 반복을 줄입니다.
이 버전의 capacity는 각 간선의 박자 길이를 반영해 `rho(A_ij * exp(-h * beats_ij)) = 1`로 계산합니다.
간선마다 다른 시간이 걸리므로 기존의 단순 `log(rho(A))`와 구분합니다.

```powershell
.\.venv-sim\Scripts\python.exe sim.py dance --robot g1 --profile expressive --bpm 117.02546482191966 --seconds 120
.\.venv-sim\Scripts\python.exe render_dance_video.py --profile expressive --seconds 60 --output sim_output/video/Rita_Billie_Jean_Expressive_v2.mp4
```
