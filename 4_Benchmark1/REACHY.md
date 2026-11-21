# Reachy 공개 primitive 비교군

이 비교군은 사전학습 신경망이 아닌 Pollen Robotics의 공개 엔지니어링
춤 라이브러리입니다. 보고서 이름은 원형의 경우 **Reachy primitive
scheduler (native)**, Misty head 변환의 경우 **Reachy primitive scheduler +
documented adaptation**으로 구분합니다. 학습·가중치 다운로드는 없습니다.

공식 소스는 [reachy_mini_dances_library](https://github.com/pollen-robotics/reachy_mini_dances_library)의
revision `6808a976d3ddc11952ece67eb2403934fbb62af9`(version 0.2.1,
Apache-2.0)로 고정했습니다. `setup_external.py`는 이 revision만 checkout하고
주요 원본 파일의 SHA-256과 크기를 `external/reachy_source_manifest.json`에
기록합니다. 다운로드한 원본 저장소와 환경은 Git에서 제외됩니다. 변경된
upstream source를 원형 재현으로 처리하지 않고 명시적으로 실패합니다.

## 다른 컴퓨터에서 준비와 안전한 오프라인 실행

아래는 `experiment/4_Benchmark1` 폴더 기준입니다. 새 clone에서는
`4_Benchmark1`으로 이동합니다. Python 3.10+와 Git이 필요합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reachy.txt
.\.venv\Scripts\python.exe setup_external.py
.\.venv\Scripts\python.exe reachy_adapter.py --seconds 14
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_reachy.py -v
```

Linux/macOS에서는 `.venv/Scripts/python.exe` 대신 `.venv/bin/python`을
사용합니다. 이번 컴퓨터의 별도 `.venv`는 설치된 시스템 수치 패키지를
공유하도록 만들었으며 NumPy 2.1.3을 사용했습니다. 전체 통합 벤치마크의
환경·잠금 의존성 및 실행 명령은 이 폴더의 README를 따릅니다.

`examples/dance_demo.py`는 실행하지 않습니다. 이 예제는 실제 `ReachyMini`
객체를 생성하고 `wake_up`, `set_target`, `goto_sleep`을 호출합니다.
최상위 upstream package initializer 역시 SDK wrapper `DanceMove`를 import합니다.
오프라인 adapter는 private namespace package 아래에서 수정하지 않은 원본
`rhythmic_motion.py`와 `collection/dance.py`만 로드합니다. 이는 SDK를
흉내 내거나 primitive를 다시 구현하는 방식이 아닙니다. 순수 NumPy 함수
호출만 사용하며 장치 연결·네트워크·모터 명령 경로는 없습니다.

## 정확히 재현한 선택 정책

고정한 원본의 [`examples/dance_demo.py`](https://github.com/pollen-robotics/reachy_mini_dances_library/blob/6808a976d3ddc11952ece67eb2403934fbb62af9/examples/dance_demo.py)를
AST로 확인하고 설정을 대조합니다.

- 등록 primitive는 **20개**, 기본 preview는 `headbanger_combo` 하나를 제외한
  **19개**입니다. `AVAILABLE_MOVES`의 insertion order를 유지하고
  `simple_nod`에서 시작합니다.
- 각 primitive의 선택 유지 시간은 **8 beat**입니다. 전체 선택 cycle은
  **152 beat**이며 두 cycle은 304 beat입니다.
- 각 함수에 전달하는 시간은 **누적 global beat phase**입니다. primitive를
  변경할 때 motion phase를 0으로 재설정하지 않습니다.
- default parameter는 deep copy하고 amplitude scale은 **1.0**입니다.
  별도 random selector, novelty objective, half-time 선택, crossfade나
  smoothing을 추가하지 않았습니다.
- 원본 하드웨어 예제의 `time.time()` jitter와 8-beat 경계 overshoot 버림은
  측정 가능한 이상적인 beat boundary로 대체합니다. 선택 순서·8-beat
  유지·제외 목록·global motion phase·파라미터는 그대로입니다.
- 선택 cycle의 반복이 모든 연속 출력의 동일 주기를 뜻하지는 않습니다.
  일부 primitive의 `subcycles_per_beat`와 global phase 때문에 실제 출력
  반복에는 선택 cycle 여러 번이 필요할 수 있습니다. 장기 recurrence를
  수치로 확인하며 14초 영상만으로 비반복을 주장하지 않습니다.

`sample_native(times_seconds, beat_phase)`에 원본 전체 음원의 공통 beat
timeline에서 계산한 phase를 공급합니다. crop마다 음악 phase를 0으로
바꾸지 않습니다. beat 간격이 변해도 caller의 누적 phase를 그대로
사용합니다. 첫 beat 이전의 phase가 음수인 구간은 neutral offset입니다.

출력은 `position_m[n,3]`(x/y/z), `orientation_rad[n,3]`(roll/pitch/yaw),
`antenna_rad[n,2]`, `primitive_ids`, source 및 scheduler metadata입니다.
14초 smoke는 `results/reachy_native_smoke/native_trajectory.npz`, CSV,
`smoke_status.json`으로 저장합니다. 이 단일 smoke 시간은 5회 반복 quick
latency 측정이나 P95로 해석하지 않습니다.

## 평가 전에 고정한 Misty 변환

`configs/reachy_adaptation.json`의 정책은 결과 확인 전에 고정했습니다.
기본 same-robot 비교는 기존 Misty 예제 profile에서 head 세 관절만
사용합니다: `(head_pitch, head_roll, head_yaw)`. 원본 RPY에서 index
`(1,0,2)`를 선택하고 부호 `(+,+,+)`, scale **1.0**으로 사용합니다.
이는 이름이 같은 Euler 좌표의 직접 투영이며 두 로봇의 실제 축을
교정한 rigid-transform/IK라고 주장하지 않습니다.

각 crop에서 첫 Euler offset을 한 번 빼고 profile neutral을 더합니다.
`q_head(t) = neutral + scale * (RPY_reordered(t) - RPY_reordered(t_first))`.
이 고정 anchoring은 finite/infinite와 동일한 neutral 초기 자세를
만들기 위한 것이며 결과에 맞춰 scale·phase를 검색하지 않습니다.
실제로 뺀 세 값은 각 run의 mapping metadata에 기록합니다. Native
원형 결과에는 anchoring을 적용하지 않습니다. 두 방식의 곡선과
경계 불연속은 그대로 남습니다.

Reachy의 x/y/z translation과 두 antenna는 제외합니다. antenna를 Misty
arm으로 같은 관절처럼 대응하지 않습니다. 5-joint head/arms profile을
명시적으로 넘길 경우 양팔은 neutral로 고정하며 기본 비교는 그
움직이지 않는 arm을 다양성 점수에 포함하지 않는 head-only subset입니다.
다른 joint 순서나 degree 단위는 구체적인 오류로 처리합니다.

clipping, smoothing, projection은 적용하지 않습니다. 예제 head bound를
넘는 원본 amplitude, 각도 변경 속도·가속도, 선택 경계 discontinuity는
통합 벤치마크가 보정 전 출력에서 그대로 측정합니다. 이 결과는
**commanded trajectory / kinematic-limit evaluation**이며 actual Misty
tracking, balance, collision-free motion이나 하드웨어 성공이 아닙니다.
