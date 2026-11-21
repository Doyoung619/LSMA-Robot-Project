# 4_Benchmark1 — 로봇 춤 생성 비교

기존 **Our finite / Our infinite**를 변경하지 않고 호출하고, 공식 **Reachy
primitive scheduler**를 실제 다운로드·샘플링하는 오프라인 실험입니다.
`3_BPM_check`의 BPM 검출기 비교와 별도입니다. 신경망 가중치가 필요 없는
Reachy 엔지니어링 baseline이며, RoboPerform 학술 재현은 `../5_Benchmark2`입니다.

## 설치 후 바로 실행

Python 3.11 이상, Git, FFmpeg가 필요합니다. clone한 Git 저장소의 루트에서:

```powershell
cd 4_Benchmark1
.\setup_benchmark.ps1
.\.venv\Scripts\python.exe run_benchmark.py --quick --robot misty-head
```

PowerShell 이외 환경의 동등한 명령:

```bash
cd 4_Benchmark1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup_external.py
.venv/bin/python run_benchmark.py --quick --robot misty-head
```

`setup_external.py`는 공식 저장소의 고정 revision을 `external/`에 내려받고
소스 해시를 검증합니다. 외부 SDK 장치 객체나 `examples/dance_demo.py`를
실행하지 않습니다. 실제 가중치가 필요한 3/5번과 달리 이 비교군은 NumPy
순수 동작 함수만 사용합니다. root의 finite/infinite는 `requirements.txt`의
editable install과 내부 bootstrap으로 연결되며 사용자 PYTHONPATH는 필요 없습니다.

기본 음원은 저장소에 포함된 `../0_pre_and_etc/Billie Jean.mp3`입니다.
`--audio /path/to/music.mp3`로 바꿀 수 있습니다. 다른 음원 결과를 같은 실행으로
덮어쓰지 않으려면 `--output /path/to/new_results`를 지정하세요.

```powershell
# 가장 빠른 실제 생성: 같은 12초를 세 방법으로 생성, timing/encoding 생략
.\.venv\Scripts\python.exe run_benchmark.py --stage stage1 --skip-timing --skip-visuals
# 원형 Reachy offsets만 별도로 저장
.\.venv\Scripts\python.exe run_benchmark.py --method reachy_native --stage stage1 --skip-timing --skip-visuals
# 기존 결과에서 그림/한국어 보고서 재생성 (추론 없음)
.\.venv\Scripts\python.exe run_benchmark.py --summarize
# 품질 결과를 보존하고 동일 workload의 timing만 다시 측정
.\.venv\Scripts\python.exe run_benchmark.py --timing-only --skip-visuals
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_results.py
```

## 고정한 실험 조건

- **Misty head 3관절**: head_pitch/head_roll/head_yaw. 기존 unverified Misty
  example profile에서 뽑은 neutral, position/velocity/acceleration 제한을 모든
  방법에 동일 적용합니다. arms/wheels는 비교에서 제외합니다.
- 입력은 같은 곡의 **shared estimated beat reference + 실제 분석된 강세**입니다.
  Billie Jean의 수동 beat 정답이 아닙니다. 본 실험은 BPM detector 차이를 비교하지 않습니다.
- 시작 자세는 매 crop에서 neutral입니다. Reachy 변환은 이름에 따른 Euler
  pitch/roll/yaw, scale=1.0, 첫 orientation offset을 빼는 고정 anchoring입니다.
  native xyz/antenna는 저장하되 Misty 팔로 임의 매핑하지 않습니다.
- native scheduler는 공식 20개 등록 동작 중 preview의 제외 동작 1개를 뺀
  19개를 8 beat씩 순회합니다. global phase를 유지하고 crossfade를 추가하지
  않습니다. [REACHY.md](REACHY.md)에 원본 코드와의 대응·라이선스가 있습니다.
- **보정 없음**: raw q를 clipping/smoothing/projection하지 않습니다. Reachy
  변환에서 나타나는 제한 위반·hard-switch jump도 그대로 보고합니다. 이 변환은
  좌표 이름에 따른 비교용 projection이며 실기 보정된 기구학 변환이 아닙니다.
- `configs/benchmark.json`과 `configs/reachy_adaptation.json`은 결과를 보기 전에
  정했습니다. 결과가 유리해질 때까지 scale, phase, octave, threshold를 찾지 않습니다.

Stage1은 12초, Stage2는 같은 곡의 start beat 16/64/112에서 각 24초입니다.
확률적 Our 방법은 seed 0/1/2, 결정적 Reachy는 crop당 한 번만 품질 평가합니다.
Long은 약 156초로 최소 120초 및 152-beat selection cycle 두 회를 포함합니다.
selector가 반복해도 primitive 자체의 global phase까지 그 주기로 반복한다는
보장은 없습니다. 원본 함수에서 유도하고 dense phase로 검증한 **Misty-head 전체
파형은 760 beats**, native 전체 채널은 **2280 beats**입니다. 실제 곡 길이가
두 전체 주기를 담지 못하므로, 다음 추가 검증은 음악을 반복하지 않고 **알려진
synthetic 120 BPM**으로 분리해 실제 생성합니다.

```powershell
# 같은 Misty-head 3방법: 1520 beats / 760.1 seconds
.\.venv\Scripts\python.exe run_benchmark.py --stage full-cycle --skip-timing --output results_full_cycle
# Reachy native 전체 채널: 4560 beats / 2280.1 seconds
.\.venv\Scripts\python.exe run_benchmark.py --method reachy_native --stage full-cycle --skip-timing --skip-visuals --output results_native_full_cycle
```

`results/upstream_period_validation.json`에 주기 근거가 있습니다. 보조 그림은
`results_full_cycle/figures/full_cycle_lag_similarity.png`에 저장됩니다. 이것은
자연스러움 평가나 영원히 반복하지 않는다는 증거가 아닙니다. 90/120/180 BPM
검사도 known synthetic beat grid이며 실제 음악 구간 평가와 별도입니다.

## 지표와 시간

공통 0.1초 grid에서 원본 q 및 수치 dq/ddq/jerk를 측정합니다. 관절의 물리적
범위로 normalization하며 실행 자체의 분산으로 나누지 않습니다. 4-beat/32-sample
window, 인접 창 제외, RMS<0.04라는 고정 recurrence 규칙과 lag-similarity를 씁니다.
작은 jitter·ID 개수·파일 hash 차이를 공통 춤 다양성으로 세지 않습니다.

동일한 velocity-energy peak detector와 half-beat 허용 grid를 사용합니다. 움직임이나
event 수가 부족하면 alignment는 N/A입니다. energy peak와 각 생성기의 선언한
pose-accent는 다른 event 정의이므로 둘의 coverage를 함께 해석해야 합니다.
이 실험은 commanded reference의 sampled kinematic feasibility이며, 실제 추종·균형·
collision-free motion이나 물리 시뮬레이터 결과가 아닙니다.

Primary timing은 CPU 1 thread, batch1, 같은 12초 출력, **2 warm-up + 5회 측정**입니다.
median/min/max와 RTF를 기록하며 5회로 안정적인 P95를 주장하지 않습니다. finite의
초기 그래프 준비와 framework/module 초기화는 별도 열입니다. 주요 생성 비용은
motion generation + documented adaptation + common correction(없음)이며, 다운로드·
시각화·인코딩·평가는 제외합니다. 기존 planner 내부의 제약 인증 비용은 그대로
generation에 포함합니다. 외부의 raw 출력은 공통 evaluator에서 검사하되 보정하지 않습니다.

초기화 열은 warm-up 이후의 per-run 생성·캐시 조회입니다. Reachy 소스는 timed
loop 전에 로드하므로 그 열을 cold module import 시간으로 읽으면 안 됩니다.
품질 행의 wall time에는 다른 실험의 부하가 있을 수 있으며, primary 속도는
idle 확인 뒤 별도로 수집한 `latency_raw.csv`만 사용합니다.

입력 beat는 캐시되어 있으므로 raw-audio end-to-end 시간은 N/A입니다. 공통 분석/
cache-load 시간은 manifest에 별도로 기록합니다. Our infinite의 horizon/lookahead,
first valid chunk, deadline miss도 trajectory sidecar에 남깁니다. crop 끝이 이미
인증된 구절 중간일 수 있으며, 보이지 않는 미래 계획 비용도 기록됩니다.
여기의 chunk deadline은 offline wall time/구절 길이의 proxy이며 실제 DAC/로봇
실행 deadline이 아닙니다.

다른 `3_BPM_check`/`5_Benchmark2` Python 실험이 실행 중이면 timing 전에 끝나기를
기다립니다(기본 900초, `--wait-idle-seconds`). 다른 프로세스를 종료하지 않습니다.
일반 OS 부하까지 제거한 전용 머신 성능이라는 뜻은 아닙니다. 3/5번의 CPU/GPU·
latent 입력·서로 다른 출력 길이를 이 timing과 합쳐 배율 순위로 만들지 않습니다.

## 결과와 재현 자료

`results/report.md`는 한국어 결과와 Table A, 영어 발표 bullet 3개를 포함합니다.
`same_robot_metrics.csv`, `latency_raw.csv`, `latency_summary.csv`, `seed_distances.csv`,
`lag_similarity.json`, `environment.json`, `source_manifest.json`, `config_snapshot.json`,
`run_status.json`, `timing_conditions.json`에서 원시 숫자와 조건을 확인할 수 있습니다.
`results/figures/`에는 recurrence/activity/feasibility/latency/long-lag 그림이 있습니다.

각 실행의 실제 NPZ/CSV는 `results/trajectories/`, 비교 영상은 `results/videos/`에
저장됩니다. 영상은 **Misty-head commanded trajectory animation**으로 표시하며
물리 시뮬레이션으로 부르지 않습니다. 모델·외부 소스·가상환경·큰 raw 결과는
Git에 넣지 않고, 코드·설정·작은 결과/그림을 versioning합니다. clone/pull 뒤 위
명령으로 raw 결과까지 다시 생성할 수 있습니다. 원형 G1 영상/결과는 5번에 분리됩니다.

이 소규모 비교로 우리 방법이 보편적으로 더 우수하다고 결론 내리지 않습니다.
다양성은 movement activity·jerk·제약 위반과 함께 읽어야 합니다.
