# 로봇 춤 생성 비교 결과

이 실험은 BPM 검출기 비교가 아니라 기존 finite/infinite와 공개 Reachy primitive scheduler의 오프라인 궤적 비교입니다.
입력: 알려진 synthetic 120 BPM beat grid. 실제 음악의 자연스러움/음원 검출 정확도 실험이 아닙니다.
대상은 동일 Misty 머리 pitch/roll/yaw 3관절의 commanded trajectory이며, 실제 장치·물리 시뮬레이터 상태가 아닙니다.
Reachy는 Apache-2.0 공개 엔지니어링 primitive 구현이며 사전학습 신경망이 아닙니다. 원형과 변환 결과는 분리했습니다.
고정 변환: 명칭에 따른 Euler pitch/roll/yaw, scale=1, 첫 Euler offset을 빼 neutral에 정렬. 위치·안테나는 제외, smoothing/clipping 없음.
실행 전 고정한 설정은 configs/benchmark.json, 외부 소스 revision/license는 source_manifest.json에 있습니다.

## Table A — 같은 로봇 / 같은 관절

|stage|method|seed|duration s|status|recurrence|activity stationary|beat error ms (allowed offbeat grid)|position violation rate|
|---|---|---:|---:|---|---:|---:|---:|---:|
|synthetic_full_cycle|our_finite|0|760.10|generated|0.2793902081501026|0.13577160899881593|0.0|0.0|
|synthetic_full_cycle|our_infinite|0|760.10|generated|0.16787424439139478|0.013550848572556243|49.999999999954525|0.0|
|synthetic_full_cycle|reachy_adapted|0|760.10|generated|0.033784255420139325|0.2527299039600053|0.0|0.1330526685085296|

## 처리 시간

이 보조 실행에서는 반복 latency를 측정하지 않았습니다. quality 행의 wall time을 독립 5회 timing 결과로 해석하지 않습니다.
기존 finite의 그래프 준비/모듈 로드는 initialization에 따로 기록됩니다. 입력은 캐시된 beat이며 raw-audio end-to-end는 N/A입니다.
|method|n|init median s|generation+adaptation median s|min s|max s|RTF|
|---|---:|---:|---:|---:|---:|---:|

초기화 열은 process/library warm-up 이후 per-run planner 생성·캐시 조회입니다. Reachy의 값은 이미 로드한 순수 모듈의 조회이며 cold source import/검증 시간이 아닙니다.
품질 행의 generation_seconds는 다른 작업의 부하가 포함될 수 있습니다. 속도 비교에는 timing_conditions.json의 idle 확인 뒤 수집한 latency_raw/summary만 사용합니다.
chunk_deadline_misses는 offline compute time과 이전 구절 길이의 비교 proxy이며 실제 DAC/hardware deadline 측정이 아닙니다.

## 해석 범위와 실패

- 관절 범위 normalization을 사용하며 실행 자체의 분산으로 나누지 않았습니다. 4-beat window, nonadjacent separation≥2 windows, RMS<0.04가 recurrence입니다.
- 속도 에너지 peak detector는 전 방법에 동일하며 의미 있는 excursion/이벤트 수가 부족하면 alignment N/A입니다. 선언한 pose-accent와 일반 에너지 peak는 서로 다른 event 정의입니다.
- q/dq/ddq 위반은 동일 0.1초 grid의 원본 commanded trajectory에서 측정한 kinematic 지표입니다. 연속시간 보증·균형·충돌·실제 추종을 뜻하지 않습니다.
- Reachy 전환은 원래 cyclic preview의 hard switch이며 global phase를 초기화하지 않습니다. 전환 불연속/위반을 보정해 숨기지 않았습니다.
- Reachy의 빠른 raw 출력이 곧 유효한 Misty 제어 출력이라는 뜻은 아닙니다. 이 고정 projection의 제한 위반과 큰 amplitude를 다양성·RTF와 함께 봐야 합니다.
- deterministic Reachy는 각 음악 구간을 한 번만 품질 평가했습니다. timing 반복은 새 춤 표본이 아닙니다.
- 여러 crop은 모두 같은 곡입니다. 이 소규모 결과로 보편적 우열이나 자연스러움을 결론 내리지 않습니다.

## Table B — RoboPerform native G1

RoboPerform의 native 상태/제공 latent/원본 G1 rollout은 ../5_Benchmark2/results/native_reproduction.csv 및 README를 보세요.
G1은 이 표의 Misty 점수와 합쳐 순위를 만들지 않았으며, custom-audio가 확인되지 않으면 Billie Jean 조건 생성으로 표시하지 않습니다.

## 재실행

```powershell
python setup_external.py
python run_benchmark.py --quick --robot misty-head
python run_benchmark.py --summarize
```

## 발표용 영어 bullet (측정 범위 한정)
- Generated commanded head trajectories for 3 successful same-robot runs using shared estimated beats.
- Reported raw kinematic violations and recurrence without post-hoc amplitude, phase, or octave tuning.
- Separated initialization from five-repeat cached-beat motion generation; native G1 reproduction is a different task.

## 전체 파형 주기 보조 검증

152 beats는 primitive 선택기의 순환입니다. 고정 원형 소스의 head 전체 파형은 760 beats, 모든 native 채널은 2280 beats 주기를 가집니다.
Misty-head 보조 실행은 1520 beats/760.1초, native 전체 보조 실행은 4560 beats/2280.1초를 생성합니다. 자연 음악의 제한된 길이를 반복 재생하지 않았습니다.
lag 760 beat 표시는 원본 함수에서 유도한 고정 주기이며, 결과를 좋게 만드는 사후 phase/octave fitting이 아닙니다. 무한 비반복 보장은 하지 않습니다.


## Table B — native G1 관측 (Table A와 순위 혼합 없음)

|method|input|rollout_ok|duration s|policy seconds|simulation seconds|custom audio|audio alignment|
|---|---|---|---:|---:|---:|---|---|
|RoboPerform device-only CPU port|released music/dance latent|True|2.0000000000000013|1.9501820000295993|0.10491800052113831|False|N/A|

[student strict load·CPU port·실제 state feedback·실패 기록·100-step 원시 시간](../../5_Benchmark2/results/report.md)을 참고하세요.
이 표의 입력은 raw Billie Jean이 아니며, 2초 G1 정책과 12초 Misty-head generator의 시간을 서로 나눠 우열을 만들지 않습니다.
