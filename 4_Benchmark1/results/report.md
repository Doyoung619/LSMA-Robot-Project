# 로봇 춤 생성 비교 결과

이 실험은 BPM 검출기 비교가 아니라 기존 finite/infinite와 공개 Reachy primitive scheduler의 오프라인 궤적 비교입니다.
음악: 동일 Billie Jean 파일의 여러 구간. Beat는 정답 annotation이 없는 shared estimated beat reference입니다.
대상은 동일 Misty 머리 pitch/roll/yaw 3관절의 commanded trajectory이며, 실제 장치·물리 시뮬레이터 상태가 아닙니다.
Reachy는 Apache-2.0 공개 엔지니어링 primitive 구현이며 사전학습 신경망이 아닙니다. 원형과 변환 결과는 분리했습니다.
고정 변환: 명칭에 따른 Euler pitch/roll/yaw, scale=1, 첫 Euler offset을 빼 neutral에 정렬. 위치·안테나는 제외, smoothing/clipping 없음.
실행 전 고정한 설정은 configs/benchmark.json, 외부 소스 revision/license는 source_manifest.json에 있습니다.

## Table A — 같은 로봇 / 같은 관절

|stage|method|seed|duration s|status|recurrence|activity stationary|beat error ms (allowed offbeat grid)|position violation rate|
|---|---|---:|---:|---|---:|---:|---:|---:|
|stage1|our_finite|0|12.00|generated|0.5|0.13333333333333333|53.847388685968276|0.0|
|stage1|our_infinite|0|12.00|generated|0.5|0.016666666666666666|38.126768168877945|0.0|
|stage1|reachy_adapted|0|12.00|generated|0.0|0.3|26.253536337755623|0.10555555555555556|
|stage2|our_finite|0|24.00|generated|0.4|0.11666666666666667|58.53046826377195|0.0|
|stage2|our_infinite|0|24.00|generated|0.5555555555555556|0.03333333333333333|54.683079577804605|0.0|
|stage2|reachy_adapted|0|24.00|generated|0.0|0.24583333333333332|28.76060901326838|0.15|
|stage2|our_finite|1|24.00|generated|0.3111111111111111|0.125|72.07508187857047|0.0|
|stage2|our_infinite|1|24.00|generated|0.6222222222222222|0.008333333333333333|37.29107727704051|0.0|
|stage2|our_finite|2|24.00|generated|0.37777777777777777|0.125|61.03754093928426|0.0|
|stage2|our_infinite|2|24.00|generated|0.6666666666666666|0.025|36.203073825892496|0.0|
|stage2|our_finite|0|24.00|generated|0.4|0.11666666666666667|58.5304682637684|0.0|
|stage2|our_infinite|0|24.00|generated|0.6888888888888889|0.020833333333333332|66.88737718214455|0.0|
|stage2|reachy_adapted|0|24.00|generated|0.0|0.18333333333333332|36.45538638520662|0.10555555555555556|
|stage2|our_finite|1|24.00|generated|0.3111111111111111|0.125|72.07508187856959|0.0|
|stage2|our_infinite|1|24.00|generated|0.4888888888888889|0.016666666666666666|64.38030450663135|0.0|
|stage2|our_finite|2|24.00|generated|0.37777777777777777|0.125|61.03754093928515|0.0|
|stage2|our_infinite|2|24.00|generated|0.6666666666666666|0.025|19.394449857657037|0.0|
|stage2|our_finite|0|24.00|generated|0.4|0.12916666666666668|58.5304682637684|0.0|
|stage2|our_infinite|0|24.00|generated|0.5333333333333333|0.020833333333333332|63.5446136147948|0.0|
|stage2|reachy_adapted|0|24.00|generated|0.0|0.23333333333333334|39.79814995255282|0.16527777777777777|
|stage2|our_finite|1|24.00|generated|0.3111111111111111|0.125|72.07508187856604|0.0|
|stage2|our_infinite|1|24.00|generated|0.6888888888888889|0.0375|48.58093077563552|0.0|
|stage2|our_finite|2|24.00|generated|0.37777777777777777|0.125|61.03754093928515|0.0|
|stage2|our_infinite|2|24.00|generated|0.6444444444444445|0.020833333333333332|55.1877046964222|0.0|
|long|our_finite|0|156.00|generated|0.30954954954954955|0.12564102564102564|43.45592637553608|0.0|
|long|our_infinite|0|156.00|generated|0.17153153153153153|0.016025641025641024|58.877586954565686|0.0|
|long|reachy_adapted|0|156.00|generated|0.017297297297297298|0.2064102564102564|33.0805169826931|0.13696581196581198|
|stress90|our_finite|0|12.00|generated|0.3333333333333333|0.11666666666666667|50.00000000000027|0.0|
|stress90|our_infinite|0|12.00|generated|0.3333333333333333|0.08333333333333333|3.552713678800501e-12|0.0|
|stress90|reachy_adapted|0|12.00|generated|0.0|0.15833333333333333|33.333333333333215|0.125|
|stress120|our_finite|0|12.00|generated|0.3333333333333333|0.11666666666666667|49.99999999999982|0.0|
|stress120|our_infinite|0|12.00|generated|0.5|0.05|0.0|0.0|
|stress120|reachy_adapted|0|12.00|generated|0.0|0.325|0.0|0.08888888888888889|
|stress180|our_finite|0|12.00|generated|0.5714285714285714|0.125|49.99999999999982|0.0|
|stress180|our_infinite|0|12.00|generated|0.3333333333333333|0.041666666666666664|33.33333333333455|0.0|
|stress180|reachy_adapted|0|12.00|generated|0.0|0.21666666666666667|33.33333333333055|0.1361111111111111|

## 처리 시간

아래 시간은 동일 12초 출력, batch1, CPU 1 thread, 방법 순차 실행, 2 warm-up 후 5회 측정입니다. 5회로 안정적인 P95를 주장하지 않습니다.
기존 finite의 그래프 준비/모듈 로드는 initialization에 따로 기록됩니다. 입력은 캐시된 beat이며 raw-audio end-to-end는 N/A입니다.
|method|n|init median s|generation+adaptation median s|min s|max s|RTF|
|---|---:|---:|---:|---:|---:|---:|
|our_finite|5|2.109336|0.023602|0.021413|0.075335|0.001967|
|our_infinite|5|0.000177|0.336158|0.302169|0.423268|0.028013|
|reachy_adapted|5|0.000047|0.002751|0.002676|0.003126|0.000229|

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
- Generated commanded head trajectories for 36 successful same-robot runs using shared estimated beats.
- Reported raw kinematic violations and recurrence without post-hoc amplitude, phase, or octave tuning.
- Separated initialization from five-repeat cached-beat motion generation; native G1 reproduction is a different task.

## Reachy native 원형 기록

|method|duration seconds|status|input|
|---|---:|---|---|
|Reachy primitive scheduler (native)|12.0|generated_native|shared estimated beat reference|

## Table B — native G1 관측 (Table A와 순위 혼합 없음)

|method|input|rollout_ok|duration s|policy seconds|simulation seconds|custom audio|audio alignment|
|---|---|---|---:|---:|---:|---|---|
|RoboPerform device-only CPU port|released music/dance latent|True|2.0000000000000013|1.9501820000295993|0.10491800052113831|False|N/A|

[student strict load·CPU port·실제 state feedback·실패 기록·100-step 원시 시간](../../5_Benchmark2/results/report.md)을 참고하세요.
이 표의 입력은 raw Billie Jean이 아니며, 2초 G1 정책과 12초 Misty-head generator의 시간을 서로 나눠 우열을 만들지 않습니다.
