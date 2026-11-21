# 공통 rollout container

Misty, Kinova, mock과 새 벤치마크의 **한 번의 rollout을 같은 형식**으로 저장합니다.
Docker 이미지가 아니라 실험 데이터 형식과 실시간 recorder입니다. 기존
`rita_core.runtime.run_session()`에 연결되어 있어 실제 로봇 실행에도 자동 적용됩니다.

## 실행과 저장 위치

`experiment/`에서 Python 3.11 이상으로 설치합니다.

```powershell
python -m pip install -e .
python 0_pre_and_etc/run.py run --mode infinite --backend mock --synthetic --headless --seconds 20 --output result/container_demo
```

기존 결과와 함께 `result/container_demo/containers/<run_id>/`에 별도 container가
생깁니다. Python에서 output 없이 `run_session()`을 호출하면 기본 저장 위치는
`6_container/runs/`입니다. 같은 위치에 반복 실행해도 고유 run ID로 분리됩니다.
CLI의 `--container-root result/all_rollouts` 또는 Python의
`container_root="result/all_rollouts"`로 모든 실행의 container 저장 위치를 모을 수 있습니다.

실제 로봇은 검토한 로컬 profile과 기존 실행 옵션을 사용합니다.

```powershell
python 0_pre_and_etc/run.py run --mode infinite --backend kinova --profile kinova.local.json --audio "your_track.wav" --seconds 30 --execute --output result/kinova_run
python 0_pre_and_etc/run.py run --mode finite --backend misty --profile misty.local.json --audio "your_track.wav" --seconds 30 --execute --output result/misty_run
```

로컬 profile은 `local_verified`, workspace 확인, endpoint와 관절 제한 설정이 필요합니다.
구체적인 장비 준비는 [하드웨어 가이드](../0_pre_and_etc/docs/HARDWARE.md)를 따릅니다.
실제 하드웨어 검증은 해당 로봇에서 수행해야 합니다.

| 파일 | 내용 |
| --- | --- |
| `manifest.json` | schema 버전, run ID, method/benchmark, seed, 로봇 profile, 관절 순서·rad 단위, 음악·beat 기준, clock mapping, 실행 상태 |
| `samples.jsonl` | timestamp와 source가 붙은 reference, feedback 또는 simulation 관절 sample |
| `events.jsonl` | 계획, dispatch, 완료, 지연, 실패, stop 등 실행 이벤트 |
| `metrics.jsonl` | 시간에 따른 source별 rolling metric snapshot |
| `live.json` | 실행 중 읽을 수 있는 source별 최신 snapshot (`channels`) |
| `summary.json` | 종료 상태와 최종 source별 측정 결과 |

데이터는 background writer가 저장하고, 제한된 queue로 control 경로와 분리합니다.
기록 실패나 queue 초과는 조용히 sample을 버리는 대신 실행 실패로 처리합니다.
로봇 stop과 정리가 끝난 뒤 logger를 닫습니다. 정상 종료 전에도 완성된 JSONL
행은 읽을 수 있으며 reader는 잘린 마지막 행을 복구할 수 있습니다.

## diversity와 sync의 의미

관절 범위로 정규화한 4-beat motion window 거리, recurrence, 움직임 크기·정지 비율,
motion accent와 beat의 시간 오차를 공통 정의로 계산합니다. Benchmark1과 같은
기본 설정을 사용하며 설정은 container에 저장됩니다. beat reference가 없는
실험에서는 시간 window로 diversity를 계산하고 음악 sync는 N/A로 남깁니다.

`reference`와 실제 `feedback`/`simulation` 결과는 source별로 구분됩니다.
mock feedback은 시뮬레이션이며 실제 로봇 측정으로 표시하지 않습니다. 정지한
로봇이나 유효한 accent가 부족한 데이터에는 완벽한 sync 점수를 주지 않습니다.

feedback 수신 timestamp와 polling timestamp를 함께 보관합니다. 동일 sample을
반복 polling한 경우 측정 시 중복을 제거하고, 불규칙 sample은 일정 주기로
resample합니다. 큰 feedback gap은 유효한 연속 궤적으로 메우지 않습니다.
Misty는 비동기 관절 상태를 조합하며 Kinova도 host 수신 시간과 controller 시간이
다를 수 있습니다. 따라서 sync는 저장된 beat와 feedback 수신 기준의 측정이고
마이크로 검증한 acoustic ground truth가 아닙니다.

온라인 accent 검출은 최근 window에 대한 잠정 결과입니다. sample이 추가되면
window 경계 결과가 달라질 수 있으며 최종 결과는 저장된 raw sample로 재계산합니다.

## 모든 rollout을 한 번에 읽기

```powershell
python 6_container/read_runs.py result --csv result/rollout_summary.csv
python 6_container/read_runs.py result --recompute --csv result/recomputed_summary.csv
```

개별 run 폴더 또는 여러 run이 들어 있는 상위 폴더를 넘깁니다. 서로 다른 로봇의
관절 수는 각 manifest의 순서를 따르며, 로봇·method·source를 유지한 채 읽습니다.

```python
from rita_container.reader import load_run, recompute_run

run = load_run("result/kinova_run/containers/<run_id>")
print(run["manifest"])
print(run["samples"][0])
summary = recompute_run("result/kinova_run/containers/<run_id>")
```

## 새 벤치마크 연결

`RobotProfile`과 선택적인 `AudioAnalysis`를 전달하면 같은 reader와 metric을
사용할 수 있습니다. 실험 고유 항목은 metadata, event, 추가 metric으로 확장합니다.

```python
from rita_container import RolloutRecorder

def my_metric(samples, profile, analysis):
    return {"sample_count": len(samples)}

with RolloutRecorder(
    "result/new_benchmark/containers",
    profile=profile,
    analysis=analysis,  # 음악 기준이 없으면 None
    metadata={"benchmark": "new_benchmark", "method": "baseline", "seed": 7,
              "execution_scope": "hardware"},
    metrics={"my_metric_v1": my_metric},
) as recorder:
    for music_time, q, dq, feedback_time in rollout_samples:
        recorder.record_sample("feedback", music_time, q, dq=dq,
                               feedback_monotonic=feedback_time)
    recorder.record_event("trial_note", note="experiment-specific annotation")
```

추가 metric은 writer 쪽에서 실행됩니다. 실제 policy inference latency는 recorder
호출을 제외한 구간에서 직접 재고 event에 저장합니다. 사용자 정의 metric의
재계산에는 해당 함수를 다시 연결해야 하므로 이름에 버전과 단위를 명시하는 편이 좋습니다.
`recompute_run(path, metrics={"my_metric_v1": my_metric})`으로 다시 연결합니다.
추가 metric 함수는 짧은 시간 안에 반환해야 합니다. 종료 후 전체 run 재계산은 raw
sample을 메모리에 읽으며 긴 rollout에서는 저장 데이터 크기에 비례한 메모리가 필요합니다.

`4_Benchmark1`의 관절 궤적은 reference 채널로, `5_Benchmark2`의 G1 MuJoCo 상태와
target은 simulation/reference 채널로 연결됩니다. RoboPerform의 공개 latent demo에는
실제 음악 timeline이 없으므로 음악 sync를 계산하지 않습니다.

```powershell
python -m unittest discover -s 6_container/tests -v
```
