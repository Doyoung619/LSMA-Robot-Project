# Misty II 실제 로봇 실행 데이터

2026-11-21에 Misty II 두 대로 실행한 결과입니다. 모든 실행은 `6_container` 형식으로 기록했습니다.
음악은 `0_pre_and_etc/Billie Jean.mp3`(약 117 BPM)이고, 로봇은 머리 3축, 양팔, 제자리 바퀴 회전을 사용했습니다.

## 폴더 구성

| 폴더 | 내용 | seed | 길이 |
| --- | --- | --- | --- |
| `single_robot/finite_seed7` | 로봇 1대, 1_finite | 7 | 60초 |
| `single_robot/finite_seed8` | 로봇 1대, 1_finite | 8 | 60초 |
| `single_robot/finite_seed9` | 로봇 1대, 1_finite | 9 | 60초 |
| `single_robot/infinite_seed7` | 로봇 1대, 2_infinite | 7 | 60초 |
| `two_robot/finite_same/{A,B}` | 로봇 2대, 같은 안무, 1_finite | 7 / 7 | 60초 |
| `two_robot/infinite_same/{A,B}` | 로봇 2대, 같은 안무, 2_infinite | 7 / 7 | 30초 |
| `two_robot/finite_different/{A,B}` | 로봇 2대, 다른 안무, 1_finite | 7 / 8 | 30초 |
| `two_robot/infinite_different/{A,B}` | 로봇 2대, 다른 안무, 2_infinite | 7 / 8 | 30초 |

로봇 1대 실행은 192.168.1.39 로봇을 썼습니다. 로봇 2대 실행에서 A는 192.168.1.38, B는 192.168.1.39입니다.

각 실행 폴더에는 다음 파일이 있습니다.

- `containers/<run_id>/`: `6_container` 형식 파일 6개 (`manifest.json`, `samples.jsonl`, `events.jsonl`, `metrics.jsonl`, `live.json`, `summary.json`)
- `summary.json`, `summary.png`, `feedback.csv`, `reference.csv`: `run.py`가 원래 저장하는 결과

`rollout_summary.csv`는 모든 실행의 최종 지표를 한 표로 모은 파일입니다. 12개 실행의 feedback과 reference 채널이 한 줄씩 들어 있습니다(24줄). 모든 줄의 상태는 `completed`, 지표 상태는 `ok`입니다.

## 읽는 방법

저장소 루트에서 실행합니다.

```bash
python 6_container/read_runs.py misty_hardware_runs --csv out.csv
python 6_container/read_runs.py misty_hardware_runs --recompute --csv out_recomputed.csv
```

## 데이터를 쓸 때 알아야 할 점

- **실제 관절 데이터는 약 5 Hz입니다.** Misty II는 `ActuatorPosition`을 관절마다 약 0.2초 간격(최대 0.32초)으로 보냅니다. 요청한 주기와 관계없이 같습니다. 박자 하나가 0.51초이므로, 박자와의 시간 오차 지표는 그만큼 정밀도가 낮습니다. 프로필의 `command_period`는 이 값에 맞춰 0.2초입니다.
- **명령 방식이 계획 곡선과 다릅니다.** 포즈 지점마다 머리 명령(Duration 지정)과 팔 명령(속도 % 지정)을 한 번씩 보냅니다. 포즈 사이의 움직임은 Misty가 스스로 보간합니다. 그래서 reference 곡선과 실제 궤적의 모양은 같지 않습니다.
- **지연 보정:** 머리는 명령 후 약 0.2초, 팔은 약 0.45초 뒤에 움직입니다. 그만큼 명령을 미리 보냈습니다.
- **추종 오차 검사를 넓혔습니다.** 머리는 0.8 rad, 팔은 3.0 rad입니다. 팔은 관절 범위 검사로만 지킵니다.
- **1_finite는 리듬 13종 중 6종만 쓰였습니다.** 동작을 키우면서 최소 리듬 기준을 4로 낮췄기 때문입니다. 2_infinite는 60초에 10종이 쓰였지만, 0.35초보다 가까운 포즈는 합쳐서 보냈습니다.
- **바퀴 회전은 데이터에 없습니다.** 바퀴에는 위치 피드백이 없어서 `samples.jsonl`에 기록되지 않습니다. 보낸 명령은 제자리 회전(LinearVelocity 0)뿐입니다.
- **manifest의 `git_dirty`는 true입니다.** 실행 당시 커밋하지 않은 수정이 있었습니다. 그 수정은 이후 `misty-hardware-dance` 브랜치에 커밋했습니다.
- **경로를 바꿨습니다.** 파일 안의 로컬 절대 경로는 `<repo>`로 바꿨습니다. 측정값은 바꾸지 않았습니다. 오디오 캐시(`audio_cache/`, 실행마다 약 49 MB)는 넣지 않았습니다.
- **영상과의 관계:** 로봇 2대 영상은 이 데이터와 같은 실행입니다. 로봇 1대 영상은 같은 설정으로 따로 실행한 것입니다.
