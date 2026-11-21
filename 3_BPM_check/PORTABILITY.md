# 다른 컴퓨터에서 BPM 벤치마크 재실행

이 폴더는 **BPM 검출기** 비교입니다. 로봇 춤 생성의 외부 비교군은
`../4_Benchmark1/`와 `../5_Benchmark2/`이며, 두 작업의 지표를 섞지 않습니다.

Git clone의 최상위가 원래 Rita-Robot 작업공간의 `experiment/`입니다.
아래 명령은 clone한 저장소의 최상위에서 시작합니다. 모델과 데이터셋은
자동으로 Git에 포함되지 않으며, 준비 단계에서 각 컴퓨터로 내려받습니다.

## Windows / CPU

Python 3.11과 FFmpeg를 준비한 뒤 PowerShell에서 실행합니다.

```powershell
cd 3_BPM_check
powershell -ExecutionPolicy Bypass -File setup_benchmark.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe run_benchmark.py --quick --device cpu
```

`setup_benchmark.ps1`은 별도 `.venv`를 만들고 `requirements.txt`의 CPU
의존성을 설치한 뒤 `download_models.py`를 실행합니다. 가중치 다운로드나
SHA256 검증이 실패하면 nonzero exit code로 준비 실패를 알립니다.
기존 로봇용 Python 환경을 활성화하거나 수정할 필요가 없습니다.

다른 셸에서는 Python 3.11로 `.venv`를 만든 뒤 해당 환경의 Python으로
다음 명령을 실행할 수 있습니다. Linux/macOS의 환경 실행 경로는
`.venv/bin/python`입니다. 현재 저장된 결과는 Windows CPU 실행이며,
다른 OS의 모든 패키지 조합을 직접 검증한 것은 아닙니다.

```bash
python -m pip install -r requirements.txt
python download_models.py
python -m unittest discover -s tests -v
python run_benchmark.py --quick --device cpu
```

`requirements-lock.txt`는 측정 당시 Windows 환경의 기록입니다. 다른 OS에
Windows 전용 lock 전체를 강제로 설치하기보다 위 direct requirements를
사용하고, 실제 패키지 버전은 새 결과의 `environment.json`에 남깁니다.

## 모델 준비와 용량

2026-11-21에 로컬에서 다음 세 공식 checkpoint의 파일과 SHA256을 다시
검증했습니다. 세 가중치 합계는 **14,041,890 bytes**입니다. 이는 PyTorch,
TensorFlow 등 프레임워크 설치 크기나 실행 RAM을 포함하지 않습니다.

| 방법 | 파일 | bytes |
|---|---|---:|
| DeepRhythm | `deeprhythm-0.7.pth` | 5,443,228 |
| TempoCNN | `deeptemp-k4-3.pb` | 147,561 |
| Beat This! | `beat_this-small0.ckpt` | 8,451,101 |

다운로드 URL은 `download_models.py`, 고정 SHA256은
`adapters/model_cache.py`에 있습니다. 준비된 파일도 실제 load 전에 해시를
다시 검사합니다. 모델을 교체하려면 URL/해시/라이선스를 의도적으로 함께
검토해야 하며, 다른 모델을 같은 결과인 것처럼 쓰지 않습니다.

`models/`, 가상환경, 데이터셋 음원과 임시 smoke 출력은 gitignored입니다.
`results/portability_verification.json`에는 이번 clone 재현 점검 범위만
작게 기록했습니다. 원래 `results/`의 전체 비교 결과를 smoke 결과로
덮어쓰지 않았습니다.

## 실행 범위와 제한

- CPU PyTorch 2.6.0 환경에서 원래 검출기와 세 모델의 실제 30초 Billie Jean
  입력 predict/load 경로를 다시 확인했습니다. 이 곡에는 정답 beat/BPM
  라벨이 없으므로 이 smoke를 검출 정확도나 모델 순위의 근거로 쓰지 않습니다.
- 이번 portability smoke는 timing 1회이므로 안정적인 P95를 주장하지 않습니다.
  기본 `--quick`의 warm-up/반복 측정과 별개입니다.
- Windows TempoCNN은 공식 가중치를 사용하는 명시적 TensorFlow 호환 경로입니다.
  native Essentia runtime 측정이라고 부르지 않습니다.
- BeatNet은 기존 Python 3.11 dependency 충돌 때문에 optional unavailable로
  기록돼 있습니다. 준비 성공한 세 모델의 결과와 혼동하지 않습니다.
- `OUR_EXISTING`은 원래 소스 호출입니다. 개발 중인 `OUR_V2`가 registry에
  존재하더라도 원래 방법을 대체하지 않으며, 독립 holdout 검증이 필요합니다.
- 모델별 라이선스와 데이터 provenance는 `INSTALLATION.md` 및
  `results/model_metadata.csv`를 확인하세요.
