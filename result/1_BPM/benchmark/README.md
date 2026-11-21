# BPM 비교: 발표용 Figure 두 개

이 폴더의 자료는 새 알고리즘 실험이 아니라, 이미 실행한 독립 평가의 원시 측정값을 정리한 것입니다.

- [Figure 1: 정확도](../01_BPM_accuracy.png)
- [Figure 2: CPU 처리 시간](../02_BPM_latency.png)
- PowerPoint용 SVG는 `vector/`에 같은 Figure 두 개가 있습니다.

## 리뷰에 쓸 수 있는 문장

Our improved classical detector achieved the highest observed octave-tolerant tempo accuracy (Acc2: 86.7%) among the four evaluated methods, while matching the fastest neural baseline's observed CPU processing latency (13.17 ms vs. 13.61 ms for TempoCNN).

우리 방법이 가장 높았다는 정확도 주장은 **Acc2(반·두 배 BPM 허용)** 기준입니다. 원래 BPM 자체를 비교하는 Acc1은 함께 표시했습니다. Acc1 최고는 Beat This!이며, TempoCNN과 우리 방법의 처리 시간은 비슷합니다. 오차 구간이 겹치므로 보편적·통계적으로 유의한 우월성이나 정식 동등성 검정을 주장하지 않습니다.

| 방법 | Acc1 | Acc2 | 중앙값 ms | 평균 ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| Ours (Classical V2) | 62.7% | 86.7% | 13.17 | 16.70 | 32.52 |
| DeepRhythm | 58.7% | 82.7% | 78.58 | 120.74 | 217.19 |
| TempoCNN k4 (TF) | 61.3% | 80.0% | 13.61 | 16.67 | 29.60 |
| Beat This! small0 | 64.0% | 76.7% | 439.30 | 954.13 | 2291.68 |

## 비교군과 평가 조건

비교군은 처음부터 요청됐고 실제 pretrained 가중치를 로딩한 세 모델을 유지했습니다: DeepRhythm 0.7, TempoCNN deeptemp-k4-3, Beat This! small0. 더 약한 모델로 교체하지 않았습니다.

**Ours는 `OUR_V2_FAST` 개선 버전**입니다. 원본 `OUR_EXISTING`과 다르며, 기존 로봇 제어 코드는 교체하지 않았습니다. 개발에 사용한 50곡과 다른 50곡을 사용했고 곡 중복은 0입니다. 8·15·30초의 150구간을 모두 정확도에 포함했습니다. 개선 버전의 템포 설정은 평가 전에 고정했으며, 컴파일 최적화 전후 150구간의 BPM과 박자 배열이 정확히 같음을 확인했습니다.

Acc1은 raw BPM 상대오차가 2% 이하일 때 성공입니다. Acc2는 예측값에 0.5, 1, 2를 각각 곱한 경우도 허용합니다. 실패한 예측을 정확도 분모에서 제외하지 않았습니다. MAE 등 전체 수치는 `data/summary.csv`에 있습니다.

정확도 오차 막대는 50곡을 곡 단위로 10,000회 bootstrap한 95% 구간입니다. 같은 곡의 세 구간을 독립 표본으로 간주하지 않았습니다. 구간이 겹치므로 최고 점수는 이번 표본의 관측값을 뜻합니다. 신경망의 학습 자료 중복 여부는 일부 확인되지 않았습니다.

속도는 네 방법이 모두 처리한 **대표 입력 6개**(두 곡 × 8·15·30초)를 전부 사용했습니다. 각 입력에서 CPU 1 thread, 준비 실행 5회, 측정 20회이며 모델당 측정값은 120개입니다. 모델 실행 순서는 매 반복마다 무작위로 섞었습니다. 파일 읽기부터 최종 BPM까지의 전체 처리 비용을 포함합니다. 오디오를 모으는 시간이나 로봇의 실제 반응 지연을 측정한 값은 아닙니다.

**이전 20.05 ms 표와의 차이:** 이전 표는 원본 검출기까지 포함한 6개 방법의 공통 성공 입력을 사용해 8초 입력이 제외된 15·30초 4구간이었습니다. 이번 그림은 표시하는 네 방법의 공통 입력 6구간을 모두 사용하므로 중앙값이 달라집니다. 새 속도 값을 만들거나 유리한 반복만 선택한 것이 아닙니다. 이전 표는 `data/previous_four_clip_summary.json`와 원래 실험 폴더에 보존돼 있습니다.

TempoCNN은 Windows에서 Essentia가 설치되지 않아 공식 가중치를 TensorFlow와 원저자의 feature frontend로 실행했습니다. Beat This!의 BPM은 ordinary beat 간격의 중앙값에서 유도했습니다. 다운비트로 BPM을 계산하지 않았습니다. 최초 컴파일·모델 로딩 비용은 warm latency에 포함되지 않으며 `data/compiler_startup.json`에 별도로 보존했습니다.

DeepRhythm의 공식 분할은 완성된 8초 단위만 사용하므로 8·15·30초 파일의 실제 신경망 문맥은 8·8·24초입니다. 파일 전체를 읽고 resampling하는 시간은 포함됩니다. TempoCNN의 공식 feature frontend는 8초 입력을 256-frame 문맥(약 11.9초)으로 zero-pad합니다. 이 처리를 임의로 바꾸지 않았고, 모델마다 요구하는 전처리를 사용했습니다.

## Figure 재생성

```powershell
# experiment/result/1_BPM/benchmark 폴더에서 실행
..\..\..\3_BPM_check\.venv\Scripts\python.exe make_figures.py
```

독립 환경에서는 Python 3.11에서 `pip install -r requirements.txt` 후 `python make_figures.py`를 실행하면 됩니다. 다운로드나 모델 추론 없이 포함된 CSV에서 그림을 다시 만듭니다. 원래 실험 CSV를 다시 묶으려면 같은 폴더에서 `python collect_inputs.py`를 먼저 실행합니다.

원시 예측 600개와 속도 측정 480개, 모델/하드웨어/분할/검증 metadata는 `data/`에 있습니다. `track_references.csv`에 50곡의 공개 BPM 주석 출처를, `input_clips.csv`에 실제 구간 길이와 WAV 해시를 보존했습니다. `source_provenance.json`은 선택 기준, 측정 범위와 원본 파일의 해시를 기록합니다. 전체 코드와 측정은 `../../../3_BPM_check/improvements/`에서 확인할 수 있습니다. 기존 `1_BPM` 영상과 분석 자산은 유지했습니다.
