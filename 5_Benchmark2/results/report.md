# RoboPerform native 재현 — 실제 실행 결과

공식 student와 제공 latent를 사용하여 G1의 **closed-loop MuJoCo state-feedback**를
실제로 실행했습니다. 이는 Misty/Kinova의 같은 로봇 비교가 아니라 별도의 native Table B입니다.

| 항목 | 관측 결과 |
|---|---|
| source revision | `e6e2a0e4cdb20f081c1a3d46d648d7a202816779` |
| checkpoint | `student_80500.pt`, 563,075,994 bytes |
| model/data 4개 총량 | 1,449,068,884 bytes |
| 실제 policy parameter 수 | 46,912,727 |
| strict student/normalizer load | True (missing/extra keys 0) |
| 유효 forward / closed-loop rollout | True / True |
| custom raw-audio inference | unavailable |
| hardware executed | False |
| 입력 | released music/dance latent + 실제 G1 history |
| 출력 구간 | 2.000초, 100 control steps |
| 원형 control period | 0.02초 |
| policy generation 총시간 | 1.950182초 |
| step median / min / max | 0.017985 / 0.015448 / 0.066037초 |
| step P95 (100회 관측) | 0.025660초 |
| policy RTF | 0.9751 |
| MuJoCo simulation 연산 | 0.104918초 |
| 영상/시각화 연산 | 3.770282초 |
| model construction / checkpoint load | 2.518330 / 0.338410초 |
| latent loading | 0.986438초 |
| 최소 base 높이 | 0.768435 m |
| base<0.35m proxy | False |
| audio alignment / raw-audio end-to-end | N/A / N/A |

CPU PyTorch 2.6.0+cpu, Python 3.11.9, MuJoCo
3.3.2, 1 inference thread에서 실행했습니다. 새로운 학습은 하지 않았습니다.
최종 수치는 다른 benchmark 추론과 인코딩을 중지한 순차 quiet 재측정입니다.
위 step P95는 100개의 연속 state-feedback step 관측이며 서로 다른 2초 궤적을
100개 생성했다는 뜻이 아닙니다. warm-up 2회·독립 workload 5회 비교 시간과도
구분합니다. Native G1 수치를 다른 로봇의 trajectory 생성 시간과 직접 나누지 않습니다.

공식 CPU 원형 실행은 CUDA에 저장된 latent의 `torch.load` 때문에 실패했습니다.
해당 실제 traceback은 `unmodified_cpu_attempt/last_attempt_traceback.txt`에 남겼습니다.
성공 run은 별도의 **device-only CPU port**를 명시적으로 사용합니다. `.cuda()`
placement와 `map_location='cpu'`, 로컬 asset 경로 및 Windows Unicode 경로용
동일 XML/mesh VFS 로딩만 추가했습니다. architecture, 가중치, normalizer,
관측 history 순서, diffusion steps, PD gain, control 주기는 동일합니다.

영상은 무음입니다. 공개 music.pt가 어떤 원본 wave와 대응하는지 이 checkout의
배포 경로에서 확인하지 못했으므로 Billie Jean을 붙이거나 beat alignment를
추정 정확도로 쓰지 않습니다. 2초의 한 가지 released-latent demo만 검증했습니다.
긴 rollout의 안정성·다양성, 실제 로봇 balance/collision/추종 성능은 검증하지 않았습니다.

## 생성 파일

- `trajectories/g1_rollout.npz`, `trajectories/g1_rollout.csv`: 실제 MuJoCo q/qvel, target, action.
- `videos/g1_released_latent.mp4`: 640×480, 25fps, 무음 상태 렌더링.
- `g1_state.png`: joint state와 base height.
- `latency_raw.csv`: 모든 policy step의 처리 시간.
- `run_status.json`, `source_manifest.json`, `environment.json`: 상태·해시·환경.

## 발표용 영어 bullets (이번 측정 범위)

- The released student checkpoint loaded strictly with zero missing or unexpected keys.
- A device-only CPU port completed a 2-second, 100-step closed-loop native G1 MuJoCo rollout.
- The measured policy processing RTF was 0.975; custom-audio inference and audio alignment remain unavailable.
