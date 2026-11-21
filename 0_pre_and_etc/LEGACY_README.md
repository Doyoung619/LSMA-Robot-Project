# Rita: Billie Jean Kinova dance demo

> ?? ?? ? ? ??? ??? `experiment/0_pre_and_etc`?? ?????. ??? ??? `../result/0_pre-result/`? ???, ?? ?? ??? `../result/0_pre-result/Play_Dance_Video.cmd`? ?????. ? ??? ?? ??? ? ?? ??? ?????.

**가상 로봇 실험 환경:** Kinova 6축·Unitree G1·단순화 Misty·Unitree Go2를 MuJoCo에서 실행하려면 `Play_Sim.cmd`를 더블 클릭하세요. 설치 및 자세 편집 방법은 [SIMULATION.md](SIMULATION.md)를 참고하세요.

**음악 동기화 영상:** [Billie Jean에 맞춰 춤추는 네 로봇 (60초, 음악 포함)](sim_output/video/Rita_Billie_Jean_4_Robots.mp4). 원본 MP3에서 측정한 117.025 BPM과 첫 박자 위상에 맞춘 1080p·30fps 영상입니다.

**큰 동작 버전:** [자세 16개·넓은 움직임으로 확장한 v2 영상](../result/0_pre-result/Rita_Billie_Jean_Expressive_v2.mp4). 같은 BPM에서 큰 전환을 1·2·4박자로 수행하며, `Play_Sim.cmd`도 이 버전으로 실행됩니다.

이 폴더의 `Billie Jean.mp3`를 컴퓨터 스피커로 재생하면서 Kinova Gen3 6-DOF가 박자에 맞춰 여러 동작을 이어 갑니다. `Rita_Robot.pdf`의 learning-free primitive library 아이디어를 바탕으로 만든 **첫 실물 데모**입니다. 논문의 transition-time 최적화, Dance Graph의 spectral radius 및 Tempo–DOF 실험 전체를 구현했다고 주장하는 코드는 아닙니다.

## 바로 실행

현재 컴퓨터에는 `.venv` 및 공식 Kinova API 설치와 오디오 분석이 완료되어 있습니다. 로봇 IP는 `192.168.2.9`, 컴퓨터 유선 IP는 `192.168.2.10/24`입니다.

- `Play_Demo.cmd` 더블 클릭: 음악 60초 + 춤.
- `Play_Full_Song.cmd` 더블 클릭: 음악 전체 약 4분 54초 + 춤.
- 실행 창에서 **Q / Esc / Ctrl+C**: 중단.
- `Stop_Rita.cmd`: 별도 연결로 정지 명령, `STOP_RITA` 파일을 남겨 재시작 방지.
- 물리 비상정지 버튼은 항상 바로 조작할 수 있어야 합니다.

`Stop_Rita.cmd`를 사용했다면, 재시작을 의도할 때만 PowerShell에서 `Remove-Item -LiteralPath .\STOP_RITA`로 래치를 해제합니다. 소프트웨어 정지는 안전 등급 비상정지 기능을 대체하지 않습니다.

로봇 팔 전체의 이동 공간을 확보하고 조이스틱/웹 수동 제어를 해제한 뒤 실행합니다. 3초 카운트다운 후 약 8초 동안 관절 한계에서 벗어난 기준 자세로 천천히 이동하고 음악을 시작합니다. 마지막에는 이 기준 자세로 부드럽게 돌아옵니다. **한계 부근의 원래 자세로 다시 돌아가지 않습니다.** 그리퍼는 작동시키지 않습니다.

## 명령행

```powershell
.\.venv\Scripts\python.exe rita.py analyze
.\.venv\Scripts\python.exe rita.py inspect
.\.venv\Scripts\python.exe rita.py preview --seconds 60
.\.venv\Scripts\python.exe rita.py run --seconds 60
.\.venv\Scripts\python.exe rita.py run
.\.venv\Scripts\python.exe rita.py run --start 45 --seconds 60 --seed 12 --intensity 0.8
.\.venv\Scripts\python.exe rita.py devices
.\.venv\Scripts\python.exe rita.py run --seconds 60 --device 8 --volume 0.5
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

명령은 폴더 안에서 실행합니다. `preview`는 로봇 상태를 읽지만 이동하지 않습니다. `preview --offline`은 마지막 저장 상태만 사용합니다. `--device` 번호는 `devices`에서 확인합니다. `--dof 1..6`은 춤 중 활성 관절을 제한합니다. 준비 동작에서는 한계 부근 관절을 안쪽으로 옮길 수 있으므로 논문의 strict joint-locking 실험과는 다릅니다.

다른 컴퓨터에서는 Python 3.11 이상, `ffmpeg`를 먼저 설치하고 `powershell -ExecutionPolicy Bypass -File .\Setup.ps1`를 실행합니다. 가상 환경에 필요한 작은 Python 패키지와 공식 Kortex wheel을 설치합니다. 신경망 가중치는 다운로드하지 않습니다. 인증 정보가 기본값과 다르면 환경 변수 `KINOVA_USERNAME`, `KINOVA_PASSWORD`를 사용합니다. 자격 증명은 로그에 기록하지 않습니다.

## 음악과 동작

- 검출 BPM: 약 **117.03**, 박자 간격 약 **0.51271초**. `data/analysis.json`에 측정 결과가 있습니다.
- FFT 다중 주파수 대역 onset, autocorrelation, robust beat-grid 회귀로 박자를 찾습니다. 고정 BPM 곡을 위한 경량 분석이며 신경망/음악 생성 모델은 사용하지 않습니다. 다른 곡의 가변 tempo, downbeat, 감정/구간 의미를 추정하지 않습니다.
- 이 파일에서는 박자 격자와 가장 가까운 onset 사이 중앙값이 약 27ms였습니다. 이는 로봇의 음악 동기화 오차를 측정한 수치가 아닙니다. 정확한 downbeat 라벨이나 장비 간 latency calibration도 아닙니다.
- 음악을 재생하는 PortAudio 출력 장치의 DAC 시계로 동작 시간을 계산해 Python 시작 시각이나 `sleep` 누적으로 생기는 drift를 피합니다. 로봇 내부 지연은 존재합니다.
- 8개 motion primitive: sway, shoulder groove, wrist wave, diagonal, roll, counterwave, nod, open/close. 그리퍼 여닫기가 아니라 여러 관절의 반대 위상 움직임입니다.
- 8박자 구절 단위로 seed 기반 선택을 하며, 4박자 구간에 C2 smoothstep으로 패턴을 섞습니다. 전체 관절이 매 waypoint에서 멈추는 방식이 아닙니다. 시작/종료 시에는 점진적으로 움직임을 키우거나 줄입니다.
- 목표 40Hz single-level joint-speed 제어에 목표 속도 feed-forward와 작은 위치 피드백을 사용합니다. TCP로 속도 명령을, UDP로 cyclic feedback을 주고받습니다. 실제 갱신 주기는 네트워크와 PC에 따라 달라지며 로그에 기록됩니다. 도달 가능한 단위 동작의 BPM 배수로 움직이며 매 박자마다 급정지하지 않습니다.

## 관절 제한 및 정지 처리

이 로봇에서 앞선 Cartesian 왕복은 `JOINT_POSITION_LIMIT_REACHED`로 중단됐고, 당시 J5가 약 -120.32도였습니다. 공식 URDF의 보수적 J5 범위는 약 ±119.75도입니다. 기본 기준 자세를 약 -103.75도로 옮겨 여유를 확보했습니다. 기존 자세가 보수적 모델 범위를 2도 이상 벗어나면 자동 준비 동작도 거부합니다.

- 공식 Gen3 6-DOF spherical-wrist URDF에서 관절 각도 범위를 읽습니다. 다른 모델은 거부합니다.
- 시작 전 전체 동작을 100Hz로 계산하고 각도, 속도, 가속도, 팔의 이동 범위와 베이스 평면 접근을 검사합니다. 춤 목표는 유한 관절 범위에서 최소 8도, 기본 설정에서는 더 큰 여유를 둡니다.
- 로봇의 현재 속도/가속도 제한과 앱의 보수적 제한 중 작은 값을 사용합니다. 제한을 높이거나 비활성화하지 않습니다.
- 실시간 관절 피드백, 추적 오차, 제동 거리, tool 이동 범위, fault, 제어권 변경, 오디오 오류, 루프 지연을 검사합니다.
- 별도 프로세스의 watchdog이 제어 heartbeat를 감시하고 약 0.4초 이상 끊기면 별도 TCP 연결로 정지합니다. 두 연결이 모두 끊기는 경우에는 로봇 내장 연결 timeout에 의존하며 물리 비상정지가 필요합니다.
- 사용자 Stop/제어권 변경/fault 후 자동으로 춤을 재개하지 않습니다. `ClearFaults`, torque 제어, low-level servoing, homing, 안전 제한 변경은 호출하지 않습니다.

**충돌 검사 범위:** URDF 기반 관절/팔 이동 범위, 베이스 평면 및 손목–어깨 거리 검사입니다. 전체 mesh 기반 self-collision 또는 방 안의 장애물 지도를 사용하는 planner는 아닙니다. 따라서 주변 공간을 확보한 데모가 전제입니다. Windows/Python 제어 및 watchdog은 안전 인증 제어기가 아닙니다.

## 파일 및 결과

- `rita.py`: 분석/미리보기/실행/정지 CLI.
- `rita_robot/audio.py`: 오디오 분석, 음악 재생, 출력 시계.
- `rita_robot/motion.py`: 연속 동작 생성 및 offline 제약 검사.
- `rita_robot/robot.py`: Kortex 연결, 실시간 guard, 독립 watchdog.
- `../result/0_pre-result/kinova_trajectory.png`, `data/plan_report.json`: 생성한 궤적과 범위.
- `logs/*_summary.json`, `logs/*.jsonl`: 완료 여부, 오류, 관절 추적 및 음악 시간 기록. 데모 완료는 summary의 `completed: true`로 확인합니다.

## 공식 참고 자료

- [Kinova Kortex 공식 API와 예제](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L)
- [Kinova 공식 6-DOF 모델 URDF](https://github.com/Kinovarobotics/ros_kortex/blob/noetic-devel/kortex_description/arms/gen3/6dof/urdf/GEN3-6DOF_VISION_URDF_ARM_V01.urdf)
- [Kinova joint-speed 및 stop API](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L/blob/master/api_python/doc/markdown/summary_pages/Base.md)

`data/robot/`의 URDF와 YAML은 위 Kinova 공식 저장소에서 가져왔습니다. 해당 파일의 라이선스는 `data/robot/KINOVA_LICENSE`를 참고하세요.
