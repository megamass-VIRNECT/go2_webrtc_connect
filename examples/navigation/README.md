# Go2 Navigation Examples

AMCL (Adaptive Monte Carlo Localization) 기반 매핑 및 로컬라이제이션 예제입니다.

## 개요

이 패키지는 Unitree Go2 로봇을 위한 자율주행 기능을 제공합니다:

- **SLAM Mapping**: Lidar 데이터를 사용한 Occupancy Grid 맵 생성
- **AMCL Localization**: 파티클 필터 기반 로봇 위치 추정
- **Navigation**: 맵 기반 경로 계획 및 추종 (향후 구현 예정)

## 설치

필요한 의존성 패키지:

```bash
pip install numpy matplotlib
```

## 사용 방법

### 1. SLAM 매핑

먼저 로봇을 움직이면서 환경의 맵을 생성합니다:

```bash
python slam_mapping.py
```

**주요 기능:**
- Lidar 스캔을 사용하여 Occupancy Grid 맵 생성
- 로봇의 오도메트리로 위치 추적
- 맵을 `go2_map.pkl` 파일로 저장
- 시각화 이미지를 `slam_map.png`로 저장

**파라미터:**
- 기본 실행 시간: 5분 (300초)
- 맵 크기: 100m x 100m
- 해상도: 5cm/cell
- Ctrl+C로 조기 중단 가능

### 2. AMCL 로컬라이제이션

생성된 맵에서 로봇의 위치를 추정합니다:

```bash
# 초기 위치를 모르는 경우 (전역 로컬라이제이션)
python amcl_localization.py --map go2_map.pkl

# 초기 위치를 아는 경우
python amcl_localization.py --map go2_map.pkl --init-x 0.0 --init-y 0.0 --init-theta 0.0

# 다른 IP 주소 사용
python amcl_localization.py --map go2_map.pkl --ip 192.168.1.100
```

**주요 기능:**
- 파티클 필터를 사용한 로봇 위치 추정
- 오도메트리와 Lidar 스캔 융합
- 실시간 위치 불확실성 계산
- Adaptive resampling으로 kidnapping problem 해결

**파라미터:**
- `--map`: 맵 파일 경로 (기본값: go2_map.pkl)
- `--init-x`: 초기 x 위치 (미터)
- `--init-y`: 초기 y 위치 (미터)
- `--init-theta`: 초기 각도 (라디안)
- `--ip`: 로봇 IP 주소 (기본값: 192.168.1.7)

## 알고리즘 설명

### SLAM (Simultaneous Localization and Mapping)

1. **Occupancy Grid Mapping**: 2D 그리드 맵에서 각 셀의 점유 확률을 추적
2. **Log-odds Update**: 베이지안 업데이트를 사용한 효율적인 맵 갱신
3. **Bresenham Ray Tracing**: Lidar 광선을 따라 자유 공간과 장애물 표시

### AMCL (Adaptive Monte Carlo Localization)

1. **Particle Filter**: 여러 파티클로 위치 가설 표현
2. **Motion Model**: 오도메트리 기반 파티클 전파
3. **Sensor Model**: Lidar 스캔으로 파티클 가중치 계산
4. **Resampling**: Low-variance resampling으로 효율적인 파티클 분포 유지
5. **Adaptive Sampling**: 로봇 납치 문제 해결을 위한 적응형 샘플링

## 파일 구조

```
examples/navigation/
├── README.md                  # 이 파일
├── slam_mapping.py           # SLAM 매핑 예제
└── amcl_localization.py      # AMCL 로컬라이제이션 예제

go2_webrtc_driver/navigation/
├── __init__.py
├── amcl/
│   ├── __init__.py
│   ├── particle_filter.py    # 파티클 필터 구현
│   ├── motion_model.py        # 모션 모델
│   └── sensor_model.py        # 센서 모델
├── mapping/
│   ├── __init__.py
│   └── slam.py               # SLAM 구현
└── utils/
    ├── __init__.py
    ├── map_utils.py          # 맵 유틸리티
    └── transforms.py         # 좌표 변환
```

## 튜닝 가이드

### SLAM 파라미터 조정

`slam_mapping.py`에서 `SLAMParams` 수정:

```python
slam_params = SLAMParams(
    map_width=2000,              # 맵 너비 (셀)
    map_height=2000,             # 맵 높이 (셀)
    map_resolution=0.05,         # 해상도 (미터/셀)
    log_odds_occupied=0.7,       # 점유 셀 업데이트 값
    log_odds_free=-0.4,          # 자유 공간 업데이트 값
    max_range=10.0,              # 최대 센서 범위 (미터)
)
```

### AMCL 파라미터 조정

`amcl_localization.py`에서 파라미터 수정:

**Motion Model:**
```python
motion_params = MotionModelParams(
    alpha1=0.2,  # 회전에서의 회전 노이즈
    alpha2=0.2,  # 이동에서의 회전 노이즈
    alpha3=0.2,  # 이동에서의 이동 노이즈
    alpha4=0.2   # 회전에서의 이동 노이즈
)
```

**Sensor Model:**
```python
sensor_params = SensorModelParams(
    z_hit=0.95,      # 정확한 측정 가중치
    z_rand=0.05,     # 랜덤 측정 가중치
    sigma_hit=0.2,   # 측정 노이즈 표준편차
    max_range=10.0,  # 최대 센서 범위
    max_beams=50     # 사용할 광선 수
)
```

## 문제 해결

### 맵이 제대로 생성되지 않는 경우

1. 로봇이 충분히 움직였는지 확인
2. Lidar가 켜져 있고 데이터가 수신되는지 확인
3. 오도메트리 데이터가 정확한지 확인

### 로컬라이제이션 성능이 낮은 경우

1. 파티클 수 증가 (`num_particles=1000`)
2. 센서 모델 파라미터 조정
3. 초기 위치를 정확히 지정
4. 맵의 품질 확인

## 향후 계획

- [ ] Global path planning (A*, Dijkstra)
- [ ] Local path planning (DWA, TEB)
- [ ] Navigation stack 통합
- [ ] ROS2 인터페이스
- [ ] 실시간 맵 시각화

## 참고 자료

- Probabilistic Robotics (Thrun, Burgard, Fox)
- ROS Navigation Stack
- AMCL ROS Package Documentation
