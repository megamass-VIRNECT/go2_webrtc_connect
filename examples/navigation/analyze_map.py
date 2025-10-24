"""
맵 파일 분석 스크립트
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# 맵 로드
with open('go2_map.pkl', 'rb') as f:
    occupancy_grid = pickle.load(f)

print("맵 정보:")
print(f"  크기: {occupancy_grid.width} x {occupancy_grid.height}")
print(f"  해상도: {occupancy_grid.resolution}m/cell")
print(f"  원점: {occupancy_grid.origin}")
print(f"  데이터 shape: {occupancy_grid.data.shape}")
print()

# 데이터 분석
data = occupancy_grid.data
print("점유 값 분포:")
print(f"  Unknown (-1): {np.sum(data == -1)} ({np.sum(data == -1) / data.size * 100:.1f}%)")
print(f"  Free (0-49): {np.sum((data >= 0) & (data < 50))} ({np.sum((data >= 0) & (data < 50)) / data.size * 100:.1f}%)")
print(f"  Occupied (50+): {np.sum(data >= 50)} ({np.sum(data >= 50) / data.size * 100:.1f}%)")
print()
print(f"데이터 범위: min={data.min()}, max={data.max()}")
print(f"평균: {data[data >= 0].mean():.2f} (unknown 제외)")
print()

# Occupied 셀의 위치 분석
occupied_mask = data >= 50
occupied_rows, occupied_cols = np.where(occupied_mask)

if len(occupied_rows) > 0:
    print("Occupied 셀 분포:")
    print(f"  개수: {len(occupied_rows)}")
    print(f"  Row 범위: [{occupied_rows.min()}, {occupied_rows.max()}]")
    print(f"  Col 범위: [{occupied_cols.min()}, {occupied_cols.max()}]")

    # World coordinates로 변환
    world_x = occupied_cols * occupancy_grid.resolution + occupancy_grid.origin[0]
    world_y = occupied_rows * occupancy_grid.resolution + occupancy_grid.origin[1]

    print(f"  World X 범위: [{world_x.min():.2f}, {world_x.max():.2f}]m")
    print(f"  World Y 범위: [{world_y.min():.2f}, {world_y.max():.2f}]m")
    print()

    # 원점으로부터의 거리 계산
    robot_x = -occupancy_grid.origin[0]
    robot_y = -occupancy_grid.origin[1]
    distances = np.sqrt((world_x - 0)**2 + (world_y - 0)**2)

    print(f"원점으로부터 거리:")
    print(f"  평균: {distances.mean():.2f}m")
    print(f"  중앙값: {np.median(distances):.2f}m")
    print(f"  최대: {distances.max():.2f}m")
    print()

    # 거리별 분포
    hist, bins = np.histogram(distances, bins=20)
    print("거리별 occupied 셀 분포:")
    for i in range(len(hist)):
        if hist[i] > 0:
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}m: {hist[i]} 셀")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 원본 맵
ax1 = axes[0]
map_image = np.zeros_like(data, dtype=np.uint8)
map_image[data < 0] = 128  # Unknown -> gray
map_image[data == 0] = 255  # Free -> white
map_image[data > 50] = 0  # Occupied -> black
mask = (data >= 0) & (data <= 50)
map_image[mask] = (255 - (data[mask] * 255 / 50)).astype(np.uint8)

ax1.imshow(map_image, cmap='gray', origin='lower')
ax1.set_title('Original Map')
ax1.set_xlabel('X (cells)')
ax1.set_ylabel('Y (cells)')

# Occupied only
ax2 = axes[1]
occupied_map = np.ones_like(data) * 255
occupied_map[data >= 50] = 0
ax2.imshow(occupied_map, cmap='gray', origin='lower')
ax2.set_title('Occupied Cells Only')
ax2.set_xlabel('X (cells)')
ax2.set_ylabel('Y (cells)')

# 거리 히트맵
ax3 = axes[2]
if len(occupied_rows) > 0:
    distance_map = np.full_like(data, -1.0, dtype=float)
    for i in range(len(occupied_rows)):
        distance_map[occupied_rows[i], occupied_cols[i]] = distances[i]

    im = ax3.imshow(distance_map, cmap='jet', origin='lower', vmin=0, vmax=5)
    plt.colorbar(im, ax=ax3, label='Distance (m)')
    ax3.set_title('Distance from Origin')
    ax3.set_xlabel('X (cells)')
    ax3.set_ylabel('Y (cells)')

plt.tight_layout()
plt.savefig('map_analysis.png', dpi=150, bbox_inches='tight')
print("\n분석 이미지 저장: map_analysis.png")
# plt.show()  # Non-blocking mode
