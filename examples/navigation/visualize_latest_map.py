"""
최신 맵 시각화 및 분석
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# 맵 로드
print("맵 로드 중...")
with open('go2_map.pkl', 'rb') as f:
    occupancy_grid = pickle.load(f)

print(f"맵 크기: {occupancy_grid.width} x {occupancy_grid.height}")
print(f"해상도: {occupancy_grid.resolution}m")
print(f"원점: {occupancy_grid.origin}")
print()

# 데이터 분석
data = occupancy_grid.data
print("점유 분포:")
print(f"  Unknown (-1): {np.sum(data == -1)} ({np.sum(data == -1) / data.size * 100:.1f}%)")
print(f"  Free (0-49): {np.sum((data >= 0) & (data < 50))} ({np.sum((data >= 0) & (data < 50)) / data.size * 100:.1f}%)")
print(f"  Occupied (50+): {np.sum(data >= 50)} ({np.sum(data >= 50) / data.size * 100:.1f}%)")
print()

# Occupied 셀 위치
occupied_mask = data >= 50
occupied_rows, occupied_cols = np.where(occupied_mask)

if len(occupied_rows) > 0:
    print(f"Occupied 셀: {len(occupied_rows)}개")

    # World coordinates
    world_x = occupied_cols * occupancy_grid.resolution + occupancy_grid.origin[0]
    world_y = occupied_rows * occupancy_grid.resolution + occupancy_grid.origin[1]

    print(f"World X 범위: [{world_x.min():.2f}, {world_x.max():.2f}]m")
    print(f"World Y 범위: [{world_y.min():.2f}, {world_y.max():.2f}]m")

    # 원점으로부터 거리
    distances = np.sqrt(world_x**2 + world_y**2)
    print(f"원점으로부터 거리: 평균 {distances.mean():.2f}m, 최대 {distances.max():.2f}m")
    print()

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 전체 맵
ax1 = axes[0]
map_image = np.zeros_like(data, dtype=np.uint8)
map_image[data < 0] = 128  # Unknown
map_image[data == 0] = 255  # Free
map_image[data > 50] = 0  # Occupied
mask = (data >= 0) & (data <= 50)
map_image[mask] = (255 - (data[mask] * 255 / 50)).astype(np.uint8)

extent = [
    occupancy_grid.origin[0],
    occupancy_grid.origin[0] + occupancy_grid.width * occupancy_grid.resolution,
    occupancy_grid.origin[1],
    occupancy_grid.origin[1] + occupancy_grid.height * occupancy_grid.resolution
]

im1 = ax1.imshow(map_image, cmap='gray', origin='lower', extent=extent)
ax1.set_xlabel('X (meters)')
ax1.set_ylabel('Y (meters)')
ax1.set_title('Full Map')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Origin')
ax1.axvline(x=0, color='r', linestyle='--', alpha=0.3)
ax1.plot(0, 0, 'ro', markersize=10, label='Origin')
ax1.legend()

# Occupied만 표시 (zoom)
ax2 = axes[1]
if len(occupied_rows) > 0:
    # Occupied 영역 주변만 표시
    x_min, x_max = world_x.min() - 1, world_x.max() + 1
    y_min, y_max = world_y.min() - 1, world_y.max() + 1

    # 해당 영역의 맵 추출
    row_min = max(0, int((y_min - occupancy_grid.origin[1]) / occupancy_grid.resolution))
    row_max = min(occupancy_grid.height, int((y_max - occupancy_grid.origin[1]) / occupancy_grid.resolution))
    col_min = max(0, int((x_min - occupancy_grid.origin[0]) / occupancy_grid.resolution))
    col_max = min(occupancy_grid.width, int((x_max - occupancy_grid.origin[0]) / occupancy_grid.resolution))

    zoomed_data = data[row_min:row_max, col_min:col_max]

    # 이미지 생성
    zoomed_image = np.zeros_like(zoomed_data, dtype=np.uint8)
    zoomed_image[zoomed_data < 0] = 128
    zoomed_image[zoomed_data == 0] = 255
    zoomed_image[zoomed_data > 50] = 0
    mask_zoom = (zoomed_data >= 0) & (zoomed_data <= 50)
    zoomed_image[mask_zoom] = (255 - (zoomed_data[mask_zoom] * 255 / 50)).astype(np.uint8)

    zoomed_extent = [x_min, x_max, y_min, y_max]

    im2 = ax2.imshow(zoomed_image, cmap='gray', origin='lower', extent=zoomed_extent)
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title('Zoomed Occupied Region')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    ax2.plot(0, 0, 'ro', markersize=8, label='Origin')
    ax2.legend()

    # Occupied 셀의 값 분포
    occupied_values = data[occupied_mask]
    print("Occupied 셀 값 분포:")
    print(f"  최소: {occupied_values.min()}")
    print(f"  최대: {occupied_values.max()}")
    print(f"  평균: {occupied_values.mean():.1f}")
    print(f"  50-69: {np.sum((occupied_values >= 50) & (occupied_values < 70))}개")
    print(f"  70-89: {np.sum((occupied_values >= 70) & (occupied_values < 90))}개")
    print(f"  90-100: {np.sum(occupied_values >= 90)}개")

plt.tight_layout()
plt.savefig('latest_map_visualization.png', dpi=150, bbox_inches='tight')
print("\n이미지 저장: latest_map_visualization.png")
