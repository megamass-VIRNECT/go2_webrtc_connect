"""
1050번 맵 시각화
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('go2_map_backup_1050.pkl', 'rb') as f:
    grid = pickle.load(f)

data = grid.data

# 이미지 생성
image = np.zeros_like(data, dtype=np.uint8)
image[data < 0] = 128  # Unknown
image[data == 0] = 255  # Free
image[data > 50] = 0  # Occupied
mask = (data >= 0) & (data <= 50)
image[mask] = (255 - (data[mask] * 255 / 50)).astype(np.uint8)

# Occupied 영역 찾기
occupied_mask = data >= 50
if np.sum(occupied_mask) > 0:
    occupied_rows, occupied_cols = np.where(occupied_mask)
    world_x = occupied_cols * grid.resolution + grid.origin[0]
    world_y = occupied_rows * grid.resolution + grid.origin[1]

    x_min, x_max = world_x.min() - 2, world_x.max() + 2
    y_min, y_max = world_y.min() - 2, world_y.max() + 2

    print(f"Occupied 영역: X=[{world_x.min():.2f}, {world_x.max():.2f}], Y=[{world_y.min():.2f}, {world_y.max():.2f}]")
else:
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5

# 전체 맵
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

ax1 = axes[0]
extent = [grid.origin[0], grid.origin[0] + grid.width * grid.resolution,
          grid.origin[1], grid.origin[1] + grid.height * grid.resolution]
ax1.imshow(image, cmap='gray', origin='lower', extent=extent)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Full Map (1050)')
ax1.grid(True, alpha=0.3)
ax1.plot(0, 0, 'ro', markersize=8, label='Origin')
ax1.legend()

# Zoom
ax2 = axes[1]
ax2.imshow(image, cmap='gray', origin='lower', extent=extent)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('Zoomed Occupied Region')
ax2.grid(True, alpha=0.3)
ax2.plot(0, 0, 'ro', markersize=8, label='Origin')
ax2.legend()

plt.tight_layout()
plt.savefig('map_1050_visualization.png', dpi=150, bbox_inches='tight')
print("이미지 저장: map_1050_visualization.png")
