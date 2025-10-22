"""
Simple script to view saved SLAM map
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load map
with open('go2_map.pkl', 'rb') as f:
    occupancy_grid = pickle.load(f)

print(f"Map size: {occupancy_grid.width} x {occupancy_grid.height}")
print(f"Resolution: {occupancy_grid.resolution}m/cell")
print(f"Origin: {occupancy_grid.origin}")

# Get map data
map_data = occupancy_grid.data
print(f"Data range: min={map_data.min()}, max={map_data.max()}, mean={map_data.mean():.2f}")
print(f"Unknown cells: {np.sum(map_data == -1)}")
print(f"Free cells: {np.sum((map_data >= 0) & (map_data < 50))}")
print(f"Occupied cells: {np.sum(map_data >= 50)}")

# Create image
image = np.ones_like(map_data, dtype=np.uint8) * 128  # Gray for unknown

# Unknown cells (-1) -> gray (128)
image[map_data == -1] = 128

# Free cells (0-49) -> white (255)
mask_free = (map_data >= 0) & (map_data < 50)
image[mask_free] = 255

# Occupied cells (50-100) -> black (0)
image[map_data >= 50] = 0

print(f"\nImage stats:")
print(f"Black pixels (occupied): {np.sum(image == 0)}")
print(f"Gray pixels (unknown): {np.sum(image == 128)}")
print(f"White pixels (free): {np.sum(image == 255)}")

# Plot
plt.figure(figsize=(12, 12))
extent = [
    occupancy_grid.origin[1],
    occupancy_grid.origin[1] + occupancy_grid.width * occupancy_grid.resolution,
    occupancy_grid.origin[0],
    occupancy_grid.origin[0] + occupancy_grid.height * occupancy_grid.resolution
]
plt.imshow(image, cmap='gray', origin='lower', extent=extent)
plt.xlabel('Y (meters)')
plt.ylabel('X (meters)')
plt.title('SLAM Map')
plt.colorbar(label='0=Occupied, 128=Unknown, 255=Free')
plt.grid(True, alpha=0.3)
plt.savefig('map_visualization.png', dpi=150, bbox_inches='tight')
print("Map saved to map_visualization.png")
plt.show()
