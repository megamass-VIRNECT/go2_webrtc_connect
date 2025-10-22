"""Visualize map to debug display issue"""
import numpy as np
import matplotlib.pyplot as plt
from go2_webrtc_driver.navigation import load_map

# Load map
map_obj = load_map("go2_map.pkl")

print(f"Map size: {map_obj.width}x{map_obj.height}")
print(f"Resolution: {map_obj.resolution}m")
print(f"Origin: {map_obj.origin}")

# Create image same way as autonomous_navigation_amcl.py
image = np.ones_like(map_obj.data, dtype=np.uint8) * 128
image[map_obj.data == -1] = 128  # Unknown -> gray
mask_free = (map_obj.data >= 0) & (map_obj.data < 50)
image[mask_free] = 255  # Free -> white
image[map_obj.data >= 50] = 0  # Occupied -> black

print(f"\nImage stats:")
print(f"  Shape: {image.shape}")
print(f"  dtype: {image.dtype}")
print(f"  min: {image.min()}, max: {image.max()}")
print(f"  unique values: {np.unique(image)}")

# Count pixels
n_gray = np.sum(image == 128)
n_white = np.sum(image == 255)
n_black = np.sum(image == 0)
print(f"\nPixel counts:")
print(f"  Gray (128): {n_gray}")
print(f"  White (255): {n_white}")
print(f"  Black (0): {n_black}")

# Display with same settings
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Method 1: Same as autonomous_navigation_amcl.py
extent = [
    map_obj.origin[1],
    map_obj.origin[1] + map_obj.width * map_obj.resolution,
    map_obj.origin[0],
    map_obj.origin[0] + map_obj.height * map_obj.resolution
]
ax1.imshow(image, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255, alpha=0.8)
ax1.set_xlabel('Y (meters)')
ax1.set_ylabel('X (meters)')
ax1.set_title('Map Display (alpha=0.8)')
ax1.grid(True, alpha=0.3)

# Method 2: Without alpha to see if that's the issue
ax2.imshow(image, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255)
ax2.set_xlabel('Y (meters)')
ax2.set_ylabel('X (meters)')
ax2.set_title('Map Display (alpha=1.0)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('map_debug.png', dpi=150)
print("\nSaved to map_debug.png")
plt.show()
