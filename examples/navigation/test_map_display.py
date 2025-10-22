"""Test map display with new settings"""
import numpy as np
import matplotlib.pyplot as plt
from go2_webrtc_driver.navigation import load_map

# Load map
map_obj = load_map("go2_map.pkl")

# Create image with NEW settings (lighter gray)
image = np.ones_like(map_obj.data, dtype=np.uint8) * 205  # Lighter gray
image[map_obj.data == -1] = 205  # Unknown -> light gray
mask_free = (map_obj.data >= 0) & (map_obj.data < 50)
image[mask_free] = 255  # Free -> white
image[map_obj.data >= 50] = 0  # Occupied -> black

print(f"Map size: {map_obj.width}x{map_obj.height}")
print(f"Image unique values: {np.unique(image)}")

# Display
fig, ax = plt.subplots(figsize=(12, 12))
extent = [
    map_obj.origin[1],
    map_obj.origin[1] + map_obj.width * map_obj.resolution,
    map_obj.origin[0],
    map_obj.origin[0] + map_obj.height * map_obj.resolution
]

ax.imshow(image, cmap='gray', origin='lower', extent=extent, vmin=0, vmax=255, alpha=1.0)
ax.set_xlabel('Y (meters)')
ax.set_ylabel('X (meters)')
ax.set_title('Map Display with Fixed Settings (unknown=205, alpha=1.0)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('map_fixed.png', dpi=150)
print("Saved to map_fixed.png")
print("\nIf you can see the map structure clearly, the fix worked!")
plt.show()
