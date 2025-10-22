"""Check map data"""
import numpy as np
from go2_webrtc_driver.navigation import load_map

# Load map
map_obj = load_map("go2_map.pkl")

print(f"Map size: {map_obj.width}x{map_obj.height}")
print(f"Resolution: {map_obj.resolution}m")
print(f"Origin: {map_obj.origin}")
print(f"Data shape: {map_obj.data.shape}")
print(f"Data type: {map_obj.data.dtype}")
print(f"Data range: min={map_obj.data.min()}, max={map_obj.data.max()}")
print(f"Unique values: {np.unique(map_obj.data)}")

# Count cell types
unknown = np.sum(map_obj.data == -1)
free = np.sum((map_obj.data >= 0) & (map_obj.data < 50))
occupied = np.sum(map_obj.data >= 50)

print(f"\nCell counts:")
print(f"  Unknown: {unknown} ({unknown/map_obj.data.size*100:.1f}%)")
print(f"  Free: {free} ({free/map_obj.data.size*100:.1f}%)")
print(f"  Occupied: {occupied} ({occupied/map_obj.data.size*100:.1f}%)")
