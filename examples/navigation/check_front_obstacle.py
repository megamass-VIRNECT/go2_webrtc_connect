"""
Quick script to check obstacle distance in front of the robot
"""

import asyncio
import logging
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

logging.basicConfig(level=logging.INFO)

obstacle_found = False
min_distance = None

def lidar_callback(message):
    """Process one LIDAR scan and find front obstacle"""
    global obstacle_found, min_distance

    try:
        data = message.get("data", {})
        data_inner = data.get("data", {})
        positions = data_inner.get("positions", [])

        if len(positions) == 0:
            return

        # Get voxel map parameters
        origin = data.get("origin", [0.0, 0.0, 0.0])
        width = data.get("width", [128, 128, 38])
        resolution = data.get("resolution", 0.05)

        # Convert voxel indices to 3D points
        voxel_indices = np.array([positions[i:i+3] for i in range(0, len(positions), 3)], dtype=np.float32)

        # Convert to world coordinates
        points = np.zeros_like(voxel_indices)
        points[:, 0] = origin[0] + (voxel_indices[:, 0] * resolution)
        points[:, 1] = origin[1] + (voxel_indices[:, 1] * resolution)
        points[:, 2] = origin[2] + (voxel_indices[:, 2] * resolution)

        # Calculate map center (robot position)
        map_center_x = origin[0] + (width[0] * resolution) / 2.0
        map_center_y = origin[1] + (width[1] * resolution) / 2.0

        # Convert to robot-centered coordinates
        points[:, 0] -= map_center_x
        points[:, 1] -= map_center_y

        # Filter by height
        height_min = -0.4
        height_max = 1.0
        height_mask = (points[:, 2] >= height_min) & (points[:, 2] <= height_max)
        points = points[height_mask]

        if len(points) == 0:
            logging.warning("No points after height filtering")
            return

        # Convert to polar coordinates
        x = points[:, 0]
        y = points[:, 1]
        ranges = np.sqrt(x**2 + y**2)
        angles = np.arctan2(y, x)

        # Filter by distance (remove robot body and very close noise)
        min_range = 1.0  # Minimum 1.0m (to exclude robot body)
        max_range = 10.0  # Maximum 10m
        range_mask = (ranges >= min_range) & (ranges <= max_range)
        ranges = ranges[range_mask]
        angles = angles[range_mask]

        # Find obstacles in front (narrowed to ±5 degrees for precise front detection)
        front_mask = np.abs(angles) < np.radians(5)
        front_ranges = ranges[front_mask]

        if len(front_ranges) > 0:
            min_dist = np.min(front_ranges)
            min_distance = min_dist
            print(f"\n{'='*60}")
            print(f"🎯 FRONT OBSTACLE DETECTED (±5°):")
            print(f"   Distance: {min_dist:.2f} meters")
            print(f"   Total front points: {len(front_ranges)}")
            print(f"   Range: [{np.min(front_ranges):.2f}m - {np.max(front_ranges):.2f}m]")
            print(f"{'='*60}\n")
            obstacle_found = True
        else:
            print("\n⚠️  No obstacles detected directly in front (±5°)")
            print(f"   Total points scanned: {len(points)}")
            print(f"   Angle range: [{np.degrees(np.min(angles)):.1f}° - {np.degrees(np.max(angles)):.1f}°]")
            obstacle_found = True

    except Exception as e:
        logging.error(f"Error processing LIDAR: {e}", exc_info=True)
        obstacle_found = True


async def main():
    try:
        # Connect to robot
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")

        logging.info("Connecting to Go2...")
        await conn.connect()
        logging.info("Connected!")

        # Disable traffic saving
        await conn.datachannel.disableTrafficSaving(True)

        # Set decoder
        conn.datachannel.set_decoder(decoder_type='libvoxel')

        # Turn on LIDAR
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
        logging.info("LIDAR turned on")

        # Wait a moment for LIDAR to start
        await asyncio.sleep(1)

        # Subscribe to LIDAR
        conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", lidar_callback)
        logging.info("Waiting for LIDAR data...")

        # Wait for callback to process data
        timeout = 10
        start_time = asyncio.get_event_loop().time()
        while not obstacle_found and (asyncio.get_event_loop().time() - start_time) < timeout:
            await asyncio.sleep(0.1)

        if not obstacle_found:
            print("\n⚠️  Timeout waiting for LIDAR data")

        # Turn off LIDAR
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "off")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
