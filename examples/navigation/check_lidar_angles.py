"""
Check LIDAR angle distribution to understand coordinate system
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

def lidar_callback(message):
    """Analyze LIDAR angle distribution"""
    global obstacle_found

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
            return

        # Convert to polar coordinates
        x = points[:, 0]
        y = points[:, 1]
        ranges = np.sqrt(x**2 + y**2)
        angles = np.arctan2(y, x)

        # Analyze angle distribution in different sectors
        print(f"\n{'='*70}")
        print(f"📊 LIDAR ANGLE DISTRIBUTION ANALYSIS")
        print(f"{'='*70}")
        print(f"Total points (after height filter): {len(points):,}")
        print(f"\nAngle Statistics:")
        print(f"  Min angle: {np.degrees(np.min(angles)):>8.1f}°")
        print(f"  Max angle: {np.degrees(np.max(angles)):>8.1f}°")
        print(f"  Mean angle: {np.degrees(np.mean(angles)):>7.1f}°")

        # Count points in different angular sectors
        print(f"\n{'─'*70}")
        print(f"Points by Angle Sector (360° coverage):")
        print(f"{'─'*70}")

        sectors = [
            ("Front      (±10°)", -10, 10),
            ("Front-Right (10-80°)", 10, 80),
            ("Right      (80-100°)", 80, 100),
            ("Back-Right (100-170°)", 100, 170),
            ("Back       (170-180° & -170-180°)", 170, 180, -180, -170),
            ("Back-Left  (-170--100°)", -170, -100),
            ("Left       (-100--80°)", -100, -80),
            ("Front-Left (-80--10°)", -80, -10),
        ]

        for sector_info in sectors:
            if len(sector_info) == 3:
                name, min_deg, max_deg = sector_info
                mask = (np.degrees(angles) >= min_deg) & (np.degrees(angles) < max_deg)
            else:  # Back sector has two ranges
                name = sector_info[0]
                mask1 = (np.degrees(angles) >= sector_info[1]) & (np.degrees(angles) <= sector_info[2])
                mask2 = (np.degrees(angles) >= sector_info[3]) & (np.degrees(angles) <= sector_info[4])
                mask = mask1 | mask2

            count = np.sum(mask)
            percentage = (count / len(points)) * 100

            if count > 0:
                sector_ranges = ranges[mask]
                min_range = np.min(sector_ranges)
                max_range = np.max(sector_ranges)
                avg_range = np.mean(sector_ranges)
                print(f"  {name:25s}: {count:6,} points ({percentage:5.1f}%) | "
                      f"Range: {min_range:.2f}-{max_range:.2f}m (avg: {avg_range:.2f}m)")
            else:
                print(f"  {name:25s}: {count:6,} points ({percentage:5.1f}%)")

        # Narrow front analysis
        print(f"\n{'─'*70}")
        print(f"NARROW FRONT SECTORS (to identify robot body):")
        print(f"{'─'*70}")

        for angle_range in [1, 2, 5, 10, 15, 20]:
            mask = np.abs(np.degrees(angles)) < angle_range
            count = np.sum(mask)
            if count > 0:
                sector_ranges = ranges[mask]
                min_range = np.min(sector_ranges)
                max_range = np.max(sector_ranges)
                avg_range = np.mean(sector_ranges)
                print(f"  ±{angle_range:2d}°: {count:6,} points | "
                      f"Min: {min_range:.2f}m, Max: {max_range:.2f}m, Avg: {avg_range:.2f}m")

        print(f"{'='*70}\n")

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
        logging.info("Analyzing LIDAR data...")

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
