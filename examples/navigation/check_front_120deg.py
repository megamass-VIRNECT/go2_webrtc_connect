"""
Check obstacles in robot's front 120 degree sector using LIDAR + IMU
"""

import asyncio
import logging
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

logging.basicConfig(level=logging.INFO)

# Global state
robot_yaw = None  # Current robot heading from IMU
lidar_received = False
imu_received = False

def sportmode_callback(message):
    """Get robot orientation from IMU"""
    global robot_yaw, imu_received

    try:
        data = message.get("data", {})
        imu_state = data.get("imu_state", {})
        rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])

        robot_yaw = rpy[2]  # Yaw angle in radians
        imu_received = True

    except Exception as e:
        logging.error(f"Error in sportmode callback: {e}")


def lidar_callback(message):
    """Process LIDAR data with front 120° filter"""
    global robot_yaw, lidar_received

    try:
        if robot_yaw is None:
            logging.warning("Waiting for IMU data...")
            return

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

        # Convert to polar coordinates (relative to robot body)
        x = points[:, 0]
        y = points[:, 1]
        ranges = np.sqrt(x**2 + y**2)
        # Angles relative to robot's X-axis (0° = robot forward direction)
        angles_relative = np.arctan2(y, x)

        # Filter by distance (minimal filtering - detect all objects in front)
        min_range = 0.0  # No minimum distance - detect everything
        max_range = 10.0
        range_mask = (ranges >= min_range) & (ranges <= max_range)
        ranges = ranges[range_mask]
        angles_relative = angles_relative[range_mask]

        if len(ranges) == 0:
            logging.warning("No points after range filtering")
            return

        # Front 120° sector: ±60° from robot's forward direction
        # Since angles_relative is already relative to robot's X-axis (forward),
        # we just need to filter ±60°
        front_angle = np.radians(60)  # 60 degrees on each side = 120° total
        front_mask = np.abs(angles_relative) <= front_angle

        front_ranges = ranges[front_mask]
        front_angles = angles_relative[front_mask]

        # Display results
        print(f"\n{'='*70}")
        print(f"🎯 ROBOT FRONT 120° SECTOR ANALYSIS")
        print(f"{'='*70}")
        print(f"Robot Heading (Yaw): {np.degrees(robot_yaw):>7.1f}°")
        print(f"Total points (after filters): {len(ranges):,}")
        print(f"Front 120° points: {len(front_ranges):,} ({100*len(front_ranges)/len(ranges):.1f}%)")

        if len(front_ranges) > 0:
            min_dist = np.min(front_ranges)
            max_dist = np.max(front_ranges)
            avg_dist = np.mean(front_ranges)

            # Find closest obstacle angle
            closest_idx = np.argmin(front_ranges)
            closest_angle = front_angles[closest_idx]

            print(f"\nObstacle Statistics:")
            print(f"  Closest obstacle: {min_dist:.2f}m at {np.degrees(closest_angle):+.1f}°")
            print(f"  Farthest: {max_dist:.2f}m")
            print(f"  Average: {avg_dist:.2f}m")

            # Distance distribution
            print(f"\nDistance Distribution:")
            bins = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
            for i in range(len(bins)-1):
                count = np.sum((front_ranges >= bins[i]) & (front_ranges < bins[i+1]))
                if count > 0:
                    pct = 100 * count / len(front_ranges)
                    print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}m: {count:5,} points ({pct:5.1f}%)")

            # Angular distribution within front sector
            print(f"\nAngular Distribution (within 120° front):")
            angle_bins = [
                ("Left  (40-60°)", 40, 60),
                ("Left  (20-40°)", 20, 40),
                ("Front ( 0-20°)", 0, 20),
                ("Center (±10°)", -10, 10),
                ("Front (-20-0°)", -20, 0),
                ("Right (-40--20°)", -40, -20),
                ("Right (-60--40°)", -60, -40),
            ]

            for name, min_deg, max_deg in angle_bins:
                mask = (np.degrees(front_angles) >= min_deg) & (np.degrees(front_angles) < max_deg)
                count = np.sum(mask)
                if count > 0:
                    sector_ranges = front_ranges[mask]
                    min_r = np.min(sector_ranges)
                    avg_r = np.mean(sector_ranges)
                    print(f"  {name}: {count:5,} pts | min: {min_r:.2f}m, avg: {avg_r:.2f}m")
        else:
            print(f"\n⚠️  No obstacles detected in front 120° sector")

        print(f"{'='*70}\n")

        lidar_received = True

    except Exception as e:
        logging.error(f"Error processing LIDAR: {e}", exc_info=True)
        lidar_received = True


async def main():
    global lidar_received, imu_received

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

        # Subscribe to sport mode for IMU data
        conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", sportmode_callback)
        logging.info("Subscribed to IMU data")

        # Wait for IMU data
        await asyncio.sleep(0.5)

        # Turn on LIDAR
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
        logging.info("LIDAR turned on")

        # Wait a moment for LIDAR to start
        await asyncio.sleep(1)

        # Subscribe to LIDAR
        conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", lidar_callback)
        logging.info("Analyzing front 120° sector...")

        # Wait for callback to process data
        timeout = 10
        start_time = asyncio.get_event_loop().time()
        while not (lidar_received and imu_received) and (asyncio.get_event_loop().time() - start_time) < timeout:
            await asyncio.sleep(0.1)

        if not imu_received:
            print("\n⚠️  Failed to receive IMU data")
        if not lidar_received:
            print("\n⚠️  Failed to receive LIDAR data")

        # Turn off LIDAR
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "off")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
