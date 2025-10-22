"""
Debug script to understand lidar coordinate system
"""

import asyncio
import logging
import numpy as np
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

logging.basicConfig(level=logging.INFO)

scan_count = 0
odom_data = None

def odom_callback(message):
    """Store odometry data"""
    global odom_data
    data = message.get("data", {})
    position = data.get("position", [0.0, 0.0, 0.0])
    imu_state = data.get("imu_state", {})
    rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])
    odom_data = {
        'position': position,
        'yaw': rpy[2]
    }

def lidar_callback(message):
    """Analyze lidar data"""
    global scan_count, odom_data

    if scan_count >= 5:
        return

    scan_count += 1

    data = message.get("data", {})
    data_inner = data.get("data", {})
    positions = data_inner.get("positions", [])

    if positions is None or (hasattr(positions, '__len__') and len(positions) == 0):
        return

    if not isinstance(positions, np.ndarray):
        positions = np.array(positions, dtype=np.float32)

    # Get voxel map parameters
    origin = data.get("origin", [0.0, 0.0, 0.0])
    width = data.get("width", [128, 128, 38])
    resolution = data.get("resolution", 0.05)

    # CRITICAL: positions are voxel INDICES, not world coordinates!
    voxel_indices = np.array([positions[i:i+3] for i in range(0, len(positions), 3)], dtype=np.float32)

    # Convert voxel indices to world coordinates
    points = np.zeros_like(voxel_indices)
    points[:, 0] = origin[0] + (voxel_indices[:, 0] * resolution)
    points[:, 1] = origin[1] + (voxel_indices[:, 1] * resolution)
    points[:, 2] = origin[2] + (voxel_indices[:, 2] * resolution)

    logging.info(f"\n{'='*60}")
    logging.info(f"SCAN {scan_count}")
    logging.info(f"{'='*60}")

    if odom_data:
        logging.info(f"\nOdometry:")
        logging.info(f"  Position: {odom_data['position']}")
        logging.info(f"  Yaw: {odom_data['yaw']:.3f} rad ({np.degrees(odom_data['yaw']):.1f} deg)")

    logging.info(f"\nVoxel Map Parameters:")
    logging.info(f"  Origin: {origin}")
    logging.info(f"  Width (cells): {width}")
    logging.info(f"  Resolution: {resolution}m/cell")

    map_size_x = width[0] * resolution
    map_size_y = width[1] * resolution
    map_center_x = origin[0] + map_size_x / 2.0
    map_center_y = origin[1] + map_size_y / 2.0

    logging.info(f"  Map size: {map_size_x:.2f}m x {map_size_y:.2f}m")
    logging.info(f"  Map center: ({map_center_x:.2f}, {map_center_y:.2f})")

    logging.info(f"\nVoxel Indices:")
    logging.info(f"  Total voxels: {len(voxel_indices)}")
    logging.info(f"  X range: [{voxel_indices[:,0].min():.0f}, {voxel_indices[:,0].max():.0f}]")
    logging.info(f"  Y range: [{voxel_indices[:,1].min():.0f}, {voxel_indices[:,1].max():.0f}]")
    logging.info(f"  Z range: [{voxel_indices[:,2].min():.0f}, {voxel_indices[:,2].max():.0f}]")

    logging.info(f"\nWorld Coordinates (after conversion):")
    logging.info(f"  Total points: {len(points)}")
    logging.info(f"  X range: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]m")
    logging.info(f"  Y range: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]m")
    logging.info(f"  Z range: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]m")

    # Show some sample points
    logging.info(f"\nSample points (first 5):")
    for i in range(min(5, len(points))):
        logging.info(f"  Voxel {voxel_indices[i]} -> World {points[i]}")

    # Convert to robot-centered
    points_robot = points.copy()
    points_robot[:, 0] -= map_center_x
    points_robot[:, 1] -= map_center_y

    logging.info(f"\nRobot-centered points:")
    logging.info(f"  X range: [{points_robot[:,0].min():.2f}, {points_robot[:,0].max():.2f}]")
    logging.info(f"  Y range: [{points_robot[:,1].min():.2f}, {points_robot[:,1].max():.2f}]")

    # Compute ranges and angles
    ranges = np.sqrt(points_robot[:, 0]**2 + points_robot[:, 1]**2)
    angles = np.arctan2(points_robot[:, 1], points_robot[:, 0])

    logging.info(f"\nPolar coordinates (robot frame):")
    logging.info(f"  Range: [{ranges.min():.2f}, {ranges.max():.2f}]m")
    logging.info(f"  Angle: [{np.degrees(angles.min()):.1f}, {np.degrees(angles.max()):.1f}] deg")

    # Show points in different directions
    logging.info(f"\nSample points by direction:")
    for angle_deg in [0, 90, 180, -90]:
        angle_rad = np.radians(angle_deg)
        # Find point closest to this angle
        angle_diffs = np.abs(angles - angle_rad)
        closest_idx = np.argmin(angle_diffs)
        logging.info(f"  ~{angle_deg}° (fwd={angle_deg==0}, left={angle_deg==90}, back={angle_deg==180}, right={angle_deg==-90}):")
        logging.info(f"    Range: {ranges[closest_idx]:.2f}m, Angle: {np.degrees(angles[closest_idx]):.1f}°")
        logging.info(f"    Robot coords: ({points_robot[closest_idx, 0]:.2f}, {points_robot[closest_idx, 1]:.2f})")

async def main():
    try:
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")

        logging.info("Connecting to Go2...")
        await conn.connect()
        logging.info("Connected!")

        await conn.datachannel.disableTrafficSaving(True)
        conn.datachannel.set_decoder(decoder_type='libvoxel')

        # Subscribe to odometry
        conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", odom_callback)

        # Turn on and subscribe to lidar
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
        conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", lidar_callback)

        logging.info("\nWaiting for 5 scans...")

        while scan_count < 5:
            await asyncio.sleep(1)

        logging.info("\n\nDone! Collected 5 scans for analysis.")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
