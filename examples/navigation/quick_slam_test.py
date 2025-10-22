"""
Quick SLAM test - 30 seconds mapping
Run this while moving the robot to create a proper map!
"""

import asyncio
import logging
import sys
import numpy as np
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.navigation import SLAM, SLAMParams

logging.basicConfig(level=logging.INFO)

class QuickSLAMTest:
    def __init__(self):
        slam_params = SLAMParams(
            map_width=2000,
            map_height=2000,
            map_resolution=0.1,
            map_origin=(-100.0, -100.0),
            max_range=50.0,
            min_range=0.5
        )
        self.slam = SLAM(slam_params)
        self.current_pose = (0.0, 0.0, 0.0)
        self.prev_odom = None
        self.scan_count = 0

    def sportmodestate_callback(self, message):
        try:
            data = message.get("data", {})
            position = data.get("position", [0.0, 0.0, 0.0])
            imu_state = data.get("imu_state", {})
            rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])
            current_odom = (position[0], position[1], rpy[2])

            if self.prev_odom is not None:
                dx_world = current_odom[0] - self.prev_odom[0]
                dy_world = current_odom[1] - self.prev_odom[1]
                dtheta = current_odom[2] - self.prev_odom[2]

                while dtheta > np.pi:
                    dtheta -= 2 * np.pi
                while dtheta < -np.pi:
                    dtheta += 2 * np.pi

                if abs(dx_world) > 0.01 or abs(dy_world) > 0.01 or abs(dtheta) > 0.01:
                    x, y, theta = self.current_pose
                    x += dx_world
                    y += dy_world
                    theta += dtheta
                    theta = np.arctan2(np.sin(theta), np.cos(theta))
                    self.current_pose = (x, y, theta)
                    logging.info(f"Robot moved to: x={x:.3f}, y={y:.3f}, theta={np.degrees(theta):.1f}°")

            self.prev_odom = current_odom

        except Exception as e:
            logging.error(f"Error in odom callback: {e}")

    def lidar_callback(self, message):
        try:
            data = message.get("data", {})
            data_inner = data.get("data", {})
            positions = data_inner.get("positions", [])

            if positions is None or (hasattr(positions, '__len__') and len(positions) == 0):
                return

            if not isinstance(positions, np.ndarray):
                positions = np.array(positions, dtype=np.float32)

            origin = data.get("origin", [0.0, 0.0, 0.0])
            width = data.get("width", [128, 128, 38])
            resolution = data.get("resolution", 0.05)

            # Convert voxel indices to world coordinates (FIXED!)
            voxel_indices = np.array([positions[i:i+3] for i in range(0, len(positions), 3)], dtype=np.float32)
            points = np.zeros_like(voxel_indices)
            points[:, 0] = origin[0] + (voxel_indices[:, 0] * resolution)
            points[:, 1] = origin[1] + (voxel_indices[:, 1] * resolution)
            points[:, 2] = origin[2] + (voxel_indices[:, 2] * resolution)

            # Convert to robot-centered
            map_center_x = origin[0] + (width[0] * resolution) / 2.0
            map_center_y = origin[1] + (width[1] * resolution) / 2.0
            points[:, 0] -= map_center_x
            points[:, 1] -= map_center_y

            # Filter by range
            ranges_temp = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            valid_mask = (ranges_temp >= self.slam.params.min_range) & (ranges_temp <= self.slam.params.max_range)
            points_filtered = points[valid_mask]

            if len(points_filtered) == 0:
                return

            # Downsample
            downsample_factor = max(1, len(points_filtered) // 500)
            points_filtered = points_filtered[::downsample_factor]

            # Convert to polar
            x = points_filtered[:, 0]
            y = points_filtered[:, 1]
            ranges = np.sqrt(x**2 + y**2).tolist()
            angles = np.arctan2(y, x).tolist()

            # Update SLAM
            self.slam.update(self.current_pose, ranges, angles)
            self.scan_count += 1

            if self.scan_count % 10 == 0:
                logging.info(f"Scan {self.scan_count}: {len(ranges)} points, pose=({self.current_pose[0]:.2f}, {self.current_pose[1]:.2f}, {np.degrees(self.current_pose[2]):.1f}°)")

        except Exception as e:
            logging.error(f"Error in lidar callback: {e}", exc_info=True)


async def main():
    try:
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")

        logging.info("Connecting to Go2...")
        await conn.connect()
        logging.info("Connected!")

        await conn.datachannel.disableTrafficSaving(True)
        conn.datachannel.set_decoder(decoder_type='libvoxel')

        mapper = QuickSLAMTest()

        conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", mapper.sportmodestate_callback)
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
        conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", mapper.lidar_callback)

        logging.info("\n" + "="*60)
        logging.info("SLAM TEST - 30 seconds")
        logging.info("MOVE THE ROBOT NOW to create a proper map!")
        logging.info("="*60 + "\n")

        await asyncio.sleep(30)

        mapper.slam.save_map("go2_map_test.pkl")
        logging.info(f"\n{'='*60}")
        logging.info(f"Test complete!")
        logging.info(f"Total scans: {mapper.scan_count}")
        logging.info(f"Final pose: ({mapper.current_pose[0]:.2f}, {mapper.current_pose[1]:.2f}, {np.degrees(mapper.current_pose[2]):.1f}°)")
        logging.info(f"Map saved to: go2_map_test.pkl")
        logging.info(f"{'='*60}")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
