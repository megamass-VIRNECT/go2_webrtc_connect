"""
SLAM Mapping Example for Go2 Robot
Creates an occupancy grid map using Lidar data
"""

import asyncio
import logging
import sys
import numpy as np
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.load_webrtc_config import load_webrtc_config
from go2_webrtc_driver.navigation import SLAM, SLAMParams

# Enable logging
logging.basicConfig(level=logging.INFO)


class SLAMMapper:
    """SLAM mapper for Go2 robot"""

    def __init__(self):
        # Initialize SLAM
        slam_params = SLAMParams(
            map_width=2000,
            map_height=2000,
            map_resolution=0.05,  # 5cm resolution
            map_origin=(-50.0, -50.0),  # 50m x 50m map
            max_range=10.0,
            min_range=0.1
        )
        self.slam = SLAM(slam_params)

        # Current pose estimate (x, y, theta)
        self.current_pose = (0.0, 0.0, 0.0)

        # Previous odometry
        self.prev_odom = None

        # Statistics
        self.scan_count = 0

    def sportmodestate_callback(self, message):
        """Handle sport mode state updates (odometry)"""
        try:
            data = message.get("data", {})

            # Extract position and orientation
            position = data.get("position", [0.0, 0.0, 0.0])
            imu_state = data.get("imu_state", {})
            rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])

            # Current odometry (x, y, yaw)
            current_odom = (position[0], position[1], rpy[2])

            # Update pose estimate
            if self.prev_odom is not None:
                # Simple odometry integration
                dx = current_odom[0] - self.prev_odom[0]
                dy = current_odom[1] - self.prev_odom[1]
                dtheta = current_odom[2] - self.prev_odom[2]

                # Update current pose
                x, y, theta = self.current_pose
                x += dx * np.cos(theta) - dy * np.sin(theta)
                y += dx * np.sin(theta) + dy * np.cos(theta)
                theta += dtheta

                # Normalize theta
                theta = np.arctan2(np.sin(theta), np.cos(theta))

                self.current_pose = (x, y, theta)

            self.prev_odom = current_odom

        except Exception as e:
            logging.error(f"Error in sportmodestate callback: {e}")

    def lidar_callback(self, message):
        """Handle LIDAR scan updates"""
        try:
            data = message.get("data", {})

            # Extract point cloud
            points = data.get("point_cloud", [])
            if not points or len(points) == 0:
                return

            # Convert point cloud to polar coordinates (range, angle)
            ranges = []
            angles = []

            for point in points:
                x, y = point[0], point[1]  # Assuming points are [x, y, z]
                range_val = np.sqrt(x**2 + y**2)
                angle = np.arctan2(y, x)

                ranges.append(range_val)
                angles.append(angle)

            # Update SLAM map
            self.slam.update(self.current_pose, ranges, angles)

            self.scan_count += 1

            if self.scan_count % 10 == 0:
                logging.info(f"Processed {self.scan_count} scans. Current pose: "
                           f"x={self.current_pose[0]:.2f}, y={self.current_pose[1]:.2f}, "
                           f"theta={self.current_pose[2]:.2f}")

        except Exception as e:
            logging.error(f"Error in lidar callback: {e}")


async def main():
    try:
        # Connection setup
        # Uncomment the appropriate connection method:
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")
        # config = load_webrtc_config()
        # conn = Go2WebRTCConnection(WebRTCConnectionMethod.Remote,
        #                           serialNumber=config["sn"],
        #                           username=config["user"]["email"],
        #                           password=config["user"]["password"])

        # Connect to the WebRTC service
        logging.info("Connecting to Go2...")
        await conn.connect()
        logging.info("Connected successfully!")

        # Disable traffic saving mode
        await conn.datachannel.disableTrafficSaving(True)

        # Set decoder type
        conn.datachannel.set_decoder(decoder_type='libvoxel')

        # Initialize SLAM mapper
        mapper = SLAMMapper()

        # Subscribe to sport mode state (odometry)
        conn.datachannel.pub_sub.subscribe("rt/sportmodestate", mapper.sportmodestate_callback)
        logging.info("Subscribed to sportmodestate")

        # Turn on LIDAR
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
        logging.info("LIDAR turned on")

        # Subscribe to LIDAR data
        conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", mapper.lidar_callback)
        logging.info("Subscribed to LIDAR data")

        # Run for specified duration
        duration = 300  # 5 minutes
        logging.info(f"Mapping for {duration} seconds... (Press Ctrl+C to stop early)")

        try:
            await asyncio.sleep(duration)
        except KeyboardInterrupt:
            logging.info("Mapping interrupted by user")

        # Save the map
        map_filename = "go2_map.pkl"
        mapper.slam.save_map(map_filename)
        logging.info(f"Map saved to {map_filename}")

        # Print statistics
        logging.info(f"\nMapping complete!")
        logging.info(f"Total scans processed: {mapper.scan_count}")
        logging.info(f"Final pose: x={mapper.current_pose[0]:.2f}, "
                    f"y={mapper.current_pose[1]:.2f}, "
                    f"theta={mapper.current_pose[2]:.2f}")
        logging.info(f"Trajectory points: {len(mapper.slam.get_trajectory())}")

        # Optionally visualize map (requires matplotlib)
        try:
            import matplotlib.pyplot as plt

            map_image = mapper.slam.get_map_image()
            plt.figure(figsize=(10, 10))
            plt.imshow(map_image, cmap='gray', origin='lower')
            plt.title('SLAM Map')
            plt.xlabel('X (cells)')
            plt.ylabel('Y (cells)')

            # Plot trajectory
            trajectory = mapper.slam.get_trajectory()
            if trajectory:
                traj_x = [(p[0] - mapper.slam.map.origin[0]) / mapper.slam.map.resolution
                         for p in trajectory]
                traj_y = [(p[1] - mapper.slam.map.origin[1]) / mapper.slam.map.resolution
                         for p in trajectory]
                plt.plot(traj_y, traj_x, 'r-', linewidth=2, label='Trajectory')
                plt.legend()

            plt.savefig('slam_map.png', dpi=150, bbox_inches='tight')
            logging.info("Map visualization saved to slam_map.png")
            plt.show()

        except ImportError:
            logging.warning("matplotlib not available, skipping visualization")

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
