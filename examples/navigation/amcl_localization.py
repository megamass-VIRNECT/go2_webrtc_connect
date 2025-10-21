"""
AMCL Localization Example for Go2 Robot
Localizes robot on a pre-built map using particle filter
"""

import asyncio
import logging
import sys
import numpy as np
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.load_webrtc_config import load_webrtc_config
from go2_webrtc_driver.navigation import AMCL, load_map
from go2_webrtc_driver.navigation.amcl import MotionModelParams, SensorModelParams

# Enable logging
logging.basicConfig(level=logging.INFO)


class AMCLLocalizer:
    """AMCL localizer for Go2 robot"""

    def __init__(self, map_file: str, initial_pose=None):
        # Load map
        logging.info(f"Loading map from {map_file}...")
        self.occupancy_map = load_map(map_file)
        logging.info(f"Map loaded: {self.occupancy_map.width}x{self.occupancy_map.height} cells, "
                    f"resolution={self.occupancy_map.resolution}m")

        # Initialize AMCL
        motion_params = MotionModelParams(
            alpha1=0.2,  # Rotation noise from rotation
            alpha2=0.2,  # Rotation noise from translation
            alpha3=0.2,  # Translation noise from translation
            alpha4=0.2   # Translation noise from rotation
        )

        sensor_params = SensorModelParams(
            z_hit=0.95,
            z_rand=0.05,
            sigma_hit=0.2,
            max_range=10.0,
            min_range=0.1,
            max_beams=50  # Use subset of beams for efficiency
        )

        self.amcl = AMCL(
            num_particles=500,
            motion_params=motion_params,
            sensor_params=sensor_params,
            resample_threshold=0.5
        )

        # Initialize AMCL
        self.amcl.initialize(self.occupancy_map, initial_pose)
        logging.info(f"AMCL initialized with {len(self.amcl.get_particles())} particles")

        # Current odometry
        self.current_odom = None

        # Statistics
        self.update_count = 0

    def sportmodestate_callback(self, message):
        """Handle sport mode state updates (odometry)"""
        try:
            data = message.get("data", {})

            # Extract position and orientation
            position = data.get("position", [0.0, 0.0, 0.0])
            imu_state = data.get("imu_state", {})
            rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])

            # Current odometry (x, y, yaw)
            self.current_odom = (position[0], position[1], rpy[2])

        except Exception as e:
            logging.error(f"Error in sportmodestate callback: {e}")

    def lidar_callback(self, message):
        """Handle LIDAR scan updates"""
        try:
            if self.current_odom is None:
                logging.warning("No odometry data available yet")
                return

            data = message.get("data", {})

            # Extract point cloud
            points = data.get("point_cloud", [])
            if not points or len(points) == 0:
                return

            # Convert point cloud to polar coordinates (range, angle)
            ranges = []
            angles = []

            for point in points:
                x, y = point[0], point[1]
                range_val = np.sqrt(x**2 + y**2)
                angle = np.arctan2(y, x)

                ranges.append(range_val)
                angles.append(angle)

            # Update AMCL
            estimated_pose = self.amcl.update(
                self.current_odom,
                ranges,
                angles,
                self.occupancy_map
            )

            self.update_count += 1

            if self.update_count % 5 == 0:
                # Get covariance for uncertainty estimate
                cov = self.amcl.get_covariance()
                pos_uncertainty = np.sqrt(cov[0, 0] + cov[1, 1])
                angle_uncertainty = np.sqrt(cov[2, 2])

                logging.info(f"Update {self.update_count}: "
                           f"Pose: x={estimated_pose[0]:.2f}, y={estimated_pose[1]:.2f}, "
                           f"theta={estimated_pose[2]:.2f} | "
                           f"Uncertainty: pos={pos_uncertainty:.3f}m, "
                           f"angle={np.degrees(angle_uncertainty):.1f}deg | "
                           f"ESS={self.amcl.particle_filter.effective_sample_size():.1f}")

        except Exception as e:
            logging.error(f"Error in lidar callback: {e}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description='AMCL Localization for Go2 Robot')
    parser.add_argument('--map', type=str, default='go2_map.pkl',
                       help='Path to map file (default: go2_map.pkl)')
    parser.add_argument('--init-x', type=float, default=None,
                       help='Initial x position (meters)')
    parser.add_argument('--init-y', type=float, default=None,
                       help='Initial y position (meters)')
    parser.add_argument('--init-theta', type=float, default=None,
                       help='Initial theta angle (radians)')
    parser.add_argument('--ip', type=str, default='192.168.1.7',
                       help='Robot IP address for LocalSTA connection')

    args = parser.parse_args()

    try:
        # Determine initial pose
        initial_pose = None
        if args.init_x is not None and args.init_y is not None and args.init_theta is not None:
            initial_pose = (args.init_x, args.init_y, args.init_theta)
            logging.info(f"Using initial pose: x={initial_pose[0]}, y={initial_pose[1]}, "
                        f"theta={initial_pose[2]}")
        else:
            logging.info("No initial pose specified, using uniform distribution")

        # Initialize localizer
        localizer = AMCLLocalizer(args.map, initial_pose)

        # Connection setup
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=args.ip)

        # Connect to the WebRTC service
        logging.info("Connecting to Go2...")
        await conn.connect()
        logging.info("Connected successfully!")

        # Disable traffic saving mode
        await conn.datachannel.disableTrafficSaving(True)

        # Set decoder type
        conn.datachannel.set_decoder(decoder_type='libvoxel')

        # Subscribe to sport mode state (odometry)
        conn.datachannel.pub_sub.subscribe("rt/sportmodestate", localizer.sportmodestate_callback)
        logging.info("Subscribed to sportmodestate")

        # Turn on LIDAR
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
        logging.info("LIDAR turned on")

        # Subscribe to LIDAR data
        conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", localizer.lidar_callback)
        logging.info("Subscribed to LIDAR data")

        logging.info("Localization running... (Press Ctrl+C to stop)")

        # Run indefinitely
        try:
            await asyncio.sleep(float('inf'))
        except KeyboardInterrupt:
            logging.info("Localization stopped by user")

        # Print final statistics
        final_pose = localizer.amcl.get_pose()
        final_cov = localizer.amcl.get_covariance()

        logging.info(f"\nLocalization stopped!")
        logging.info(f"Total updates: {localizer.update_count}")
        logging.info(f"Final pose: x={final_pose[0]:.2f}, y={final_pose[1]:.2f}, "
                    f"theta={final_pose[2]:.2f}")
        logging.info(f"Position uncertainty: {np.sqrt(final_cov[0,0] + final_cov[1,1]):.3f}m")
        logging.info(f"Angle uncertainty: {np.degrees(np.sqrt(final_cov[2,2])):.1f} degrees")

    except FileNotFoundError:
        logging.error(f"Map file not found: {args.map}")
        logging.error("Please run slam_mapping.py first to create a map")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
