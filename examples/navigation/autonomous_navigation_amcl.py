"""
Autonomous Navigation with AMCL Localization
Automatically estimates robot position using AMCL, then navigates to goal
"""

import asyncio
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.navigation import load_map, AMCL
from go2_webrtc_driver.navigation.planning import AStarPlanner
from go2_webrtc_driver.navigation.control import PurePursuitController

logging.basicConfig(level=logging.INFO)


class AMCLNavigator:
    """Autonomous navigation with AMCL localization"""

    def __init__(self, map_file: str):
        # Load map
        self.map = load_map(map_file)
        logging.info(f"Map loaded: {self.map.width}x{self.map.height}, resolution={self.map.resolution}m")

        # Initialize AMCL for localization
        self.amcl = AMCL(num_particles=50)  # Use fewer particles for real-time performance
        # Initialize with map (particles will be spread uniformly)
        self.amcl.initialize(self.map)
        logging.info("AMCL initialized with 50 particles")

        # Initialize planner and controller (reduce inflation to avoid false obstacles)
        self.planner = AStarPlanner(inflation_radius=0.15)
        self.controller = PurePursuitController(
            look_ahead_distance=0.8,
            max_linear_velocity=0.3,
            max_angular_velocity=0.6,
            goal_tolerance=0.3
        )

        # State
        self.current_pose = (0.0, 0.0, 0.0)
        self.amcl_pose = None
        self.current_odom = None  # Current accumulated odometry for AMCL
        self.last_scan = None  # Store last lidar scan
        self.last_scan_angles = None
        self.path = None
        self.goal = None
        self.is_localized = False
        self.localization_confidence = 0.0

        # Navigation state
        self.is_navigating = False

        # WebRTC connection
        self.conn = None

        # Async queue for LiDAR processing - will be created in async context!
        self.lidar_queue = None

        # Connection health monitoring
        self.last_message_time = None
        self.message_count = 0

        # AMCL processing lock to prevent concurrent updates
        self._amcl_processing = False

        # Visualization - DISABLED for now to test connection stability
        self.fig = None
        self.ax = None
        # self._setup_visualization()  # COMMENTED OUT

    def _setup_visualization(self):
        """Setup visualization - use non-blocking backend"""
        # CRITICAL: Use Agg backend for thread-safe non-blocking operation
        import matplotlib
        matplotlib.use('TkAgg')  # Use TkAgg for better async compatibility

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_xlabel('Y (meters)')
        self.ax.set_ylabel('X (meters)')
        self.ax.set_title('AMCL-based Autonomous Navigation')
        self.ax.grid(True, alpha=0.3)

        # Display map
        map_image = self._get_map_image()
        extent = [
            self.map.origin[1],
            self.map.origin[1] + self.map.width * self.map.resolution,
            self.map.origin[0],
            self.map.origin[0] + self.map.height * self.map.resolution
        ]
        self.map_plot = self.ax.imshow(map_image, cmap='gray', origin='lower',
                                       extent=extent, vmin=0, vmax=255, alpha=1.0)

        # Initialize plots
        self.particles_plot = self.ax.scatter([], [], c='cyan', s=1, alpha=0.3, label='Particles')
        self.path_plot, = self.ax.plot([], [], 'g-', linewidth=2, label='Planned Path')
        self.robot_plot, = self.ax.plot([], [], 'bo', markersize=10, label='Robot (AMCL)')
        self.robot_arrow = None
        self.goal_plot, = self.ax.plot([], [], 'r*', markersize=20, label='Goal')

        # Status text
        self.status_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                        verticalalignment='top', bbox=dict(boxstyle='round',
                                        facecolor='wheat', alpha=0.8))

        self.ax.legend()
        plt.tight_layout()

        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        plt.show(block=False)
        # Force initial draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _get_map_image(self):
        """Get map as image"""
        image = np.ones_like(self.map.data, dtype=np.uint8) * 205  # Lighter gray for unknown
        image[self.map.data == -1] = 205  # Unknown -> light gray
        mask_free = (self.map.data >= 0) & (self.map.data < 50)
        image[mask_free] = 255  # Free -> white
        image[self.map.data >= 50] = 0  # Occupied -> black
        return image

    def _on_click(self, event):
        """Handle mouse click to set goal - DISABLED"""
        pass  # Matplotlib disabled

    async def _plan_and_execute(self, goal_x: float, goal_y: float):
        """Plan path and execute navigation"""
        self.goal = (goal_x, goal_y)
        self.goal_plot.set_data([goal_y], [goal_x])
        self.fig.canvas.draw_idle()

        # Use AMCL pose for planning
        start = (self.amcl_pose[0], self.amcl_pose[1])

        logging.info(f"Planning path from ({start[0]:.2f}, {start[1]:.2f}) to goal...")

        # Check if start position is valid
        start_row, start_col = self.map.world_to_map(start[0], start[1])
        if self.map.is_valid(start_row, start_col):
            start_occupancy = self.map.data[start_row, start_col]
            logging.info(f"Start cell occupancy: {start_occupancy} (row={start_row}, col={start_col})")

        self.path = self.planner.plan(self.map, start, self.goal)

        if self.path is None:
            logging.error("Failed to plan path!")
            logging.error(f"Try reducing inflation_radius or check if AMCL localization is accurate")
            return

        logging.info(f"Path planned with {len(self.path)} waypoints")

        # Visualize path
        path_x = [p[0] for p in self.path]
        path_y = [p[1] for p in self.path]
        self.path_plot.set_data(path_y, path_x)
        self.fig.canvas.draw_idle()

        # Set path for controller
        self.controller.set_path(self.path)

        # Start navigation
        self.is_navigating = True
        logging.info("Starting navigation...")
        await self._navigate()

    async def _navigate(self):
        """Execute navigation"""
        if self.conn is None:
            logging.error("Not connected to robot")
            self.is_navigating = False
            return

        rate = 0.2  # 5 Hz control rate
        last_pose = self.amcl_pose
        pose_stuck_count = 0

        # Wait a bit before starting to ensure connection is stable
        await asyncio.sleep(0.5)

        while self.is_navigating:
            if self.amcl_pose is None:
                logging.warning("Lost localization!")
                await self._send_velocity_command(0.0, 0.0)
                self.is_navigating = False
                break

            # Check if pose is updating (detect connection loss)
            if last_pose == self.amcl_pose:
                pose_stuck_count += 1
                if pose_stuck_count > 20:  # 10 seconds without pose update
                    logging.warning("Pose not updating! Connection may be lost.")
                    logging.warning("Stopping navigation. Please restart the program.")
                    await self._send_velocity_command(0.0, 0.0)
                    self.is_navigating = False
                    break
            else:
                pose_stuck_count = 0
                last_pose = self.amcl_pose

            # Use AMCL pose for control
            linear_vel, angular_vel, goal_reached = self.controller.compute_control(self.amcl_pose)

            # Debug: Log first few control commands
            if not hasattr(self, '_nav_loop_count'):
                self._nav_loop_count = 0

            if self._nav_loop_count < 5:
                logging.info(f"Control: lin={linear_vel:.2f}, ang={angular_vel:.2f}, waypoint={self.controller.current_waypoint_index}/{len(self.path)}, pose=({self.amcl_pose[0]:.2f},{self.amcl_pose[1]:.2f})")
            self._nav_loop_count += 1

            if goal_reached:
                logging.info("Goal reached!")
                await self._send_velocity_command(0.0, 0.0)
                self.is_navigating = False
                break

            await self._send_velocity_command(linear_vel, angular_vel)

            if self.controller.current_waypoint_index % 10 == 0:
                progress = self.controller.get_progress() * 100
                logging.info(f"Progress: {progress:.1f}%, Confidence: {self.localization_confidence:.2f}")

            await asyncio.sleep(rate)


    async def _send_velocity_command(self, linear: float, angular: float):
        """Send velocity command to robot"""
        if self.conn is None:
            logging.warning("No connection, cannot send velocity command")
            return

        command = {
            "x": float(linear),
            "y": 0.0,
            "z": float(angular)
        }

        try:
            # Check if datachannel is still open
            if not hasattr(self.conn, 'datachannel') or self.conn.datachannel is None:
                logging.warning("Datachannel not available")
                return

            self.conn.datachannel.pub_sub.publish_without_callback(
                "rt/api/sport/request",
                {"api_id": 1008, "parameter": command}
            )
        except Exception as e:
            logging.error(f"Failed to send velocity command: {e}")
            # Don't raise, just log

    async def _async_sportmode_callback(self, message):
        """Async wrapper for sportmodestate callback - non-blocking!"""
        # Add debug counter
        if not hasattr(self, '_sportmode_callback_count'):
            self._sportmode_callback_count = 0
        self._sportmode_callback_count += 1

        if self._sportmode_callback_count % 100 == 0:
            logging.info(f"[DEBUG] Sportmode callbacks: {self._sportmode_callback_count}")

        # Call sync function, but yield control to event loop
        await asyncio.sleep(0)  # Yield to event loop!
        self.sportmodestate_callback(message)

    def sportmodestate_callback(self, message):
        """Handle odometry updates - MUST BE LIGHTWEIGHT!"""
        # CRITICAL: This runs synchronously in the event loop
        # Do minimal work here to avoid blocking heartbeat
        try:
            # Update connection health (fast operation)
            import time
            self.last_message_time = time.time()
            self.message_count += 1

            # NO THROTTLING - process all messages to keep connection alive

            # Extract data (fast - just dict access)
            data = message.get("data", {})
            position = data.get("position", [0.0, 0.0, 0.0])
            imu_state = data.get("imu_state", {})
            rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])

            # Accumulate odometry (fast - just arithmetic)
            if self.current_odom is None:
                self.current_odom = [0.0, 0.0, 0.0]
                self.prev_pos = position
                self.prev_yaw = rpy[2]
            else:
                # Calculate delta
                dx = position[0] - self.prev_pos[0]
                dy = position[1] - self.prev_pos[1]
                dtheta = rpy[2] - self.prev_yaw

                # Normalize angle (simplified)
                if dtheta > np.pi:
                    dtheta -= 2 * np.pi
                elif dtheta < -np.pi:
                    dtheta += 2 * np.pi

                # Accumulate
                self.current_odom[0] += dx
                self.current_odom[1] += dy
                self.current_odom[2] += dtheta

                self.prev_pos = position
                self.prev_yaw = rpy[2]

        except Exception as e:
            # Don't even log in callback - too slow
            pass

    async def _process_lidar_queue(self):
        """Background task to process LiDAR messages from queue"""
        while True:
            try:
                # Get message from queue with timeout to allow cleanup
                try:
                    message = await asyncio.wait_for(self.lidar_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # No message available, continue loop
                    continue

                # Process the message
                await self._process_lidar_message(message)

                # Mark task as done
                self.lidar_queue.task_done()

            except asyncio.CancelledError:
                logging.info("LiDAR processing task cancelled")
                break
            except Exception as e:
                logging.error(f"Error in LiDAR processing task: {e}", exc_info=True)
                # Continue processing even if one message fails
                await asyncio.sleep(0.1)

    async def _process_lidar_message(self, message):
        """Process a single LiDAR message (runs in background task)"""
        try:
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

            # Convert voxel indices to world coordinates
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

            # Filter and convert to polar
            ranges_temp = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            valid_mask = (ranges_temp >= 0.5) & (ranges_temp <= 10.0)
            points_filtered = points[valid_mask]

            if len(points_filtered) == 0:
                return

            # Aggressive downsample for AMCL performance (use only ~30 points)
            downsample_factor = max(1, len(points_filtered) // 30)
            points_filtered = points_filtered[::downsample_factor]

            # Convert to polar
            x = points_filtered[:, 0]
            y = points_filtered[:, 1]
            ranges = np.sqrt(x**2 + y**2).tolist()
            angles = np.arctan2(y, x).tolist()

            # Initialize scan counter
            if not hasattr(self, '_scan_count'):
                self._scan_count = 0

            self._scan_count += 1

            # CRITICAL: Process AMCL only every 10th scan to reduce CPU load!
            if self._scan_count % 10 != 0:
                return

            # CRITICAL: Skip if another AMCL update is already running
            if self._amcl_processing:
                logging.debug("Skipping AMCL update - previous update still running")
                return

            # Update AMCL with sensor data
            if self.current_odom is not None:
                # Debug: Log first update
                if not hasattr(self, '_first_amcl_update'):
                    self._first_amcl_update = True
                    logging.info(f"First AMCL update: odom={self.current_odom}, scan_points={len(ranges)}")

                # Debug: Log before update
                if self._scan_count % 20 == 0:
                    logging.info(f"Calling AMCL update... scan={len(ranges)} points")

                try:
                    # Set processing flag
                    self._amcl_processing = True

                    # CRITICAL: Run AMCL update in a separate thread to avoid blocking event loop!
                    # This allows the event loop to continue processing heartbeats
                    self.amcl_pose = await asyncio.to_thread(
                        self.amcl.update,
                        tuple(self.current_odom),
                        ranges,
                        angles,
                        self.map
                    )

                    # Clear processing flag
                    self._amcl_processing = False

                    # Debug: Log after update
                    if self._scan_count % 20 == 0:
                        logging.info(f"AMCL update complete. Pose: ({self.amcl_pose[0]:.2f}, {self.amcl_pose[1]:.2f}, {np.degrees(self.amcl_pose[2]):.1f}°)")
                except Exception as e:
                    logging.error(f"AMCL update failed: {e}", exc_info=True)
                    self._amcl_processing = False  # Clear flag on error
                    return

                # Get particles for visualization and confidence
                particles = self.amcl.get_particles()

                # Debug: Log particle count
                if self._scan_count % 20 == 0:
                    logging.info(f"AMCL: {len(particles)} particles")
            else:
                logging.warning("Waiting for odometry data...")
                return

            # Check localization confidence
            self.localization_confidence = self._compute_confidence(particles)

            # Log confidence periodically
            if self._scan_count % 20 == 0:
                logging.info(f"Localization confidence: {self.localization_confidence:.3f}")

            if self.localization_confidence > 0.3 and not self.is_localized:  # Lower threshold
                self.is_localized = True
                logging.info(f"Robot localized at: ({self.amcl_pose[0]:.2f}, {self.amcl_pose[1]:.2f})")
                logging.info(f"Confidence: {self.localization_confidence:.3f}")
                logging.info(f"Map bounds: x=[{self.map.origin[0]:.1f}, {self.map.origin[0] + self.map.height*self.map.resolution:.1f}], y=[{self.map.origin[1]:.1f}, {self.map.origin[1] + self.map.width*self.map.resolution:.1f}]")
                logging.info("Ready! Click on map to set goal.")
                # Don't force update here - let regular update cycle handle it

            # Visualization disabled
            # if self._scan_count % 50 == 0:
            #     self._update_visualization()  # DISABLED

        except Exception as e:
            logging.error(f"Error processing lidar message: {e}", exc_info=True)

    async def _async_lidar_callback(self, message):
        """Async wrapper for lidar callback - non-blocking!"""
        # Add debug counter
        if not hasattr(self, '_lidar_callback_count'):
            self._lidar_callback_count = 0
        self._lidar_callback_count += 1

        if self._lidar_callback_count % 10 == 0:
            logging.info(f"[DEBUG] LiDAR callbacks: {self._lidar_callback_count}")

        # Call sync function, but yield control to event loop
        await asyncio.sleep(0)  # Yield to event loop!
        self.lidar_callback(message)

    def lidar_callback(self, message):
        """Handle lidar updates - MUST BE ULTRA LIGHTWEIGHT!"""
        # CRITICAL: This runs synchronously and blocks the event loop
        # Only do the absolute minimum here
        try:
            # If queue is full, drop oldest (keep newest data)
            if self.lidar_queue.full():
                try:
                    self.lidar_queue.get_nowait()
                except:
                    pass

            # Put new message in queue (non-blocking)
            self.lidar_queue.put_nowait(message)

        except:
            # Don't even log - too slow for callback
            pass

    def _compute_confidence(self, particles):
        """Compute localization confidence from particle distribution"""
        if len(particles) == 0:
            return 0.0

        weights = np.array([p.weight for p in particles])
        # Effective sample size
        if np.sum(weights) > 0:
            normalized_weights = weights / np.sum(weights)
            n_eff = 1.0 / np.sum(normalized_weights ** 2)
            confidence = n_eff / len(particles)
        else:
            confidence = 0.0

        return confidence

    def _update_visualization(self):
        """Update visualization"""
        try:
            if self.amcl_pose is not None:
                # Update robot position
                x, y, theta = self.amcl_pose
                logging.debug(f"Updating viz: robot at ({x:.2f}, {y:.2f})")
                self.robot_plot.set_data([y], [x])

                # Update robot direction
                if self.robot_arrow is not None:
                    self.robot_arrow.remove()

                arrow_length = 0.5
                dx = arrow_length * np.cos(theta)
                dy = arrow_length * np.sin(theta)
                self.robot_arrow = self.ax.arrow(y, x, dy, dx,
                                                head_width=0.2, head_length=0.3,
                                                fc='red', ec='red', alpha=0.8)

            # Update particles
            particles = self.amcl.get_particles()
            if len(particles) > 0:
                particle_y = [p.y for p in particles[::10]]  # Subsample for performance
                particle_x = [p.x for p in particles[::10]]
                self.particles_plot.set_offsets(np.c_[particle_y, particle_x])

            # Update status text
            status = f"Localized: {self.is_localized}\n"
            status += f"Confidence: {self.localization_confidence:.2f}\n"
            if self.amcl_pose:
                status += f"Pose: ({self.amcl_pose[0]:.2f}, {self.amcl_pose[1]:.2f}, {np.degrees(self.amcl_pose[2]):.0f}°)\n"
            status += f"Navigating: {self.is_navigating}"
            self.status_text.set_text(status)

            # Update canvas (events processed in main loop)
            self.fig.canvas.draw_idle()

        except Exception as e:
            logging.error(f"Error updating visualization: {e}")


async def main():
    try:
        map_file = "go2_map.pkl"

        logging.info("=" * 60)
        logging.info("AMCL-based Autonomous Navigation")
        logging.info("=" * 60)

        # Connect to robot FIRST (before matplotlib initialization)
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")
        logging.info("Connecting to Go2...")
        await conn.connect()
        logging.info("Connected!")

        # Initialize navigator AFTER connection (matplotlib can take time)
        logging.info("Initializing navigator and GUI...")
        navigator = AMCLNavigator(map_file)
        logging.info("Navigator initialized!")

        # Create asyncio queue NOW (in correct event loop context)
        navigator.lidar_queue = asyncio.Queue(maxsize=2)

        navigator.conn = conn

        await conn.datachannel.disableTrafficSaving(True)
        conn.datachannel.set_decoder(decoder_type='libvoxel')

        # Don't send stand up command - assume robot is already ready
        logging.info("Robot should be in sport mode already...")
        await asyncio.sleep(0.5)  # Brief delay

        # Subscribe to odometry and lidar
        # Use SIMPLE SYNC callbacks - they work!
        conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", navigator.sportmodestate_callback)
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
        conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", navigator.lidar_callback)

        # Start background tasks
        lidar_task = asyncio.create_task(navigator._process_lidar_queue())
        logging.info("Background task started (LiDAR processing)")

        logging.info("\nWaiting for AMCL to localize robot...")
        logging.info("Move the robot slowly to help localization converge.\n")

        # Keep running - NO GUI
        try:
            logging.info("Running... Press Ctrl+C to stop")
            while True:
                # No matplotlib - just sleep
                await asyncio.sleep(1)
        finally:
            # Cancel background task on exit
            lidar_task.cancel()
            try:
                await lidar_task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        logging.info("\nStopping...")
        if navigator.conn and navigator.is_navigating:
            await navigator._send_velocity_command(0.0, 0.0)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    import threading
    import time

    # Create a new event loop for asyncio
    loop = asyncio.new_event_loop()

    def run_asyncio_loop(loop):
        """Run asyncio event loop in separate thread"""
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main())
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()

    # Start asyncio in separate thread
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    asyncio_thread.start()

    print("\nPress Ctrl+C to stop...")

    try:
        # Main thread just waits
        while True:
            time.sleep(0.1)  # Regular sleep, not asyncio.sleep!
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=2)
        sys.exit(0)
