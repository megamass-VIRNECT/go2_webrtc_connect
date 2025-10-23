"""
Autonomous Navigation with AMCL Localization
Automatically estimates robot position using AMCL, then navigates to goal
"""

import asyncio
import logging
import sys
import threading
import numpy as np
import matplotlib.pyplot as plt
import json
import queue
from multiprocessing import Process, Queue, Event
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.navigation import load_map, AMCL
from go2_webrtc_driver.navigation.planning import AStarPlanner
from go2_webrtc_driver.navigation.control import PurePursuitController
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

logging.basicConfig(level=logging.INFO)


def amcl_worker_process(map_file: str, input_queue: Queue, output_queue: Queue, stop_event: Event):
    """
    Dedicated PROCESS for AMCL - completely independent from main event loop
    This is CRITICAL - Thread won't work due to Python GIL blocking heartbeat
    """
    import time
    update_count = 0

    try:
        # Load map
        map_obj = load_map(map_file)
        amcl = AMCL(num_particles=20)  # Minimal particles for real-time performance
        amcl.initialize(map_obj)

        # Send init confirmation
        output_queue.put({'type': 'init', 'msg': f"AMCL initialized: {map_obj.width}x{map_obj.height}"})

        while not stop_event.is_set():
            try:
                # Get data with timeout
                data = input_queue.get(timeout=0.1)

                update_count += 1
                odom = data['odom']
                ranges = data['ranges']
                angles = data['angles']

                start_time = time.time()

                # Perform AMCL update
                pose = amcl.update(tuple(odom), ranges, angles, map_obj)

                update_time = time.time() - start_time

                particles = amcl.get_particles()

                elapsed = time.time() - start_time

                # Log slow updates
                if elapsed > 1.0:
                    output_queue.put({
                        'type': 'warning',
                        'msg': f"Slow AMCL update #{update_count}: {elapsed:.2f}s (scan_points={len(ranges)}, particles={len(particles)})"
                    })

                # Compute confidence
                weights = np.array([p.weight for p in particles])
                if np.sum(weights) > 0:
                    normalized_weights = weights / np.sum(weights)
                    n_eff = 1.0 / np.sum(normalized_weights ** 2)
                    confidence = n_eff / len(particles)
                else:
                    confidence = 0.0

                # Send result back
                result = {
                    'type': 'update',
                    'pose': pose,
                    'confidence': confidence,
                    'particles': [(p.x, p.y, p.theta, p.weight) for p in particles[::3]],
                    'update_count': update_count,
                    'elapsed': elapsed
                }

                # Non-blocking put - if queue full, replace old data
                try:
                    output_queue.put_nowait(result)
                except:
                    try:
                        output_queue.get_nowait()  # Remove old
                        output_queue.put_nowait(result)  # Add new
                    except:
                        pass

            except queue.Empty:
                continue
            except Exception as e:
                output_queue.put({'type': 'error', 'msg': str(e)})

    except Exception as e:
        output_queue.put({'type': 'fatal', 'msg': str(e)})


class AMCLNavigator:
    """Autonomous navigation with AMCL localization"""

    def __init__(self, map_file: str, loop=None, enable_visualization: bool = True):
        # Load map
        self.map_file = map_file
        self.map = load_map(map_file)
        logging.info(f"Map loaded: {self.map.width}x{self.map.height}, resolution={self.map.resolution}m")

        # AMCL processing in separate PROCESS (CRITICAL: no GIL blocking!)
        self.amcl_input_queue = Queue(maxsize=2)
        self.amcl_output_queue = Queue(maxsize=10)  # Larger output queue
        self.amcl_stop_event = Event()
        self.amcl_process = None
        self.particles_cache = []

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
        self.amcl_pose = None  # AMCL pose (continuously updated)
        self.current_odom = None  # Current accumulated odometry for AMCL input
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

        # Event loop reference (for scheduling coroutines from callbacks)
        self.loop = loop
        if self.loop is None:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = None

        # Visualization state
        self.fig = None
        self.ax = None
        self.visualization_requested = enable_visualization
        self.visualization_initialized = False
        self.particles_plot = None
        self.path_plot = None
        self.robot_plot = None
        self.robot_arrow = None
        self.goal_plot = None
        self.status_text = None

        # Connection management
        self._connection_callbacks_registered = False
        self._reconnecting = False

        if self.visualization_requested and threading.current_thread() is threading.main_thread():
            self._setup_visualization()

        # Start AMCL worker thread
        self.start_amcl_worker()

    def start_amcl_worker(self):
        """Start AMCL processing in separate process"""
        if self.amcl_process is None or not self.amcl_process.is_alive():
            self.amcl_stop_event.clear()
            self.amcl_process = Process(
                target=amcl_worker_process,
                args=(self.map_file, self.amcl_input_queue, self.amcl_output_queue, self.amcl_stop_event),
                daemon=True
            )
            self.amcl_process.start()
            logging.info("[Main] AMCL worker process started (PID: %d)", self.amcl_process.pid)

    def stop_amcl_worker(self):
        """Stop AMCL worker process"""
        if self.amcl_process and self.amcl_process.is_alive():
            self.amcl_stop_event.set()
            self.amcl_process.join(timeout=2)
            if self.amcl_process.is_alive():
                self.amcl_process.terminate()
            logging.info("[Main] AMCL worker process stopped")

    def _setup_visualization(self):
        """Setup visualization - use non-blocking backend"""
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
        self.visualization_initialized = True

    def initialize_visualization(self):
        """Initialize visualization from the main thread if requested"""
        if not self.visualization_requested or self.visualization_initialized:
            return

        self._setup_visualization()

    async def on_connection_ready(self):
        """Configure data channel after (re)connection"""
        if self.conn is None or self.conn.datachannel is None:
            logging.warning("Connection not ready for configuration")
            return

        try:
            await self.conn.datachannel.disableTrafficSaving(True)
        except Exception as exc:
            logging.error(f"Failed to disable traffic saving: {exc}")

        try:
            self.conn.datachannel.set_decoder(decoder_type='libvoxel')
        except Exception as exc:
            logging.error(f"Failed to set decoder: {exc}")

        await self.init()

        # If we had an active goal before disconnect, attempt to resume navigation
        if self.goal is not None and self.path:
            asyncio.create_task(self._resume_navigation_after_reconnect())

    async def init(self):
        if self.conn is None or self.conn.datachannel is None:
            return

        self.conn.datachannel.pub_sub.subscribe(
            RTC_TOPIC["LF_SPORT_MOD_STATE"],
            self.sportmodestate_callback
        )

        self.conn.datachannel.pub_sub.subscribe(
            RTC_TOPIC["ULIDAR_ARRAY"],
            lambda message: asyncio.create_task(self.lidar_callback_task(message))
        )

        self.conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["ULIDAR_SWITCH"], "on")

        # Start background task to poll AMCL results
        asyncio.create_task(self._poll_amcl_results())

        # Check and switch motion mode if needed
        # logging.info("Checking current motion mode...")
        # response = await self.conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["MOTION_SWITCHER"],
        #     {"api_id": 1001}
        # )
        #
        # current_motion_switcher_mode = "unknown"
        # if response['data']['header']['status']['code'] == 0:
        #     data = json.loads(response['data']['data'])
        #     current_motion_switcher_mode = data['name']
        #     logging.info(f"Current motion mode: {current_motion_switcher_mode}")

        # Try to switch to normal mode if not already
        # if current_motion_switcher_mode != "normal":
        #     logging.info(f"Attempting to switch from '{current_motion_switcher_mode}' to 'normal' mode...")
        #     switch_response = await self.conn.datachannel.pub_sub.publish_request_new(
        #         RTC_TOPIC["MOTION_SWITCHER"],
        #         {
        #             "api_id": 1002,
        #             "parameter": {"name": "normal"}
        #         }
        #     )
        #     switch_code = switch_response.get('data', {}).get('header', {}).get('status', {}).get('code', 'unknown') if switch_response else 'no response'
        #
        #     if switch_code == 0:
        #         logging.info("Mode switch accepted, waiting for transition...")
        #         await asyncio.sleep(5)
        #
        #         # Verify mode changed
        #         verify_response = await self.conn.datachannel.pub_sub.publish_request_new(
        #             RTC_TOPIC["MOTION_SWITCHER"],
        #             {"api_id": 1001}
        #         )
        #         if verify_response['data']['header']['status']['code'] == 0:
        #             verify_data = json.loads(verify_response['data']['data'])
        #             new_mode = verify_data['name']
        #             if new_mode == "normal":
        #                 logging.info("✓ Successfully switched to 'normal' mode")
        #             else:
        #                 logging.warning(f"✗ Mode switch failed: still in '{new_mode}' mode")
        #                 logging.warning(f"Continuing with '{new_mode}' mode...")
        #     else:
        #         logging.warning(f"✗ Mode switch rejected with code: {switch_code}")
        #         logging.warning(f"Continuing with '{current_motion_switcher_mode}' mode...")
        # else:
        #     logging.info("Already in 'normal' mode")

        # Stand up robot
        # logging.info("Sending StandUp command...")
        # standup_response = await self.conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"],
        #     {
        #         "api_id": SPORT_CMD["StandUp"],
        #         "parameter": {}
        #     }
        # )
        # if standup_response and standup_response.get('data', {}).get('header', {}).get('status', {}).get('code') == 0:
        #     logging.info("✓ StandUp command successful")
        # else:
        #     code = standup_response.get('data', {}).get('header', {}).get('status', {}).get('code', 'unknown') if standup_response else 'no response'
        #     logging.warning(f"✗ StandUp command failed with code: {code}")
        # await asyncio.sleep(2)

        # Enable continuous gait mode
        # logging.info("Enabling continuous gait...")
        # continuous_response = await self.conn.datachannel.pub_sub.publish_request_new(
        #     RTC_TOPIC["SPORT_MOD"],
        #     {
        #         "api_id": SPORT_CMD["ContinuousGait"],
        #         "parameter": {"flag": True}
        #     }
        # )
        # if continuous_response and continuous_response.get('data', {}).get('header', {}).get('status', {}).get('code') == 0:
        #     logging.info("✓ ContinuousGait command successful")
        # else:
        #     code = continuous_response.get('data', {}).get('header', {}).get('status', {}).get('code', 'unknown') if continuous_response else 'no response'
        #     logging.warning(f"✗ ContinuousGait command failed with code: {code}")
        # await asyncio.sleep(1)

        logging.info("Robot initialization complete")

    def register_connection_callbacks(self):
        if self.conn is None or self._connection_callbacks_registered:
            return

        self.conn.add_connection_state_callback(self._on_connection_state_change)
        self.conn.add_ice_connection_state_callback(self._on_ice_state_change)
        self._connection_callbacks_registered = True

    def _on_connection_state_change(self, state: str):
        logging.info(f"Connection state changed: {state}")
        if state in {"closed", "failed"}:
            if self.loop:
                asyncio.run_coroutine_threadsafe(self._attempt_reconnect(), self.loop)

    def _on_ice_state_change(self, state: str):
        logging.info(f"ICE state changed: {state}")
        if state in {"closed", "failed"} and self.loop:
            asyncio.run_coroutine_threadsafe(self._attempt_reconnect(), self.loop)

    async def _attempt_reconnect(self):
        if self._reconnecting:
            return

        self._reconnecting = True
        self.is_navigating = False
        logging.warning("WebRTC connection lost. Attempting to reconnect...")

        try:
            for attempt in range(3):
                try:
                    await self.conn.reconnect()
                    break
                except Exception as exc:
                    logging.error(f"Reconnect attempt {attempt + 1} failed: {exc}")
                    await asyncio.sleep(2)
            else:
                logging.error("Unable to reconnect after multiple attempts")
                return

            logging.info("Reconnected to Go2. Restoring subscriptions...")
            await self.on_connection_ready()
            logging.info("Connection restored")
        finally:
            self._reconnecting = False

    async def _resume_navigation_after_reconnect(self):
        # Wait for localization to stabilize again
        for _ in range(20):
            if self.localization_confidence > 0.3 and self.amcl_pose is not None:
                break
            await asyncio.sleep(0.3)

        if self.goal is None:
            return

        logging.info("Resuming navigation after reconnect")
        await self._plan_and_execute(self.goal[0], self.goal[1])


    def _get_map_image(self):
        """Get map as image"""
        image = np.ones_like(self.map.data, dtype=np.uint8) * 205  # Lighter gray for unknown
        image[self.map.data == -1] = 205  # Unknown -> light gray
        mask_free = (self.map.data >= 0) & (self.map.data < 50)
        image[mask_free] = 255  # Free -> white
        image[self.map.data >= 50] = 0  # Occupied -> black
        return image

    def _on_click(self, event):
        """Handle mouse click to set goal"""
        if self.fig is None or self.ax is None:
            return

        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if not self.is_localized or self.amcl_pose is None:
            logging.info("Robot not localized yet; click ignored.")
            return

        if self.loop is None:
            logging.warning("Event loop unavailable; cannot schedule path planning.")
            return

        goal_y = float(event.xdata)
        goal_x = float(event.ydata)

        logging.info(f"Goal selected: ({goal_x:.2f}, {goal_y:.2f})")

        # Stop current navigation before replanning
        if self.is_navigating:
            self.is_navigating = False
            asyncio.run_coroutine_threadsafe(self._send_velocity_command(0.0, 0.0), self.loop)

        asyncio.run_coroutine_threadsafe(self._plan_and_execute(goal_x, goal_y), self.loop)

    async def _plan_and_execute(self, goal_x: float, goal_y: float):
        """Plan path and execute navigation"""
        # Clear AMCL input queue to stop pending updates
        try:
            while not self.amcl_input_queue.empty():
                self.amcl_input_queue.get_nowait()
        except:
            pass

        self.goal = (goal_x, goal_y)
        if self.fig is not None and self.goal_plot is not None:
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
        if self.fig is not None and self.path_plot is not None:
            self.path_plot.set_data(path_y, path_x)
            self.fig.canvas.draw_idle()

        # Set path for controller
        self.controller.set_path(self.path)

        # Start navigation - use continuous AMCL pose updates
        self.is_navigating = True
        logging.info("Starting navigation...")
        logging.info(f"Starting from AMCL pose: {self.amcl_pose}")
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
            # Use AMCL pose continuously updated during navigation
            current_nav_pose = self.amcl_pose

            if current_nav_pose is None:
                logging.warning("Lost localization!")
                await self._send_velocity_command(0.0, 0.0)
                self.is_navigating = False
                break

            # Check if pose is updating (detect connection loss)
            # AMCL updates are slow (3-6s), so allow longer timeout
            if last_pose == current_nav_pose:
                pose_stuck_count += 1
                if pose_stuck_count > 50:  # 25 seconds without pose update (AMCL is slow)
                    logging.warning("Pose not updating! Connection may be lost.")
                    logging.warning("Stopping navigation. Please restart the program.")
                    await self._send_velocity_command(0.0, 0.0)
                    self.is_navigating = False
                    break
            else:
                pose_stuck_count = 0
                last_pose = current_nav_pose

            # Use current navigation pose for control
            linear_vel, angular_vel, goal_reached = self.controller.compute_control(current_nav_pose)

            # Debug: Log first few control commands
            if not hasattr(self, '_nav_loop_count'):
                self._nav_loop_count = 0

            if self._nav_loop_count < 5:
                logging.info(f"Control: lin={linear_vel:.2f}, ang={angular_vel:.2f}, waypoint={self.controller.current_waypoint_index}/{len(self.path)}, pose=({current_nav_pose[0]:.2f},{current_nav_pose[1]:.2f})")
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

            if not getattr(self.conn.datachannel, 'data_channel_opened', False):
                logging.warning("Datachannel not open; skipping velocity command")
                return

            # Debug: Log first few commands
            if not hasattr(self, '_cmd_count'):
                self._cmd_count = 0
                self._failed_cmd_count = 0
            self._cmd_count += 1

            if self._cmd_count <= 3 or self._cmd_count % 50 == 0:
                logging.info(f"[CMD #{self._cmd_count}] Sending velocity: lin={linear:.3f}, ang={angular:.3f}")

            # Use publish_request_new for first few commands to detect issues early
            # After that, use publish_request_no_wait for performance (5Hz control loop)
            if self._cmd_count <= 5:
                response = await self.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {
                        "api_id": SPORT_CMD["Move"],
                        "parameter": command
                    }
                )
                if response and response.get('data', {}).get('header', {}).get('status', {}).get('code') == 0:
                    logging.info(f"[CMD #{self._cmd_count}] ✓ Move command successful")
                else:
                    code = response.get('data', {}).get('header', {}).get('status', {}).get('code', 'unknown') if response else 'no response'
                    logging.warning(f"[CMD #{self._cmd_count}] ✗ Move command failed with code: {code}")
                    self._failed_cmd_count += 1

                    # If first commands fail, stop navigation
                    if self._failed_cmd_count >= 3:
                        logging.error("Multiple move commands failed! Stopping navigation.")
                        self.is_navigating = False
            else:
                # After initial validation, use fire-and-forget for high-frequency control
                # This sends "req" type (not "msg") so robot will process it
                self.conn.datachannel.pub_sub.publish_request_no_wait(
                    RTC_TOPIC["SPORT_MOD"],
                    {
                        "api_id": SPORT_CMD["Move"],
                        "parameter": command
                    }
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

                # Odometry is only used as input to AMCL, not for navigation
                # Navigation uses AMCL pose directly (updated continuously)

                self.prev_pos = position
                self.prev_yaw = rpy[2]

        except Exception as e:
            # Don't even log in callback - too slow
            pass

    async def lidar_callback_task(self, message):
        """
        Lightweight LiDAR callback - just preprocess and send to AMCL worker
        NO BLOCKING OPERATIONS - keeps event loop free for heartbeat
        """
        try:
            data = message.get("data", {})
            data_inner = data.get("data", {})
            positions = data_inner.get("positions", [])

            if positions is None or (hasattr(positions, '__len__') and len(positions) == 0):
                return

            # Initialize scan counter
            if not hasattr(self, '_scan_count'):
                self._scan_count = 0
            self._scan_count += 1

            # Process less frequently - only every 10th scan
            # AMCL is slow (3-6s) but runs in separate process (non-blocking)
            if self._scan_count % 10 != 0:
                return

            # Don't send if we don't have odometry yet
            if self.current_odom is None:
                return

            # AMCL runs in separate process, so it won't block main loop
            # Continue updating position even during navigation for accuracy

            # Quick preprocessing (fast operations only)
            if not isinstance(positions, np.ndarray):
                positions = np.array(positions, dtype=np.float32)

            origin = data.get("origin", [0.0, 0.0, 0.0])
            width = data.get("width", [128, 128, 38])
            resolution = data.get("resolution", 0.05)

            # Convert voxel to world coords
            voxel_indices = np.array([positions[i:i+3] for i in range(0, len(positions), 3)], dtype=np.float32)
            points = np.zeros_like(voxel_indices)
            points[:, 0] = origin[0] + (voxel_indices[:, 0] * resolution)
            points[:, 1] = origin[1] + (voxel_indices[:, 1] * resolution)
            points[:, 2] = origin[2] + (voxel_indices[:, 2] * resolution)

            # Robot-center
            map_center_x = origin[0] + (width[0] * resolution) / 2.0
            map_center_y = origin[1] + (width[1] * resolution) / 2.0
            points[:, 0] -= map_center_x
            points[:, 1] -= map_center_y

            # Filter
            ranges_temp = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            valid_mask = (ranges_temp >= 0.5) & (ranges_temp <= 10.0)
            points_filtered = points[valid_mask]

            if len(points_filtered) == 0:
                return

            # Aggressive downsample to ~15 points for fast processing
            downsample_factor = max(1, len(points_filtered) // 15)
            points_filtered = points_filtered[::downsample_factor]

            # Convert to polar
            x = points_filtered[:, 0]
            y = points_filtered[:, 1]
            ranges = np.sqrt(x**2 + y**2).tolist()
            angles = np.arctan2(y, x).tolist()

            # Send to AMCL worker process (non-blocking)
            amcl_data = {
                'odom': list(self.current_odom),
                'ranges': ranges,
                'angles': angles
            }

            # Non-blocking put - drop if queue full (we want latest data only)
            try:
                self.amcl_input_queue.put_nowait(amcl_data)
            except:
                # Queue full - clear and put new data
                try:
                    self.amcl_input_queue.get_nowait()
                    self.amcl_input_queue.put_nowait(amcl_data)
                except:
                    pass

        except Exception as e:
            logging.error(f"Error in lidar callback: {e}", exc_info=True)

    async def _poll_amcl_results(self):
        """
        Background task to poll AMCL results from worker process
        Runs independently - never blocks heartbeat
        """
        logging.info("[Main] Starting AMCL result polling task")
        last_log_time = 0

        while True:
            try:
                # Non-blocking check for results
                try:
                    result = self.amcl_output_queue.get_nowait()

                    # Handle different message types
                    msg_type = result.get('type', 'update')

                    if msg_type == 'init':
                        logging.info(f"[Main] AMCL worker ready: {result['msg']}")
                        continue
                    elif msg_type == 'warning':
                        logging.warning(f"[Main] {result['msg']}")
                        continue
                    elif msg_type == 'error':
                        logging.error(f"[Main] AMCL worker error: {result['msg']}")
                        continue
                    elif msg_type == 'fatal':
                        logging.error(f"[Main] AMCL worker fatal: {result['msg']}")
                        continue

                    # Update pose and confidence
                    self.amcl_pose = result['pose']
                    self.localization_confidence = result['confidence']
                    self.particles_cache = result['particles']

                    # Log first result
                    if not hasattr(self, '_first_result_logged'):
                        self._first_result_logged = True
                        logging.info(f"[Main] First AMCL result: pose={self.amcl_pose}, elapsed={result['elapsed']:.3f}s")

                    # Check if we just became localized
                    if self.localization_confidence > 0.3 and not self.is_localized:
                        self.is_localized = True
                        logging.info(f"Robot localized at: ({self.amcl_pose[0]:.2f}, {self.amcl_pose[1]:.2f})")
                        logging.info(f"Confidence: {self.localization_confidence:.3f}")
                        logging.info(f"Map bounds: x=[{self.map.origin[0]:.1f}, {self.map.origin[0] + self.map.height*self.map.resolution:.1f}], y=[{self.map.origin[1]:.1f}, {self.map.origin[1] + self.map.width*self.map.resolution:.1f}]")
                        logging.info("Ready! Click on map to set goal.")

                    # Update visualization periodically
                    if result['update_count'] % 5 == 0 and self.fig is not None:
                        self._update_visualization()

                    # Log periodically (every 5 seconds)
                    import time
                    current_time = time.time()
                    if current_time - last_log_time > 5.0:
                        logging.info(f"AMCL: pose=({self.amcl_pose[0]:.2f}, {self.amcl_pose[1]:.2f}), conf={self.localization_confidence:.2f}, process_time={result['elapsed']:.3f}s")
                        last_log_time = current_time

                except queue.Empty:
                    pass

                # Sleep to avoid busy-waiting (doesn't block other tasks)
                await asyncio.sleep(0.05)  # 20Hz polling

            except Exception as e:
                logging.error(f"Error polling AMCL results: {e}", exc_info=True)
                await asyncio.sleep(0.5)


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

            # Update particles from cache
            if len(self.particles_cache) > 0:
                particle_y = [p[1] for p in self.particles_cache]  # y coordinate
                particle_x = [p[0] for p in self.particles_cache]  # x coordinate
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


navigator_instance = None
navigator_ready_event = threading.Event()


async def main():
    navigator = None
    try:
        map_file = "go2_map.pkl"

        logging.info("=" * 60)
        logging.info("AMCL-based Autonomous Navigation (Multi-Process)")
        logging.info("=" * 60)

        # Connect to robot FIRST (before matplotlib initialization)
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")
        logging.info("Connecting to Go2...")
        await conn.connect()
        logging.info("Connected!")

        # Initialize navigator AFTER connection (matplotlib can take time)
        logging.info("Initializing navigator and GUI...")
        navigator = AMCLNavigator(map_file, loop=asyncio.get_running_loop(), enable_visualization=True)
        logging.info("Navigator initialized!")
        logging.info("AMCL processing runs in separate PROCESS - bypasses GIL, event loop stays responsive!")

        global navigator_instance
        navigator_instance = navigator
        navigator_ready_event.set()

        navigator.conn = conn
        navigator.register_connection_callbacks()

        # Configure data channel and subscriptions
        await navigator.on_connection_ready()

        logging.info("Robot setup complete and ready for navigation")

        logging.info("\nWaiting for AMCL to localize robot...")
        logging.info("Move the robot slowly to help localization converge.\n")

        # Keep running while async callbacks feed localization and planning
        logging.info("Running... Press Ctrl+C to stop")
        while True:
            await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("\nStopping...")
        if navigator and navigator.conn and navigator.is_navigating:
            await navigator._send_velocity_command(0.0, 0.0)
        if navigator:
            navigator.stop_amcl_worker()
        navigator_ready_event.set()
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)
        if navigator:
            navigator.stop_amcl_worker()
        navigator_ready_event.set()


if __name__ == "__main__":
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

    # Wait until navigator is ready, then initialize visualization on the main thread
    navigator_ready_event.wait()
    if navigator_instance is not None:
        navigator_instance.initialize_visualization()

    print("\nPress Ctrl+C to stop...")

    try:
        # Main thread just waits and services Matplotlib events
        while True:
            if navigator_instance and navigator_instance.fig is not None:
                plt.pause(0.05)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=2)
        sys.exit(0)
