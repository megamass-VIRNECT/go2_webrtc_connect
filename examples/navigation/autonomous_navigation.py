"""
Autonomous Navigation Example
Load map, set goal, and autonomously navigate to it
"""

import asyncio
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.navigation import load_map
from go2_webrtc_driver.navigation.planning import AStarPlanner
from go2_webrtc_driver.navigation.control import PurePursuitController

logging.basicConfig(level=logging.INFO)


class AutonomousNavigator:
    """Autonomous navigation system"""

    def __init__(self, map_file: str):
        # Load map
        self.map = load_map(map_file)
        logging.info(f"Map loaded: {self.map.width}x{self.map.height}, resolution={self.map.resolution}m")

        # Initialize planner and controller
        self.planner = AStarPlanner(inflation_radius=0.3)
        self.controller = PurePursuitController(
            look_ahead_distance=0.8,
            max_linear_velocity=0.3,
            max_angular_velocity=0.6,
            goal_tolerance=0.3
        )

        # Current state
        self.current_pose = (0.0, 0.0, 0.0)  # (x, y, theta)
        self.prev_odom = None
        self.path = None
        self.goal = None

        # Visualization
        self.fig = None
        self.ax = None
        self.map_plot = None
        self.path_plot = None
        self.robot_plot = None
        self.robot_arrow = None
        self.goal_plot = None

        # WebRTC connection
        self.conn = None

        self._setup_visualization()

    def _setup_visualization(self):
        """Setup matplotlib visualization with interactive goal selection"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.set_xlabel('Y (meters)')
        self.ax.set_ylabel('X (meters)')
        self.ax.set_title('Autonomous Navigation - Click to set goal')
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
        self.path_plot, = self.ax.plot([], [], 'g-', linewidth=2, label='Planned Path')
        self.robot_plot, = self.ax.plot([], [], 'bo', markersize=10, label='Robot')
        self.robot_arrow = None
        self.goal_plot, = self.ax.plot([], [], 'r*', markersize=20, label='Goal')

        self.ax.legend()
        plt.tight_layout()

        # Connect click event for goal selection
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        plt.show(block=False)

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
        if event.inaxes != self.ax:
            return

        # Get clicked position (in plot coordinates: y, x)
        goal_y = event.xdata
        goal_x = event.ydata

        logging.info(f"Goal set: ({goal_x:.2f}, {goal_y:.2f})")

        # Plan path
        asyncio.create_task(self._plan_and_execute(goal_x, goal_y))

    async def _plan_and_execute(self, goal_x: float, goal_y: float):
        """Plan path and execute navigation"""
        self.goal = (goal_x, goal_y)

        # Update goal visualization
        self.goal_plot.set_data([goal_y], [goal_x])
        self.fig.canvas.draw_idle()

        # Plan path
        logging.info("Planning path...")
        start = (self.current_pose[0], self.current_pose[1])
        self.path = self.planner.plan(self.map, start, self.goal)

        if self.path is None:
            logging.error("Failed to plan path!")
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
        logging.info("Starting navigation...")
        await self._navigate()

    async def _navigate(self):
        """Execute navigation along planned path"""
        if self.conn is None:
            logging.error("Not connected to robot")
            return

        rate = 0.1  # 10 Hz control loop

        while True:
            # Compute control command
            linear_vel, angular_vel, goal_reached = self.controller.compute_control(self.current_pose)

            if goal_reached:
                logging.info("Goal reached!")
                # Stop robot
                await self._send_velocity_command(0.0, 0.0)
                break

            # Send velocity command
            await self._send_velocity_command(linear_vel, angular_vel)

            # Update visualization
            self._update_visualization()

            # Progress info
            if self.controller.current_waypoint_index % 10 == 0:
                progress = self.controller.get_progress() * 100
                logging.info(f"Navigation progress: {progress:.1f}%")

            await asyncio.sleep(rate)

    async def _send_velocity_command(self, linear: float, angular: float):
        """Send velocity command to robot"""
        # Use sportmode API to control robot
        # Format: {"x": linear_vel, "y": 0, "z": angular_vel}
        command = {
            "x": float(linear),
            "y": 0.0,
            "z": float(angular)
        }

        try:
            self.conn.datachannel.pub_sub.publish_without_callback(
                "rt/api/sport/request",
                {"api_id": 1008, "parameter": command}  # Move command
            )
        except Exception as e:
            logging.error(f"Failed to send velocity command: {e}")

    def sportmodestate_callback(self, message):
        """Handle odometry updates"""
        try:
            data = message.get("data", {})
            position = data.get("position", [0.0, 0.0, 0.0])
            imu_state = data.get("imu_state", {})
            rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])

            current_odom = (position[0], position[1], rpy[2])

            if self.prev_odom is not None:
                dx = current_odom[0] - self.prev_odom[0]
                dy = current_odom[1] - self.prev_odom[1]
                dtheta = current_odom[2] - self.prev_odom[2]

                # Normalize angle
                while dtheta > np.pi:
                    dtheta -= 2 * np.pi
                while dtheta < -np.pi:
                    dtheta += 2 * np.pi

                # Update pose
                x, y, theta = self.current_pose
                x += dx
                y += dy
                theta += dtheta
                theta = np.arctan2(np.sin(theta), np.cos(theta))

                self.current_pose = (x, y, theta)

            self.prev_odom = current_odom

        except Exception as e:
            logging.error(f"Error in odometry callback: {e}")

    def _update_visualization(self):
        """Update visualization"""
        try:
            # Update robot position
            x, y, theta = self.current_pose
            self.robot_plot.set_data([y], [x])

            # Update robot direction arrow
            if self.robot_arrow is not None:
                self.robot_arrow.remove()

            arrow_length = 0.5
            dx = arrow_length * np.cos(theta)
            dy = arrow_length * np.sin(theta)
            self.robot_arrow = self.ax.arrow(y, x, dy, dx,
                                            head_width=0.2, head_length=0.3,
                                            fc='red', ec='red', alpha=0.8)

            # Update canvas without blocking
            self.fig.canvas.draw_idle()

        except Exception as e:
            logging.error(f"Error updating visualization: {e}")


async def main():
    try:
        # Check if map file exists
        map_file = "go2_map.pkl"

        logging.info("=" * 60)
        logging.info("Autonomous Navigation System")
        logging.info("=" * 60)

        # Initialize navigator
        navigator = AutonomousNavigator(map_file)

        # Connect to robot
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")
        logging.info("Connecting to Go2...")
        await conn.connect()
        logging.info("Connected!")

        navigator.conn = conn

        await conn.datachannel.disableTrafficSaving(True)

        # Subscribe to odometry
        conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", navigator.sportmodestate_callback)
        logging.info("Subscribed to odometry")

        logging.info("\nReady! Click on the map to set a goal.")
        logging.info("The robot will autonomously navigate to the clicked location.")
        logging.info("Press Ctrl+C to stop.\n")

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logging.info("\nStopping navigation...")
        if navigator.conn:
            await navigator._send_velocity_command(0.0, 0.0)
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
