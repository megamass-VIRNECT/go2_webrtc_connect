"""
로봇 좌표계 테스트
로봇의 실제 방향과 odometry/lidar 데이터 비교
"""

import asyncio
import logging
import numpy as np
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

logging.basicConfig(level=logging.INFO)


class CoordinateTest:
    def __init__(self):
        self.odom_data = None
        self.lidar_scan_count = 0

    def sportmodestate_callback(self, message):
        """Odometry 데이터 저장"""
        data = message.get("data", {})
        position = data.get("position", [0.0, 0.0, 0.0])
        imu_state = data.get("imu_state", {})
        rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])

        self.odom_data = {
            'x': position[0],
            'y': position[1],
            'z': position[2],
            'roll': rpy[0],
            'pitch': rpy[1],
            'yaw': rpy[2]
        }

    def lidar_callback(self, message):
        """라이다 데이터 분석 - 전방 장애물 방향 확인"""
        try:
            if self.odom_data is None:
                return

            self.lidar_scan_count += 1

            if self.lidar_scan_count != 10:  # 10번째 스캔만 분석
                return

            data = message.get("data", {})
            data_inner = data.get("data", {})
            positions_raw = data_inner.get("positions", [])

            if positions_raw is None or len(positions_raw) == 0:
                return

            # Voxel indices
            positions_np = np.asarray(positions_raw, dtype=np.float32)
            if positions_np.ndim == 1:
                voxel_indices = positions_np.reshape(-1, 3)
            else:
                voxel_indices = positions_np

            # Voxel map parameters
            origin = np.array(data.get("origin", [0.0, 0.0, 0.0]), dtype=np.float32)
            resolution = data.get("resolution", 0.05)

            # World coordinates
            world_points = origin + (voxel_indices * resolution)

            # Robot position
            robot_x = self.odom_data['x']
            robot_y = self.odom_data['y']
            robot_yaw = self.odom_data['yaw']

            # Relative coordinates (world frame, robot-centered)
            relative_x = world_points[:, 0] - robot_x
            relative_y = world_points[:, 1] - robot_y

            # Height filter
            height_mask = (world_points[:, 2] >= -0.2) & (world_points[:, 2] <= 1.0)
            relative_x = relative_x[height_mask]
            relative_y = relative_y[height_mask]

            # Distance and angle in WORLD frame (NOT rotated)
            ranges_world = np.sqrt(relative_x**2 + relative_y**2)
            angles_world = np.arctan2(relative_y, relative_x)  # atan2(Y, X)

            # Filter by distance
            valid_mask = (ranges_world >= 0.5) & (ranges_world <= 3.0)
            relative_x = relative_x[valid_mask]
            relative_y = relative_y[valid_mask]
            ranges_world = ranges_world[valid_mask]
            angles_world = angles_world[valid_mask]

            logging.info("=" * 80)
            logging.info("좌표계 분석 (World Frame, Robot-Centered)")
            logging.info("=" * 80)
            logging.info(f"로봇 Odometry:")
            logging.info(f"  Position: ({robot_x:.3f}, {robot_y:.3f})")
            logging.info(f"  Yaw: {np.degrees(robot_yaw):.1f}°")
            logging.info("")

            # 방향별 포인트 분포 (world frame 기준)
            logging.info("World Frame에서 방향별 포인트 분포:")
            logging.info("  (로봇 중심 기준, 아직 회전하지 않음)")

            # +X 방향 (0°)
            mask_px = np.abs(angles_world - 0) < np.radians(15)
            logging.info(f"  +X 방향 (0°): {np.sum(mask_px)}개")
            if np.sum(mask_px) > 0:
                logging.info(f"    평균 거리: {ranges_world[mask_px].mean():.2f}m")

            # +Y 방향 (90°)
            mask_py = np.abs(angles_world - np.pi/2) < np.radians(15)
            logging.info(f"  +Y 방향 (90°): {np.sum(mask_py)}개")
            if np.sum(mask_py) > 0:
                logging.info(f"    평균 거리: {ranges_world[mask_py].mean():.2f}m")

            # -X 방향 (180°)
            mask_nx = (np.abs(angles_world - np.pi) < np.radians(15)) | (np.abs(angles_world + np.pi) < np.radians(15))
            logging.info(f"  -X 방향 (180°): {np.sum(mask_nx)}개")
            if np.sum(mask_nx) > 0:
                logging.info(f"    평균 거리: {ranges_world[mask_nx].mean():.2f}m")

            # -Y 방향 (-90°)
            mask_ny = np.abs(angles_world + np.pi/2) < np.radians(15)
            logging.info(f"  -Y 방향 (-90°): {np.sum(mask_ny)}개")
            if np.sum(mask_ny) > 0:
                logging.info(f"    평균 거리: {ranges_world[mask_ny].mean():.2f}m")

            logging.info("")
            logging.info("해석:")
            logging.info("  - 로봇 앞에 장애물이 있다면, 어느 방향에 포인트가 많이 나타나는지 확인하세요.")
            logging.info("  - 로봇의 '전방'이 World frame의 어느 방향인지 확인할 수 있습니다.")
            logging.info("  - Yaw 각도와 비교하여 좌표계 규약을 파악하세요.")
            logging.info("")

            # Now rotate to body frame
            cos_yaw = np.cos(-robot_yaw)
            sin_yaw = np.sin(-robot_yaw)
            body_x = relative_x * cos_yaw - relative_y * sin_yaw
            body_y = relative_x * sin_yaw + relative_y * cos_yaw

            ranges_body = np.sqrt(body_x**2 + body_y**2)
            angles_body = np.arctan2(body_y, body_x)

            logging.info("Body Frame에서 방향별 포인트 분포:")
            logging.info("  (로봇 yaw만큼 회전 후)")

            # Body +X (로봇 전방)
            mask_front = np.abs(angles_body) < np.radians(15)
            logging.info(f"  Body +X (전방): {np.sum(mask_front)}개")
            if np.sum(mask_front) > 0:
                logging.info(f"    평균 거리: {ranges_body[mask_front].mean():.2f}m")

            logging.info("=" * 80)

        except Exception as e:
            logging.error(f"Error: {e}", exc_info=True)


async def main():
    try:
        # 연결
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")
        logging.info("연결 중...")
        await conn.connect()
        logging.info("연결 성공!")

        await conn.datachannel.disableTrafficSaving(True)
        conn.datachannel.set_decoder(decoder_type='libvoxel')

        # 테스트 초기화
        test = CoordinateTest()

        # 구독
        conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", test.sportmodestate_callback)
        conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", test.lidar_callback)

        # 라이다 켜기
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")

        logging.info("")
        logging.info("=" * 80)
        logging.info("좌표계 테스트 시작")
        logging.info("=" * 80)
        logging.info("로봇 앞에 장애물을 놓고 어느 방향에서 감지되는지 확인하세요.")
        logging.info("10번째 스캔 데이터를 분석합니다...")
        logging.info("=" * 80)
        logging.info("")

        # 5초간 실행
        await asyncio.sleep(5)

    except Exception as e:
        logging.error(f"오류: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
