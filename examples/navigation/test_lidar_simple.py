"""
간단한 라이다 테스트 - 앞쪽 장애물 감지 확인
"""

import asyncio
import logging
import numpy as np
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

logging.basicConfig(level=logging.INFO)


class SimpleLidarTest:
    def __init__(self):
        self.scan_count = 0
        self.robot_pos = None
        self.robot_yaw = None

    def sportmodestate_callback(self, message):
        """Odometry 업데이트"""
        data = message.get("data", {})
        position = data.get("position", [0.0, 0.0, 0.0])
        imu_state = data.get("imu_state", {})
        rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])

        self.robot_pos = np.array([position[0], position[1]])
        self.robot_yaw = rpy[2]

    def lidar_callback(self, message):
        """라이다 데이터 처리"""
        try:
            if self.robot_pos is None:
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

            # World coordinates로 변환
            world_points = origin + (voxel_indices * resolution)

            # Height 필터 (지면 근처만)
            height_mask = (world_points[:, 2] >= -0.2) & (world_points[:, 2] <= 1.0)
            filtered = world_points[height_mask]

            # 로봇 기준 상대 좌표로 변환 (평행이동만)
            relative = filtered[:, :2] - self.robot_pos

            # 2D 거리와 각도 계산
            ranges = np.linalg.norm(relative, axis=1)
            angles = np.arctan2(relative[:, 1], relative[:, 0])

            # 거리 필터
            valid_mask = (ranges >= 0.5) & (ranges <= 5.0)
            ranges = ranges[valid_mask]
            angles = angles[valid_mask]

            self.scan_count += 1

            if self.scan_count % 10 == 0:
                # 전방 ±30도 장애물 찾기
                front_mask = np.abs(angles) < np.radians(30)
                if np.sum(front_mask) > 0:
                    front_ranges = ranges[front_mask]
                    min_dist = front_ranges.min()
                    logging.info(f"[Scan {self.scan_count}] 전방 장애물: {min_dist:.2f}m (로봇 yaw: {np.degrees(self.robot_yaw):.1f}°)")
                else:
                    logging.info(f"[Scan {self.scan_count}] 전방 장애물 없음 (로봇 yaw: {np.degrees(self.robot_yaw):.1f}°)")

                # 방향별 포인트 수 출력
                front = np.sum(np.abs(angles) < np.radians(30))
                back = np.sum(np.abs(angles - np.pi) < np.radians(30)) + np.sum(np.abs(angles + np.pi) < np.radians(30))
                left = np.sum(np.abs(angles - np.pi/2) < np.radians(30))
                right = np.sum(np.abs(angles + np.pi/2) < np.radians(30))

                logging.info(f"  방향별 포인트: 전방={front}, 후방={back}, 좌={left}, 우={right}")

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
        test = SimpleLidarTest()

        # 구독
        conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", test.sportmodestate_callback)
        conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", test.lidar_callback)

        # 라이다 켜기
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
        logging.info("라이다 켜짐 - 장애물 감지 중...")
        logging.info("로봇 앞에 장애물을 놓고 감지되는지 확인하세요!")

        # 30초간 실행
        await asyncio.sleep(30)

        logging.info(f"\n총 {test.scan_count}개 스캔 처리됨")

    except Exception as e:
        logging.error(f"오류: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
