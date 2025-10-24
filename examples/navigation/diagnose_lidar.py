"""
라이다 데이터 좌표계 진단 스크립트
실제 라이다 데이터가 어떤 좌표계를 사용하는지 확인
"""

import asyncio
import logging
import numpy as np
import matplotlib.pyplot as plt
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

logging.basicConfig(level=logging.INFO)


class LidarDiagnostic:
    def __init__(self):
        self.scan_count = 0
        self.first_scan_data = None

    def lidar_callback(self, message):
        """라이다 데이터 분석"""
        try:
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
            origin = data.get("origin", [0.0, 0.0, 0.0])
            width = data.get("width", [128, 128, 38])
            resolution = data.get("resolution", 0.05)

            # World coordinates로 변환
            origin_np = np.asarray(origin, dtype=np.float32)
            world_points = origin_np + (voxel_indices * resolution)

            self.scan_count += 1

            if self.scan_count == 1:
                logging.info("=" * 80)
                logging.info("첫 번째 라이다 스캔 데이터 분석")
                logging.info("=" * 80)
                logging.info(f"Voxel map origin: {origin}")
                logging.info(f"Voxel map width: {width}")
                logging.info(f"Voxel map resolution: {resolution}m")
                logging.info(f"총 포인트 개수: {len(voxel_indices)}")
                logging.info("")

                logging.info("Voxel indices 범위:")
                logging.info(f"  X: [{voxel_indices[:,0].min():.1f}, {voxel_indices[:,0].max():.1f}]")
                logging.info(f"  Y: [{voxel_indices[:,1].min():.1f}, {voxel_indices[:,1].max():.1f}]")
                logging.info(f"  Z: [{voxel_indices[:,2].min():.1f}, {voxel_indices[:,2].max():.1f}]")
                logging.info("")

                logging.info("World coordinates 범위:")
                logging.info(f"  X: [{world_points[:,0].min():.2f}m, {world_points[:,0].max():.2f}m]")
                logging.info(f"  Y: [{world_points[:,1].min():.2f}m, {world_points[:,1].max():.2f}m]")
                logging.info(f"  Z: [{world_points[:,2].min():.2f}m, {world_points[:,2].max():.2f}m]")
                logging.info("")

                # Voxel map 중심 계산
                map_center_x = origin[0] + (width[0] * resolution) / 2.0
                map_center_y = origin[1] + (width[1] * resolution) / 2.0
                logging.info(f"Voxel map 중심 (추정 로봇 위치): ({map_center_x:.2f}, {map_center_y:.2f})")
                logging.info("")

                # Robot-centered로 변환
                robot_centered = world_points.copy()
                robot_centered[:, 0] -= map_center_x
                robot_centered[:, 1] -= map_center_y

                logging.info("Robot-centered coordinates 범위:")
                logging.info(f"  X: [{robot_centered[:,0].min():.2f}m, {robot_centered[:,0].max():.2f}m]")
                logging.info(f"  Y: [{robot_centered[:,1].min():.2f}m, {robot_centered[:,1].max():.2f}m]")
                logging.info(f"  Z: [{robot_centered[:,2].min():.2f}m, {robot_centered[:,2].max():.2f}m]")
                logging.info("")

                # 2D 거리 및 각도 계산
                ranges_2d = np.sqrt(robot_centered[:, 0]**2 + robot_centered[:, 1]**2)
                angles_2d = np.arctan2(robot_centered[:, 1], robot_centered[:, 0])

                logging.info("2D 거리 및 각도:")
                logging.info(f"  거리: [{ranges_2d.min():.2f}m, {ranges_2d.max():.2f}m]")
                logging.info(f"  각도: [{np.degrees(angles_2d.min()):.1f}°, {np.degrees(angles_2d.max()):.1f}°]")
                logging.info("")

                # 방향별 포인트 분포 확인
                front_mask = np.abs(angles_2d) < np.radians(30)
                back_mask = (np.abs(angles_2d - np.pi) < np.radians(30)) | (np.abs(angles_2d + np.pi) < np.radians(30))
                left_mask = np.abs(angles_2d - np.pi/2) < np.radians(30)
                right_mask = np.abs(angles_2d + np.pi/2) < np.radians(30)

                logging.info("방향별 포인트 분포:")
                logging.info(f"  전방 (±30°): {np.sum(front_mask)}개")
                if np.sum(front_mask) > 0:
                    logging.info(f"    평균 거리: {ranges_2d[front_mask].mean():.2f}m")
                    logging.info(f"    최소 거리: {ranges_2d[front_mask].min():.2f}m")

                logging.info(f"  후방 (180°±30°): {np.sum(back_mask)}개")
                if np.sum(back_mask) > 0:
                    logging.info(f"    평균 거리: {ranges_2d[back_mask].mean():.2f}m")

                logging.info(f"  좌측 (90°±30°): {np.sum(left_mask)}개")
                if np.sum(left_mask) > 0:
                    logging.info(f"    평균 거리: {ranges_2d[left_mask].mean():.2f}m")

                logging.info(f"  우측 (-90°±30°): {np.sum(right_mask)}개")
                if np.sum(right_mask) > 0:
                    logging.info(f"    평균 거리: {ranges_2d[right_mask].mean():.2f}m")

                logging.info("=" * 80)

                # 데이터 저장 (시각화용)
                self.first_scan_data = {
                    'robot_centered': robot_centered,
                    'ranges_2d': ranges_2d,
                    'angles_2d': angles_2d,
                    'origin': origin,
                    'width': width,
                    'resolution': resolution
                }

        except Exception as e:
            logging.error(f"Error in lidar callback: {e}", exc_info=True)

    def sportmodestate_callback(self, message):
        """Odometry 데이터 분석"""
        if self.scan_count == 1:  # 첫 스캔과 동시에 출력
            data = message.get("data", {})
            position = data.get("position", [0.0, 0.0, 0.0])
            imu_state = data.get("imu_state", {})
            rpy = imu_state.get("rpy", [0.0, 0.0, 0.0])

            logging.info("")
            logging.info("Odometry 데이터:")
            logging.info(f"  Position (world): {position}")
            logging.info(f"  RPY: {[np.degrees(x) for x in rpy]}")
            logging.info("")

    def visualize(self):
        """첫 번째 스캔 데이터 시각화"""
        if self.first_scan_data is None:
            logging.warning("시각화할 데이터가 없습니다.")
            return

        robot_centered = self.first_scan_data['robot_centered']

        # Z 필터링 (지면 근처만)
        height_mask = (robot_centered[:, 2] >= -0.2) & (robot_centered[:, 2] <= 1.0)
        filtered = robot_centered[height_mask]

        # 2D plot (top view)
        plt.figure(figsize=(12, 10))

        # X-Y plot (robot centered)
        plt.subplot(2, 2, 1)
        plt.scatter(filtered[:, 0], filtered[:, 1], s=1, alpha=0.5)
        plt.scatter(0, 0, c='red', s=100, marker='o', label='Robot')
        plt.arrow(0, 0, 1, 0, head_width=0.3, head_length=0.5, fc='red', ec='red')
        plt.xlabel('X (forward, meters)')
        plt.ylabel('Y (left, meters)')
        plt.title('Top View (Robot Centered)')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()

        # Polar plot
        plt.subplot(2, 2, 2, projection='polar')
        ranges_2d = np.sqrt(filtered[:, 0]**2 + filtered[:, 1]**2)
        angles_2d = np.arctan2(filtered[:, 1], filtered[:, 0])
        plt.scatter(angles_2d, ranges_2d, s=1, alpha=0.5)
        plt.title('Polar View')

        # X-Z plot (side view)
        plt.subplot(2, 2, 3)
        plt.scatter(filtered[:, 0], filtered[:, 2], s=1, alpha=0.5)
        plt.xlabel('X (forward, meters)')
        plt.ylabel('Z (up, meters)')
        plt.title('Side View (X-Z)')
        plt.grid(True)
        plt.axhline(y=0, color='r', linestyle='--', label='Ground level')
        plt.legend()

        # Y-Z plot (side view)
        plt.subplot(2, 2, 4)
        plt.scatter(filtered[:, 1], filtered[:, 2], s=1, alpha=0.5)
        plt.xlabel('Y (left, meters)')
        plt.ylabel('Z (up, meters)')
        plt.title('Side View (Y-Z)')
        plt.grid(True)
        plt.axhline(y=0, color='r', linestyle='--', label='Ground level')
        plt.legend()

        plt.tight_layout()
        plt.savefig('lidar_diagnostic.png', dpi=150, bbox_inches='tight')
        logging.info("시각화 저장: lidar_diagnostic.png")
        plt.show()


async def main():
    try:
        # 연결
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")
        logging.info("Go2에 연결 중...")
        await conn.connect()
        logging.info("연결 성공!")

        # Traffic saving 비활성화
        await conn.datachannel.disableTrafficSaving(True)

        # Decoder 설정
        conn.datachannel.set_decoder(decoder_type='libvoxel')

        # Diagnostic 초기화
        diagnostic = LidarDiagnostic()

        # 구독
        conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", diagnostic.sportmodestate_callback)
        conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", diagnostic.lidar_callback)

        # 라이다 켜기
        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
        logging.info("라이다 켜짐 - 데이터 수집 중...")

        # 첫 스캔 받을 때까지 대기
        while diagnostic.scan_count < 1:
            await asyncio.sleep(0.1)

        # 추가로 몇 초 더 대기 (여러 스캔 확인)
        await asyncio.sleep(3)

        logging.info(f"\n총 {diagnostic.scan_count}개 스캔 수신됨")

        # 시각화
        diagnostic.visualize()

    except Exception as e:
        logging.error(f"오류 발생: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
