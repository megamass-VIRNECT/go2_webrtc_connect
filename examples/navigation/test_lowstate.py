"""
Test script to examine rt/lf/sportmodestate data structure
"""

import asyncio
import logging
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

logging.basicConfig(level=logging.INFO)

message_count = 0

def lowstate_callback(message):
    """Examine sportmodestate messages"""
    global message_count
    message_count += 1

    if message_count == 1:
        # Print full structure on first message
        data = message.get("data", {})
        logging.info(f"\n=== FIRST MESSAGE ===")
        logging.info(f"Message keys: {list(message.keys())}")
        logging.info(f"Data keys: {list(data.keys())}")

        # Check position and velocity
        position = data.get("position", [])
        velocity = data.get("velocity", [])
        logging.info(f"\nPosition: {position}")
        logging.info(f"Velocity: {velocity}")

        # Check imu_state
        imu_state = data.get("imu_state", {})
        logging.info(f"\nIMU State keys: {list(imu_state.keys())}")
        logging.info(f"IMU State data: {imu_state}")

        logging.info(f"\nFull data structure:")
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                logging.info(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                logging.info(f"  {key}: {value}")

    elif message_count % 50 == 0:
        # Print position/orientation periodically
        data = message.get("data", {})

        position = data.get("position", [])
        velocity = data.get("velocity", [])
        imu_state = data.get("imu_state", {})
        rpy = imu_state.get("rpy", [])

        logging.info(f"\nMessage {message_count}:")
        logging.info(f"  Position (x, y, z): {position}")
        logging.info(f"  Velocity (vx, vy, vz): {velocity}")
        if rpy:
            logging.info(f"  RPY (roll, pitch, yaw): {rpy}")

async def main():
    try:
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")

        logging.info("Connecting to Go2...")
        await conn.connect()
        logging.info("Connected!")

        await conn.datachannel.disableTrafficSaving(True)

        conn.datachannel.pub_sub.subscribe("rt/lf/sportmodestate", lowstate_callback)
        logging.info("Subscribed to rt/lf/sportmodestate")

        logging.info("\nWaiting 30 seconds... MOVE THE ROBOT to see changes!\n")

        await asyncio.sleep(30)

        logging.info(f"\n\nTotal messages received: {message_count}")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
