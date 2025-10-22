"""
Test script to see what topics are being received
"""

import asyncio
import logging
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

logging.basicConfig(level=logging.INFO)

received_topics = set()

def generic_callback(message):
    """Log all received messages"""
    topic = message.get("topic", "unknown")
    if topic not in received_topics:
        received_topics.add(topic)
        logging.info(f"NEW TOPIC RECEIVED: {topic}")
        logging.info(f"  Message keys: {list(message.keys())}")
        if "data" in message:
            data = message["data"]
            if isinstance(data, dict):
                logging.info(f"  Data keys: {list(data.keys())}")

async def main():
    try:
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")

        logging.info("Connecting to Go2...")
        await conn.connect()
        logging.info("Connected!")

        await conn.datachannel.disableTrafficSaving(True)

        # Subscribe to common topics
        topics = [
            "rt/sportmodestate",
            "rt/lf/sportmodestate",  # Try with lf prefix!
            "rt/lf/lowstate",
            "rt/lowstate",
            "rt/state",
            "rt/odom",
            "rt/imu",
            "rt/lf/odom",
            "rt/lf/imu",
        ]

        for topic in topics:
            conn.datachannel.pub_sub.subscribe(topic, generic_callback)
            logging.info(f"Subscribed to {topic}")

        logging.info("\nWaiting for messages for 20 seconds...")
        logging.info("Move the robot to generate odometry data!\n")

        await asyncio.sleep(20)

        logging.info(f"\n\nSummary: Received messages from {len(received_topics)} topics:")
        for topic in sorted(received_topics):
            logging.info(f"  - {topic}")

    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
