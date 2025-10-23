"""
Simple test to verify robot can move
"""
import asyncio
import logging
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

logging.basicConfig(level=logging.INFO)

async def test_movement():
    # Connect
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")
    logging.info("Connecting...")
    await conn.connect()
    logging.info("Connected!")
    
    await asyncio.sleep(1)
    
    # Stand up
    logging.info("Standing up...")
    conn.datachannel.pub_sub.publish_without_callback(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["StandUp"], "parameter": {}}
    )
    await asyncio.sleep(3)
    
    # Enable continuous gait
    logging.info("Enabling continuous gait...")
    conn.datachannel.pub_sub.publish_without_callback(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["ContinuousGait"], "parameter": {"flag": True}}
    )
    await asyncio.sleep(1)
    
    # Test forward movement
    logging.info("Moving forward for 2 seconds...")
    for i in range(10):
        conn.datachannel.pub_sub.publish_without_callback(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Move"], "parameter": {"x": 0.3, "y": 0.0, "z": 0.0}}
        )
        await asyncio.sleep(0.2)
    
    # Stop
    logging.info("Stopping...")
    conn.datachannel.pub_sub.publish_without_callback(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["Move"], "parameter": {"x": 0.0, "y": 0.0, "z": 0.0}}
    )
    await asyncio.sleep(1)
    
    logging.info("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_movement())
