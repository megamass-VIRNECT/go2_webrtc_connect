"""
Check and set motion mode
"""
import asyncio
import logging
import json
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

logging.basicConfig(level=logging.INFO)

async def test_motion():
    # Connect
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="192.168.1.7")
    logging.info("Connecting...")
    await conn.connect()
    logging.info("Connected!")
    
    await asyncio.sleep(1)
    
    # Check current motion mode
    logging.info("Checking motion mode...")
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["MOTION_SWITCHER"],
        {"api_id": 1001}
    )
    
    if response['data']['header']['status']['code'] == 0:
        data = json.loads(response['data']['data'])
        current_mode = data['name']
        logging.info(f"Current motion mode: {current_mode}")
        
        if current_mode != "normal":
            logging.info(f"Switching from {current_mode} to 'normal'...")
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "normal"}}
            )
            await asyncio.sleep(5)
            logging.info("Mode switched!")
        else:
            logging.info("Already in 'normal' mode")
    
    # Stand up
    logging.info("Standing up...")
    conn.datachannel.pub_sub.publish_without_callback(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["StandUp"], "parameter": {}}
    )
    await asyncio.sleep(3)
    
    # Damping OFF
    logging.info("Disabling damping...")
    conn.datachannel.pub_sub.publish_without_callback(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["Damp"], "parameter": {}}
    )
    await asyncio.sleep(1)
    
    # Balance Stand
    logging.info("Balance stand...")
    conn.datachannel.pub_sub.publish_without_callback(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["BalanceStand"], "parameter": {}}
    )
    await asyncio.sleep(2)
    
    # Move forward
    logging.info("Moving forward...")
    for i in range(15):
        conn.datachannel.pub_sub.publish_without_callback(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Move"], "parameter": {"x": 0.5, "y": 0.0, "z": 0.0}}
        )
        logging.info(f"Sent move command {i+1}/15")
        await asyncio.sleep(0.2)
    
    # Stop
    logging.info("Stopping...")
    for i in range(5):
        conn.datachannel.pub_sub.publish_without_callback(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Move"], "parameter": {"x": 0.0, "y": 0.0, "z": 0.0}}
        )
        await asyncio.sleep(0.1)
    
    await asyncio.sleep(2)
    logging.info("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_motion())
