"""
Test movement with publish_request_new to get response
"""
import asyncio
import logging
import json
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
    
    # Check motion mode
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
            logging.info(f"Switching from '{current_mode}' to 'normal' mode...")
            response = await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "normal"}}
            )
            switch_status = response['data']['header']['status']
            logging.info(f"Mode switch response: {switch_status}")

            if switch_status['code'] == 0:
                logging.info("Mode switch command accepted, waiting for transition...")
                await asyncio.sleep(5)

                # Verify mode was actually changed
                logging.info("Verifying mode change...")
                verify_response = await conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["MOTION_SWITCHER"],
                    {"api_id": 1001}
                )

                if verify_response['data']['header']['status']['code'] == 0:
                    verify_data = json.loads(verify_response['data']['data'])
                    new_mode = verify_data['name']
                    logging.info(f"Current mode after switch attempt: {new_mode}")

                    if new_mode == "normal":
                        logging.info("✓ Successfully switched to 'normal' mode")
                    else:
                        logging.warning(f"✗ Mode switch failed: still in '{new_mode}' mode")
            else:
                logging.error(f"✗ Mode switch rejected with code: {switch_status['code']}")
                logging.warning(f"Continuing in '{current_mode}' mode...")
        else:
            logging.info("Already in 'normal' mode")
    
    # StandUp with response
    logging.info("Sending StandUp command...")
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["StandUp"], "parameter": {}}
    )
    logging.info(f"StandUp response: {response['data']['header']['status']}")
    await asyncio.sleep(3)

    # Enable continuous gait
    logging.info("Enabling continuous gait...")
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["ContinuousGait"], "parameter": {"flag": True}}
    )
    logging.info(f"ContinuousGait response: {response['data']['header']['status']}")
    await asyncio.sleep(1)
    
    # Move forward
    logging.info("Sending FORWARD Move commands...")
    for i in range(10):
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Move"], "parameter": {"x": 0.5, "y": 0.0, "z": 0.0}}
        )
        if i == 0:
            logging.info(f"First FORWARD Move response: {response['data']['header']['status']}")
        await asyncio.sleep(0.2)

    # Stop
    logging.info("Stopping...")
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["Move"], "parameter": {"x": 0.0, "y": 0.0, "z": 0.0}}
    )
    logging.info(f"Stop response: {response['data']['header']['status']}")
    await asyncio.sleep(1)

    # Move backward
    logging.info("Sending BACKWARD Move commands...")
    for i in range(10):
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Move"], "parameter": {"x": -0.5, "y": 0.0, "z": 0.0}}
        )
        if i == 0:
            logging.info(f"First BACKWARD Move response: {response['data']['header']['status']}")
        await asyncio.sleep(0.2)

    # Final Stop
    logging.info("Final stopping...")
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["Move"], "parameter": {"x": 0.0, "y": 0.0, "z": 0.0}}
    )
    logging.info(f"Final stop response: {response['data']['header']['status']}")
    
    await asyncio.sleep(2)
    logging.info("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_movement())
