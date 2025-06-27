# api/webrtc_api.py

import json
import time
import hashlib
from flask import Blueprint, request, jsonify
from go2_webrtc_driver.util import (
    fetch_token,
    fetch_public_key,
    fetch_turn_server_info,
)
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection
from go2_webrtc_driver.constants import WebRTCConnectionMethod
from go2_webrtc_driver.unitree_auth import send_sdp_to_remote_peer, send_sdp_to_local_peer
from go2_webrtc_driver.load_webrtc_config import load_webrtc_config
from go2_webrtc_driver.multicast_scanner import discover_ip_sn

webrtc_api = Blueprint("webrtc_api", __name__)
config = load_webrtc_config()
sn = config["sn"]
email = config["user"]["email"]
password = config["user"]["password"]
local_turn_server_info = config["iceServerInfo"]
token = None
public_key = None
remote_turn_server_info = None

@webrtc_api.route("/api/fetch-remote-configuration", methods=["POST"])
def fetch_remote_configuration():
    global token
    global public_key
    global remote_turn_server_info

    try:
        ensure_webrtc_session()

        conn = Go2WebRTCConnection(connectionMethod=WebRTCConnectionMethod.Remote)
        configuration = conn.create_webrtc_configuration(remote_turn_server_info)

        return jsonify(configuration)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@webrtc_api.route("/api/fetch-local-configuration", methods=["POST"])
def fetch_local_configuration():
    return jsonify(local_turn_server_info)

@webrtc_api.route("/api/send-remote-offer", methods=["POST"])
def send_remote_offer():
    data = request.get_json()
    local_description = data.get("local_description")  # { "type": "...", "sdp": "..." }

    try:
        ensure_webrtc_session()

        # send SDP offer to remote peer
        sdp_offer_json = {
            "id": "",
            "turnserver": remote_turn_server_info,
            "sdp": local_description["sdp"],
            "type": local_description["type"],
            "token": token
        }

        peer_answer_json = send_sdp_to_remote_peer(sn, json.dumps(sdp_offer_json), token, public_key)

        return jsonify(json.loads(peer_answer_json))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@webrtc_api.route("/api/send-local-offer", methods=["POST"])
def send_local_offer():
    data = request.get_json()
    local_description = data.get("local_description")  # { "type": "...", "sdp": "..." }

    try:
        # send SDP offer to local peer
        sdp_offer_json = {
            "id": "",
            "sdp": local_description["sdp"],
            "type": local_description["type"]
        }

        peer_answer_json = send_sdp_to_local_peer(discover_ip(), json.dumps(sdp_offer_json))

        return jsonify(json.loads(peer_answer_json))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def ensure_webrtc_session():
    global token, public_key, remote_turn_server_info

    if token and public_key and remote_turn_server_info:
        return

    token = fetch_token(email, password)
    if not token:
        raise ValueError("Invalid email or password")

    public_key = fetch_public_key()
    if not public_key:
        raise ValueError("Failed to fetch public key")

    remote_turn_server_info = fetch_turn_server_info(sn, token, public_key)
    if not remote_turn_server_info:
        raise ValueError("Failed to fetch TURN server info")

def discover_ip():
    discovered_ip_sn_addresses = discover_ip_sn()
    if discovered_ip_sn_addresses:
        if sn in discovered_ip_sn_addresses:
            return discovered_ip_sn_addresses[sn]
        else:
            raise ValueError("The provided serial number wasn't found on the network. Provide an IP address instead.")
    else:
        raise ValueError("No devices found on the network. Provide an IP address instead.")