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
from go2_webrtc_driver.unitree_auth import send_sdp_to_remote_peer

webrtc_api = Blueprint("webrtc_api", __name__)

# 전역 세션 캐시
session_cache = {}
SESSION_TTL_SECONDS = 300  # 5분

def make_session_key(email: str, password: str) -> str:
    """세션을 구분하기 위한 해시 키 생성"""
    raw = f"{email}:{password}"
    return hashlib.sha256(raw.encode()).hexdigest()

@webrtc_api.route("/api/fetch-configuration", methods=["POST"])
def fetch_connection_configuration():
    data = request.get_json()
    sn = data.get("sn")
    email = data.get("email")
    password = data.get("password")

    if not email or not password or not sn:
        return jsonify({"error": "Missing email or password or sn"}), 400

    try:
        token = fetch_token(email, password)
        if not token:
            return jsonify({"error": "Invalid email or password"}), 401

        public_key = fetch_public_key()
        if not public_key:
            return jsonify({"error": "Failed to fetch public key"}), 500

        turn_server_info = fetch_turn_server_info(sn, token, public_key)
        if not turn_server_info:
            return jsonify({"error": "Failed to fetch turn_server_info"}), 500

        conn = Go2WebRTCConnection(connectionMethod=WebRTCConnectionMethod.Remote)
        configuration = conn.create_webrtc_configuration(turn_server_info)

        # 세션 캐시에 저장
        session_key = make_session_key(email, password)
        session_cache[session_key] = {
            "token": token,
            "public_key": public_key,
            "turn_server_info": turn_server_info,
            "sn": sn,
            "timestamp": time.time()
        }

        return jsonify(configuration)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@webrtc_api.route("/api/send-offer", methods=["POST"])
def send_offer():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    local_description = data.get("local_description")  # { "type": "...", "sdp": "..." }

    if not email or not password or not local_description:
        return jsonify({"error": "Missing required parameters"}), 400

    session_key = make_session_key(email, password)
    session = session_cache.get(session_key)

    if not session:
        return jsonify({"error": "Session not found"}), 403

    if time.time() - session["timestamp"] > SESSION_TTL_SECONDS:
        del session_cache[session_key]
        return jsonify({"error": "Session expired"}), 403

    try:
        sn = session["sn"]
        token = session["token"]
        public_key = session["public_key"]
        turn_server_info = session["turn_server_info"]

        # send SDP offer to remote peer
        sdp_offer_json = {
            "id": "",
            "turnserver": turn_server_info,
            "sdp": local_description["sdp"],
            "type": local_description["type"],
            "token": token
        }

        peer_answer_json = send_sdp_to_remote_peer(sn, json.dumps(sdp_offer_json), token, public_key)

        return jsonify(json.loads(peer_answer_json))

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# import json
# from flask import Blueprint, request, jsonify
# from go2_webrtc_driver.util import fetch_token, fetch_public_key, fetch_turn_server_info
# from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection
# from go2_webrtc_driver.constants import WebRTCConnectionMethod
# from go2_webrtc_driver.unitree_auth import send_sdp_to_remote_peer
#
# webrtc_api = Blueprint("webrtc_api", __name__)
#
# @webrtc_api.route("/api/fetch-configuration", methods=["POST"])
# def fetch_connection_configuration():
#     data = request.get_json()
#     sn = data.get("sn")
#     email = data.get("email")
#     password = data.get("password")
#
#     if not email or not password or not sn:
#         return jsonify({"error": "Missing email or password or sn"}), 400
#
#     try:
#         token = fetch_token(email, password)
#         if not token:
#             return jsonify({"error": "Invalid email or password"}), 401
#
#         public_key = fetch_public_key()
#         if not public_key:
#             return jsonify({"error": "Failed to fetch public key"}), 500
#
#         turn_server_info = fetch_turn_server_info(sn, token, public_key)
#         if not turn_server_info:
#             return jsonify({"error": "Failed to fetch turn_server_info"}), 500
#
#         conn = Go2WebRTCConnection(connectionMethod=WebRTCConnectionMethod.Remote)
#         rtc_config = conn.create_webrtc_configuration(turn_server_info)
#
#         rtc_config_json = {
#             "iceServers": [
#                 {
#                     "urls": server.urls,
#                     "username": getattr(server, "username", None),
#                     "credential": getattr(server, "credential", None)
#                 }
#                 for server in rtc_config.iceServers
#             ]
#         }
#
#         return jsonify({
#             "sn": sn,
#             "token": token,
#             "public_key": public_key.export_key().decode() if hasattr(public_key, "export_key") else str(public_key),
#             "turn_server_info": turn_server_info,
#             "rtc_configuration": rtc_config_json
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
#
# def get_answer_from_remote_peer(sn, token, public_key, local_description, turn_server_info):
#     sdp_offer_json = {
#         "id": "",
#         "turnserver": turn_server_info,
#         "sdp": local_description.sdp,
#         "type": local_description.type,
#         "token": token
#     }
#
#     peer_answer_json = send_sdp_to_remote_peer(sn, json.dumps(sdp_offer_json), token, public_key)
#
#     return peer_answer_json