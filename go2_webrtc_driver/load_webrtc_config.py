import json
import os

def load_webrtc_config(config_path="webrtc_config.json"):
    """
    webrtc_config.json 파일을 읽어서 딕셔너리로 반환합니다.

    Args:
        config_path (str): 설정 파일 경로 (기본값: 현재 디렉토리의 webrtc_config.json)

    Returns:
        dict: { "email": ..., "password": ..., "sn": ... }
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 유효성 검증
    required_keys = {"email", "password", "sn"}
    if not required_keys.issubset(config.keys()):
        raise ValueError(f"다음 필드가 필요합니다: {required_keys}")

    return config
