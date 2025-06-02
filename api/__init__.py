# api/__init__.py
from flask import Blueprint
from api.webrtc_api import webrtc_api  # ← 변경된 이름으로 import

def register_blueprints(app):
    app.register_blueprint(webrtc_api)
