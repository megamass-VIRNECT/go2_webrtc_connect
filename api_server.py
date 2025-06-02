# api_server.py
from flask import Flask
from flask_cors import CORS              # 추가
from api import register_blueprints

app = Flask(__name__)
CORS(app)                                 # CORS 활성화
register_blueprints(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
