import os
import sys

# 确保工作目录为项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)
sys.path.append(project_root)

from flask import Flask
from api.chat import chat_bp
from api.mask import mask_bp

app = Flask(__name__)

# 注册蓝图
app.register_blueprint(chat_bp, url_prefix='/v1/chat')
app.register_blueprint(mask_bp, url_prefix='/v1/mask')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)