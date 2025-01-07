import os
import sys 

# 添加脚本所在目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(current_dir, ".."))
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