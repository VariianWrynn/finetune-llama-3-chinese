from flask import Blueprint, request, jsonify
from gui.llm_inference import handle_mask
import time

mask_bp = Blueprint('mask', __name__)

@mask_bp.route('/inference', methods=['POST'])
def mask_inference():
    data = request.get_json()
    model = data.get("model", "default-mask-model")
    input_data = data.get("input", {})
    params = data.get("params", {})
    # 其他参数

    # 调用推理逻辑
    output = handle_mask(input_data, model, **params)

    response = {
        "id": f"maskinfr-{int(time.time())}",
        "object": "mask.inference",
        "created": int(time.time()),
        "model": model,
        "output": output
    }
    return jsonify(response)
