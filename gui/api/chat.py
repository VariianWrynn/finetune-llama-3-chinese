from flask import Blueprint, request, jsonify
from gui.llm_inference import handle_chat
import time

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/completions', methods=['POST'])
def chat_completions():
    data = request.get_json()
    model = data.get("model", "default-model")
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 200)
    temperature = data.get("temperature", 0.7)
    top_p = data.get("top_p", 1.0)
    top_k = data.get("top_k", 50)
    # 其他参数

    # 调用推理逻辑
    answer = handle_chat(messages, model, max_tokens, temperature, top_p, top_k)

    # 添加超参数信息到回复中
    answer += f"\n\n[Model: {model}, Max Tokens: {max_tokens}, Temperature: {temperature}, Top P: {top_p}, Top K: {top_k}]"

    response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": answer
            },
            "finish_reason": "stop",
            "index": 0
        }]
    }
    return jsonify(response)
