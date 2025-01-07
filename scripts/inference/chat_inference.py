import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 获取当前文件的目录，并构造模型路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
model_path = os.path.join(project_root, "models/base_model/gpt2-medium")

def generate_chat_response(messages, model_path="./models/base_model/gpt2-medium", max_tokens=200, temperature=0.7, top_p=1.0, top_k=0):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    # 只使用最新的一条消息生成回复
    latest_message = messages[-1]
    conversation_history = f"{latest_message['role']}: {latest_message['content']}"
    inputs = tokenizer(conversation_history, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=True, 
            top_k=top_k, 
            top_p=top_p, 
            temperature=temperature,
            repetition_penalty=1.5
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    import json
    import sys

    input_data = json.loads(sys.argv[1])
    response = generate_chat_response(
        messages=input_data.get("messages", []),
        model_path=input_data.get("model", "./models/base_model/gpt2-medium"),
        max_tokens=input_data.get("max_tokens", 200),
        temperature=input_data.get("temperature", 0.7),
        top_p=input_data.get("top_p", 1.0)
    )
    print(json.dumps({"response": response}))
