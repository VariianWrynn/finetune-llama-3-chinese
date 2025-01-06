import sys
import os

# 添加父目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.inference.chat_inference import generate_chat_response
from scripts.inference.mask_inference import generate_mask_output

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
import torch

# 全局变量，加载模型和分词器
current_dir = os.path.dirname(os.path.abspath(__file__))

chat_model_path = os.path.join(current_dir, "../models/base_model/gpt2-medium")
chat_model_path = chat_model_path.replace("\\", "/")
print(f"Resolved chat model path: {chat_model_path}")

chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_path, use_fast=False)
chat_model = AutoModelForCausalLM.from_pretrained(chat_model_path)
chat_model.eval()

mask_model_path = os.path.join(current_dir, "../models/merged_model")
mask_model_path = mask_model_path.replace("\\", "/")
print(f"Resolved mask model path: {mask_model_path}")

mask_tokenizer = AutoTokenizer.from_pretrained(mask_model_path, use_fast=False)
mask_model = AutoModelForMaskedLM.from_pretrained(mask_model_path)
mask_model.eval()

def generate_chat_response(messages, model_name="gpt2-medium", max_tokens=100, temperature=0.8, top_k=50, top_p=0.95):
    # 构造对话历史
    conversation_history = "".join([f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in messages])
    
    # 转换为模型输入
    inputs = chat_tokenizer(conversation_history, return_tensors="pt")
    
    # 推理
    with torch.no_grad():
        outputs = chat_model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            do_sample=True, 
            top_k=top_k, 
            top_p=top_p, 
            temperature=temperature,
            repetition_penalty=1.5
        )
    
    # 解码输出
    response = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_mask_response(input_data, model_name="default-mask-model", **kwargs):
    if isinstance(input_data, str):
        input_data = {"input": input_data}
    
    user_input = input_data.get("input", "")
    if "<mask>" not in user_input:
        return "Please include <mask> in your input."

    inputs = mask_tokenizer(user_input, return_tensors="pt")

    with torch.no_grad():
        outputs = mask_model(**inputs)
        predictions = outputs.logits

    mask_token_index = torch.where(inputs["input_ids"] == mask_tokenizer.mask_token_id)[1]
    if mask_token_index.size(0) == 0:
        return "No <mask> token found in the input."

    mask_token_logits = predictions[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    top_5_predictions = [mask_tokenizer.decode([token]) for token in top_5_tokens]
    return top_5_predictions

def handle_chat(messages, model="default-model", max_tokens=200, temperature=0.7, top_p=1.0, top_k=50):
    return generate_chat_response(messages, model, max_tokens, temperature, top_k, top_p)

def handle_mask(input_data, model="default-mask-model", **kwargs):
    return generate_mask_response(input_data, model, **kwargs)
