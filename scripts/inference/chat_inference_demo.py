import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./models/base_model/gpt2-medium"  # 更新后的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

conversation_history = ""

while True:
    user_input = input("User: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Bye!")
        break
    
    # 将用户输入添加到对话历史中
    conversation_history += f"User: {user_input}\n"
    
    # 将对话历史转成模型可理解的张量
    inputs = tokenizer(conversation_history, return_tensors="pt")
    
    # 在CPU上推理（如果有GPU，请改成 to("cuda") 并把模型放到 model.to("cuda")）
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95, 
            temperature=0.8,
            repetition_penalty=1.5
        )
    
    # 解码成文本
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model:", response)
    
    # 将模型的回复添加到对话历史中
    conversation_history += f"Model: {response}\n"
