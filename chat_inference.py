import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./merged_model_out"  # 或者你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

while True:
    user_input = input("User: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Bye!")
        break
    
    # 将用户输入转成模型可理解的张量
    inputs = tokenizer(user_input, return_tensors="pt")
    
    # 在CPU上推理（如果有GPU，请改成 to("cuda") 并把模型放到 model.to("cuda")）
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95, 
            temperature=0.8,
            repetition_penalty = 1.5
        )
    
    # 解码成文本
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Model:", response)
