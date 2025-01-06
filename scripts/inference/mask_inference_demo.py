import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_path = "./models/merged_model"  # 模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForMaskedLM.from_pretrained(model_path)
model.eval()

while True:
    user_input = input("User: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Bye!")
        break
    
    # 确保用户输入包含 [MASK] 标记
    if "<mask>" not in user_input:
        print("Please include <mask> in your input.")
        continue
    
    # 将用户输入转成模型可理解的张量
    inputs = tokenizer(user_input, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
    
    # 获取掩码位置的预测结果
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    if mask_token_index.size(0) == 0:
        print("No <mask> token found in the input.")
        continue
    
    mask_token_logits = predictions[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
    # 打印预测结果
    print("Top 5 predictions for <mask>:")
    for token in top_5_tokens:
        print(tokenizer.decode([token]))
