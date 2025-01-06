import torch
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 获取当前文件的目录，并构造模型路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
model_path = os.path.join(project_root, "models/merged_model")

def generate_mask_output(input_data, model_path="./models/merged_model", **kwargs):
    print(f"####################################\nmodel_path: {model_path}\n####################################")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model.eval()

    user_input = input_data.get("input", "")
    if "<mask>" not in user_input:
        return "Please include <mask> in your input."

    inputs = tokenizer(user_input, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits

    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    if mask_token_index.size(0) == 0:
        return "No <mask> token found in the input."

    mask_token_logits = predictions[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    top_5_predictions = [tokenizer.decode([token]) for token in top_5_tokens]
    return top_5_predictions

if __name__ == "__main__":
    import json
    import sys

    input_data = json.loads(sys.argv[1])
    output = generate_mask_output(
        input_data,
        model_path=input_data.get("model", "./models/merged_model"),
        **input_data.get("params", {})
    )
    print(json.dumps({"output": output}))
