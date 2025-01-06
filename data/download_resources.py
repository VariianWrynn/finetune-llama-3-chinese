import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def download_model(model_name, model_save_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, force_download=True)
    tokenizer.save_pretrained(model_save_dir)
    model.save_pretrained(model_save_dir)
    print("model_save_dir:", model_save_dir)
    print("Files in model_save_dir:", os.listdir(model_save_dir))

def download_dataset(dataset_name, subset_name, save_dir):
    dataset = load_dataset(dataset_name, subset_name)
    dataset.save_to_disk(save_dir)

if __name__ == "__main__":
    model_names = {
        "distilroberta-base": "./models/base_model/distilroberta-base",
        #"mymusise/gpt2-small-chinese": "./models/base_model/gpt2-small-chinese"
        "openai-community/gpt2-medium": "./models/base_model/gpt2-medium"
        #"EleutherAI/gpt-neo-1.3B": "../models/base_model/gpt-neo"
    }
    dataset_name = "wikitext"
    subset_name = "wikitext-2-raw-v1"
    dataset_save_dir = "./data/local_dataset/wikitext"

    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)

    for model_name, model_save_dir in model_names.items():
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        #if model data is downloaded, skip downloading
        if os.listdir(model_save_dir):
            continue
        print(f"Downloading model {model_name}...")
        download_model(model_name, model_save_dir)
        print(f"Model {model_name} downloaded.")

    print("Downloading dataset...")
    download_dataset(dataset_name, subset_name, dataset_save_dir)
    print("Dataset downloaded.")
