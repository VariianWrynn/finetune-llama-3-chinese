import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def download_model(model_name, save_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

def download_dataset(dataset_name, subset_name, save_dir):
    dataset = load_dataset(dataset_name, subset_name)
    dataset.save_to_disk(save_dir)

if __name__ == "__main__":
    model_names = {
        "distilroberta-base": "./local_model_distilroberta-base"
        #"EleutherAI/gpt-neo-1.3B": "./local_model_gpt-neo"
    }
    dataset_name = "wikitext"
    subset_name = "wikitext-2-raw-v1"
    dataset_save_dir = "./local_dataset"

    if not os.path.exists(dataset_save_dir):
        os.makedirs(dataset_save_dir)

    for model_name, model_save_dir in model_names.items():
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        print(f"Downloading model {model_name}...")
        download_model(model_name, model_save_dir)
        print(f"Model {model_name} downloaded.")

    print("Downloading dataset...")
    download_dataset(dataset_name, subset_name, dataset_save_dir)
    print("Dataset downloaded.")
