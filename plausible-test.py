import os
import torch
import argparse

# 禁用 TensorFlow 警告
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 确保输出目录存在
output_dir = "./lora_test_out"
# 如果已有与 output_dir 同名的文件，先删除
if os.path.isfile(output_dir):
    os.remove(output_dir)
# 若文件夹尚不存在则创建
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

def parse_args():
    parser = argparse.ArgumentParser()
    #EleutherAI/gpt-neo-1.3B
    #distilroberta-base
    parser.add_argument("--base_model_name", default="distilroberta-base")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--train_samples", type=int, default=200)
    parser.add_argument("--val_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    return parser.parse_args()

def load_and_preprocess_dataset(tokenizer, train_samples, val_samples):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    small_train = dataset["train"].select(range(train_samples))
    small_val = dataset["validation"].select(range(val_samples))
    def preprocess(examples):
        tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    small_train = small_train.map(preprocess, batched=True)
    small_val = small_val.map(preprocess, batched=True)
    small_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    small_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return small_train, small_val

def main():
    args = parse_args()
    # 1. 准备基础模型和 tokenizer
    base_model_name = args.base_model_name
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(args.base_model_name)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    model = get_peft_model(model, peft_config)
    print("LoRA parameters added to the model.")

    # 加载和预处理数据集
    small_train, small_val = load_and_preprocess_dataset(tokenizer, args.train_samples, args.val_samples)

    # 5. 训练参数设置（CPU 上就很慢，max_steps 或 epoch 就设置得很小）
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir + "/logging",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,  # CPU 能力有限，batch 不可太大
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_strategy="no",  # 演示用，不频繁保存
        # 设置 CPU 训练
        use_cpu=True,
    )

    # 6. 使用 Trainer 进行微调
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train,
        eval_dataset=small_val,
    )

    print("Starting training...")
    trainer.train()
    print("Training done.")

    # 7. 评估
    results = trainer.evaluate()
    print("Evaluation results:", results)

    # 8. 保存模型（LoRA 权重）
    model.save_pretrained("lora_weights")
    print("LoRA model saved.")

    # 合并 LoRA 权重到基模型中
    merged_model = model.merge_and_unload()
    
    # 创建新文件夹
    merged_output_dir = "./merged_model_out"
    if not os.path.isdir(merged_output_dir):
        os.makedirs(merged_output_dir)
    
    # 保存合并后的模型
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    print("Merged model saved.")

if __name__ == "__main__":
    main()
