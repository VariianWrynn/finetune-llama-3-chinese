import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# 设置代理
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

def main():
    # 1. 准备基础模型和 tokenizer
    base_model_name = "distilroberta-base"  # 或 "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2  # SST-2 是二分类
    )

    # 2. 准备 LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.SEQUENCE_CLASSIFICATION,
        r=8,             # LoRA rank
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    print("LoRA parameters added to the model.")

    # 3. 准备数据集 (只取很小部分做演示)
    dataset = load_dataset("glue", "sst2")
    # 只取前 200 条做微调（演示用）
    small_train = dataset["train"].select(range(200))
    small_val = dataset["validation"].select(range(50))

    # 4. 数据预处理函数
    def preprocess(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

    small_train = small_train.map(preprocess, batched=True)
    small_val = small_val.map(preprocess, batched=True)

    # 把文本字段转换成模型所需的张量
    small_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    small_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 5. 训练参数设置（CPU 上就很慢，max_steps 或 epoch 就设置得很小）
    training_args = TrainingArguments(
        output_dir="./lora_test_out",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,  # CPU 能力有限，batch 不可太大
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="no",  # 演示用，不频繁保存
        # 设置 CPU 训练
        no_cuda=True,
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
    # 注意：只会保存 LoRA 增量权重，加上 base model 路径才是完整模型
    model.save_pretrained("lora_distilroberta_sst2")
    print("LoRA model saved.")

    # 如果你想合并 LoRA 权重到基模型中再保存，也可以参考 PEFT 文档


if __name__ == "__main__":
    main()
