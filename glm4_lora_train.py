import os

os.environ["WANDB_API_KEY"] = '+++++++++++'  # 将引号内的+替换成自己在wandb上的一串值
os.environ["WANDB_MODE"] = "offline"  # 离线  （此行代码不用修改）

import json

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import DataCollatorForSeq2Seq, Trainer

from prompt import prompt


def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer((f"[gMASK]<sop><|system|>\n{prompt}<|user|>\n"
                             f"{example['input']}<|assistant|>\n"
                             ).strip(),
                            add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def read_file_content(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def process_dataset(output_file: str):
    # 遍历text目录底下的所有文件
    file_path = "./train/text"
    dataset = []
    for file in os.listdir(file_path):
        dataset.append({
            "input": read_file_content(file_path + "/" + file),
            "output": read_file_content("./train/labels" + "/" + file)
        })
    # 保存数据集
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(json.dumps(dataset, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    output_file = "./dataset_train.json"
    glm4_model_path = './ZhipuAI/glm-4-9b-chat'
    lora_path = './GLM4_lora'

    process_dataset(output_file)
    df = pd.read_json(output_file)
    ds = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(glm4_model_path, use_fast=False,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(glm4_model_path,
                                                 device_map="cuda:0", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model.enable_input_require_grads()  # 开启梯度检查点

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  # 现存问题只微调部分演示即可
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1  # Dropout 比例
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 配置训练参数
    args = TrainingArguments(
        output_dir="./output/GLM4",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=50,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=1e-5,
        save_on_each_node=True,
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    peft_model_id = lora_path
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
