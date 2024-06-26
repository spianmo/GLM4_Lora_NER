import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import prompt


def read_file_content(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def process_dataset():
    # 遍历text目录底下的所有文件
    file_path = "./val/text"
    dataset = []
    for file in os.listdir(file_path):
        dataset.append({
            "text": read_file_content(file_path + "/" + file),
            "file_name": file
        })
    return dataset


if __name__ == '__main__':
    mode_path = './ZhipuAI/glm-4-9b-chat'
    lora_path = './GLM4_lora'

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval()

    # 加载lora权重
    model = PeftModel.from_pretrained(model, model_id=lora_path)

    val_list = process_dataset()

    # 遍历val_list
    for val in val_list:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "user", "content": val["text"]}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to('cuda')

        gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            with open(f"./val/labels/{val['file_name']}", 'w', encoding='utf-8') as f:
                result = tokenizer.decode(outputs[0], skip_special_tokens=True) + '\n'
                print(result)
                f.write(result)
