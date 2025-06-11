import json
from pathlib import Path
from itertools import combinations
from typing import List
import torch
import torch._dynamo
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

import os
os.environ["TORCH_COMPILE"] = "disable"
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # disables dynamo more forcefully
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCH_LOGS"] = "+dynamo"


# === Configuration ===
PEFT_MODEL_PATH = "./lora_finetuned_models/Llama-3.2-1B-ct_train_v2_r6_parent/checkpoint-6720"
TYPES_FILE = "data/task_c/MatOnto/test_types.txt"
OUTPUT_FILE = "data/outputs/llama_3_1b_ct_train_v2_r6_parent.json"
MAX_NEW_TOKENS = 10
MAX_INPUT_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
TOKEN = "your_token"

# === Load list of types ===
def load_list(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# === Format prompt with LLaMA 3.2 chat template ===
def format_chat_prompt(type_1: str, type_2: str) -> str:
    bos = "<|begin_of_text|>"
    eot = "<|eot_id|>"
    start = "<|start_header_id|>"
    end = "<|end_header_id|>"

    return (
        f"{bos}{start}system{end}\n\n"
        "You are an ontology expert specializing in material science. Your task is to assess whether the first concept is a subclass of the second—meaning it represents a more specific type within the same category. Rely on general and scientific knowledge. Respond strictly with 'true' or 'false'. Do not explain your answer."
        f"{eot}\n"
        f"{start}user{end}\n\n"
        f'Is "{type_1}" the parent class of "{type_2}"? Answer with "true" or "false". Answer:{eot}\n'
        f"{start}assistant{end}\n\n"
    )

# === Load LoRA model + tokenizer ===
def load_lora_model_and_tokenizer(peft_path: str):
    config = PeftConfig.from_pretrained(peft_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=TOKEN
    )
    model = PeftModel.from_pretrained(base_model, peft_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, token=TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model

# === Inference ===
@torch._dynamo.disable
def generate_relation(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=MAX_INPUT_LENGTH).input_ids.to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded

# === Main process ===
def main():
    types = load_list(TYPES_FILE)

    try:
        with open(OUTPUT_FILE, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    tokenizer, model = load_lora_model_and_tokenizer(PEFT_MODEL_PATH)

    for parent, child in combinations(types, 2):
        pair_id = f"{parent}__{child}"
        if pair_id not in results:
            prompt = format_chat_prompt(parent, child)
            print(f"\n🔍 Generating for: {parent} ⊆ {child}")
            print(f"📝 Prompt:\n{prompt}")
            generated_text = generate_relation(model, tokenizer, prompt)

            results[pair_id] = {
                "parent": parent,
                "child": child,
                "generated_text": generated_text,
                "prompt": prompt
            }

            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
