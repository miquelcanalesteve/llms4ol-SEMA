import os
import json
import random
import numpy as np
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, DatasetDict

# === Seed for reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Configuration ===
MODEL_NAME = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "lora_finetuned_models/Llama-3.2-1B-ct_train_v2_r6_domain"
TRAIN_PATH = "data/task_c/MatOnto_augmented/train_6_simple_question_subclass.json"
VAL_PATH = "data/task_c/MatOnto_augmented/val_6_simple_question_subclass.json"
MAX_LENGTH = 512
BATCH_SIZE = 1
EPOCHS = 6
LR = 2e-5
TOKEN = "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"
SYSTEM_PROMPT_PATH = "system_prompts.json"
LORA_TARGET_MODELS_PATH = "config_lora.json"

with open(LORA_TARGET_MODELS_PATH, "r", encoding="utf-8") as f:
    LORA_TARGET_MODELS = json.load(f)
LORA_TARGET = LORA_TARGET_MODELS["Llama"]["full"]

with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPTS = json.load(f)
SYSTEM_PROMPT = SYSTEM_PROMPTS["system_prompts"][0]["prompt"]

print("✅ Configuration loaded.")

# === Load Dataset ===
def load_and_format(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📄 Loaded {len(data)} examples from {path}")
    return [{"prompt": item["prompt"], "response": item["response"]} for item in data]

train_samples = load_and_format(TRAIN_PATH)
val_samples = load_and_format(VAL_PATH)

dataset_dict = DatasetDict({
    "train": Dataset.from_list(train_samples),
    "validation": Dataset.from_list(val_samples),
})

print("✅ DatasetDict created.")

# === Tokenizer ===
print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})
print(f"✅ Tokenizer loaded. Vocabulary size: {len(tokenizer)}")

# === Tokenization with proper label masking ===
def tokenize(example):
    bos = "<|begin_of_text|>"
    eot = "<|eot_id|>"
    start = "<|start_header_id|>"
    end = "<|end_header_id|>"

    # Define full prompt and assistant response
    prompt_only = (
        f"{bos}{start}system{end}\n\n{SYSTEM_PROMPT}{eot}\n"
        f"{start}user{end}\n\n{example['prompt']}{eot}\n"
        f"{start}assistant{end}\n\n"
    )

    full_chat = prompt_only + example["response"] + eot

    # Print the first few formatted prompts
    if "printed_template_count" not in globals():
        globals()["printed_template_count"] = 0
    if globals()["printed_template_count"] < 5:
        print(f"\n📝 Example {globals()['printed_template_count'] + 1} of chat-formatted prompt:\n")
        print(full_chat)
        globals()["printed_template_count"] += 1

    # Tokenize full chat
    full_tokens = tokenizer(full_chat, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    # Compute labels: -100 for prompt, token ids for assistant response
    prompt_len = len(tokenizer(prompt_only, add_special_tokens=False)["input_ids"])
    labels = [-100] * prompt_len

    response_ids = tokenizer(example["response"] + eot, add_special_tokens=False)["input_ids"]
    labels += response_ids

    # Pad or truncate labels
    if len(labels) < MAX_LENGTH:
        labels += [-100] * (MAX_LENGTH - len(labels))
    else:
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

print("🔄 Tokenizing datasets...")
tokenized_datasets = dataset_dict.map(tokenize, batched=False)
print("✅ Tokenization complete.")

# === Load base model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Loading model to device: {device}")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=TOKEN).to(device)
model.resize_token_embeddings(len(tokenizer))
print("✅ Model loaded and resized.")

# === LoRA configuration ===
print("🔧 Applying LoRA configuration...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=LORA_TARGET,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Trainer setup ===
print("⚙️ Setting up Trainer...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=True,
    seed=SEED,
    logging_steps=100,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)

# === Start training ===
print("🚀 Starting training...")
trainer.train()

# === Save final model and tokenizer ===
print("💾 Saving final model and tokenizer...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# === Export logs to XLSX ===
print("📊 Exporting training logs...")
log_data = []
for entry in trainer.state.log_history:
    if "loss" in entry or "eval_loss" in entry:
        log_data.append({
            "epoch": entry.get("epoch"),
            "train_loss": entry.get("loss"),
            "eval_loss": entry.get("eval_loss")
        })

df_log = pd.DataFrame(log_data)
xlsx_path = os.path.join(OUTPUT_DIR, "training_logs.xlsx")
df_log.to_excel(xlsx_path, index=False)

print("\n✅ Training complete. Logs saved to:", xlsx_path)
