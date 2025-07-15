from torch.utils.data import default_collate
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from typing import Dict, List, Tuple, Type
from prepare_data.preprocess import *
import os
import json
import random
import numpy as np
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import lightning as L
from lightning.fabric import Fabric
from lightning.fabric.strategies import FSDPStrategy
from torch.optim import AdamW
from tqdm import tqdm
from jsonargparse import CLI
from torch import Tensor

from datasets import load_from_disk


####################################################

def setup(
    hf_token: str = "your_token",
    use_fsdp: bool = True,  # <-- NEW PARAMETER
    config_path: str = ""
):
    # Load the model config
    with open(config_path) as f:
        config = json.load(f)
    
    # === Seed ===
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    lr = config["lr"]
    epochs = config["epochs"]
    patience = config["lora_val_patience"]
    precision = config["precision"]
    target_modules = config["lora_train"]["target_modules"]
    joined_target_modules = ['__'.join(target_modules[:i]) for i in range(1, len(target_modules)+1)]

    output_dir = "models/"+config["model_name"]+"_"+str(epochs)+"_max_ep_"+str(patience)+"_pat_"+str(lr)+"_lr_"+joined_target_modules[-1]+"_"+str(seed)

    # === Choose strategy ===
    num_gpus = torch.cuda.device_count()
    if use_fsdp and num_gpus > 1:
        policy = {LlamaDecoderLayer}
        strategy = FSDPStrategy(
            auto_wrap_policy=policy,
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
        print("🔄 Using FSDP strategy")
    else:
        strategy = "auto"
        print("🔁 Using default strategy (single GPU or no FSDP)")

    # === Fabric ===
    fabric = Fabric(strategy=strategy, precision=precision)
    fabric.seed_everything(seed)
    devices = fabric.world_size
    fabric.print(f"Using {devices} device(s)")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"pad_token": "<|end_of_text|>"})
    
    train_dataset = load_from_disk(config["training_tokenized"])
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    validation_dataset = load_from_disk(config["eval_tokenized"])
    validation_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataloader = DataLoader(train_dataset, batch_size=config["micro_batch_size_train"], shuffle=True,
                                  collate_fn=get_sft_collate_fn(max_seq_length=config["max_seq_length"], pad_id=0, ignore_index=-100))
    val_dataloader = DataLoader(validation_dataset, batch_size=config["micro_batch_size_eval"], shuffle=False,
                                       collate_fn=get_sft_collate_fn(max_seq_length=config["max_seq_length"], pad_id=0, ignore_index=-100))


    train_loader, val_loader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    # === Model + LoRA + FSDP ===
    with fabric.init_module():
        model = AutoModelForCausalLM.from_pretrained(config["model_name"], token=hf_token, torch_dtype=torch.bfloat16)

        model.resize_token_embeddings(len(tokenizer))
        lora_config = LoraConfig(
            r=config["lora_train"]["lora_r"], 
            lora_alpha=config["lora_train"]["lora_alpha"],
            target_modules=config["lora_train"]["target_modules"],
            lora_dropout=config["lora_train"]["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)

    model = model.to(torch.bfloat16)
    optimizer = AdamW(model.parameters(), lr=lr)
    model, optimizer = fabric.setup(model, optimizer)

    # === Training loop ===
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        fabric.print(f"\u2705 Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # === Validation ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        fabric.print(f"\u274C Epoch {epoch+1} | Validation Loss: {avg_val_loss:.4f}")

        epoch_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir, safe_serialization=True)
        tokenizer.save_pretrained(epoch_dir)
        fabric.print(f"💾 Model saved to {epoch_dir}")


        # === Early stopping check ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                fabric.barrier()
                fabric.print(f"\nEarly stopping triggered after epoch {epoch+1} due to increased validation loss.")
                break

    # === Save model ===
    # if fabric.global_rank == 0:
    fabric.print(f"\nBest model at epoch {best_epoch} with validation loss {best_val_loss:.4f}")
    fabric.barrier()
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Save best epoch info to txt
    with open(os.path.join(output_dir, "best_epoch.txt"), "w") as f:
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_loss={best_val_loss:.6f}\n")

    fabric.print(f"Model saved to {output_dir}")
    

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    CLI(setup)