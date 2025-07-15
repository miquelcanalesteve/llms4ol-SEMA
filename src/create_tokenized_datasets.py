import json
from pathlib import Path
from transformers import AutoTokenizer
from jsonargparse import CLI
from prepare_data.preprocess import tokenizer_dataset


def main(config_path: Path = Path("config.json")):
    # === Load the model config ===
    with open(config_path) as f:
        config = json.load(f)

    # === Load the tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # === Tokenize the datasets ===
    train_dataset = tokenizer_dataset(
        config["training_data"], tokenizer, config, config["max_seq_length"]
    )
    val_dataset = tokenizer_dataset(
        config["eval_data"], tokenizer, config, config["max_seq_length"]
    )

    # === Save tokenized datasets ===
    train_dataset.save_to_disk(config["training_tokenized"])
    val_dataset.save_to_disk(config["eval_tokenized"])
    print("✅ Tokenized datasets saved.")


if __name__ == "__main__":
    CLI(main)
