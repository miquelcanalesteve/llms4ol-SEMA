import json
from pathlib import Path
from jsonargparse import CLI

def extract_answer_from_generated_text(generated: str, debug=False) -> str:
    """
    Return 'true' or 'false' if the generated text contains an answer.
    """
    text = generated.lower()
    if debug:
        print(f"🔍 Lowercased generated text:\n{text[:300]}...\n")
    if "answer:\nassistant\n\ntrue" in text or "answer:assistant\n\ntrue" in text:
        return "true"
    elif "answer:\nassistant\n\nfalse" in text or "answer:assistant\n\nfalse" in text:
        return "false"
    return ""

def is_true_answer(entry):
    answer = extract_answer_from_generated_text(entry.get("generated_text", ""))
    return answer == "true"


def get_best_epoch(txt_path: Path) -> int:
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("best_epoch="):
                return int(line.strip().split("=")[1])
    raise ValueError("best_epoch not found in file.")

def extract_true_relations(input_path: Path, output_path: Path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = []
    for item in data.values():
        if is_true_answer(item):
            result.append({
                "parent": item["parent"],
                "child": item["child"]
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted {len(result)} true relations to {output_path}")

def main(config_path: Path = "config.json"):
    # === Load config ===
    with open(config_path) as f:
        config = json.load(f)

    lr = config["lr"]
    epochs = config["epochs"]
    patience = config["lora_val_patience"]
    target_modules = config["lora_train"]["target_modules"]
    joined_target_modules = ['__'.join(target_modules[:i]) for i in range(1, len(target_modules) + 1)]

    model_name = config["model_name"]
    best_epoch_path = f"models/{model_name}_{epochs}_max_ep_{patience}_pat_{lr}_lr_{joined_target_modules[-1]}/best_epoch.txt"
    best_epoch = get_best_epoch(best_epoch_path)

    peft_model_path = f"models/{model_name}_{epochs}_max_ep_{patience}_pat_{lr}_lr_{joined_target_modules[-1]}/epoch_{best_epoch}"
    input_file = "../data/predictions/" + peft_model_path.replace("/", "_") + ".json"
    output_file = input_file[:-5] + "_true.json"

    extract_true_relations(Path(input_file), Path(output_file))

if __name__ == "__main__":
    CLI(main)
