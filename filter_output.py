import json
from pathlib import Path
import re

def extract_answer_from_generated_text(generated: str, debug=False) -> str:
    """
    Return 'true' or 'false' if the generated text contains an answer.
    """
    text = generated.lower()

    if debug:
        print(f"🔍 Lowercased generated text:\n{text[:300]}...\n")  # Print first 300 chars

    if "answer:\nassistant\n\ntrue" in text or "answer:assistant\n\ntrue" in text:
        return "true"
    elif "answer:\nassistant\n\nfalse" in text or "answer:assistant\n\nfalse" in text:
        return "false"
    return ""


def is_true_answer(entry):
    answer = extract_answer_from_generated_text(entry["generated_text"])
    return answer == "true"

def filter_true_answers(input_path: Path, output_path: Path):
    # Load input JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter entries based on the parsed answer
    filtered = {
        key: value for key, value in data.items()
        if is_true_answer(value)
    }

    # Save result
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved {len(filtered)} true entries to {output_path}")

if __name__ == "__main__":
    input_file = Path("task_c_llama_3_1b_ct_train_v2_r6_domain_2.json")
    output_file = Path("task_c_llama_3_1b_ct_train_v2_r6_domain_2_true.json")
    filter_true_answers(input_file, output_file)
