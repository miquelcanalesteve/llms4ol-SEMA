import json
import random
from pathlib import Path
from typing import Optional

# === Configurable Function ===
def generate_dataset(
    input_file: str,
    output_dir: str,
    combined_file: str,
    false_to_true_ratio: int,
    template_type: str,
    descriptions_file: Optional[str] = None
):
    # === Setup ===
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    descriptions = {}
    if descriptions_file:
        with open(descriptions_file, "r", encoding="utf-8") as f:
            descriptions = json.load(f)

    true_pairs = {(d["parent"], d["child"]) for d in data}
    types = list({item["parent"] for item in data} | {item["child"] for item in data})

    # === Template Selection ===
    def format_prompt(parent, child):
        if template_type == "owl":
            parent_class = ''.join(word.capitalize() for word in parent.split())
            child_class = ''.join(word.capitalize() for word in child.split())
            return f"""matonto:{parent_class} a owl:Class;
dcterms:title '{parent}'.

matonto:{child_class} a owl:Class;
dcterms:title '{child}';
rdfs:subClassOf matonto:{parent_class}.

Answer:"""

        elif template_type == "owl_wiki":
            parent_class = ''.join(word.capitalize() for word in parent.split())
            child_class = ''.join(word.capitalize() for word in child.split())
            parent_desc = clean_description(descriptions.get(parent, ""))
            child_desc = clean_description(descriptions.get(child, ""))
            return f"""matonto:{parent_class} a owl:Class;
dcterms:title '{parent}';
dcterms:description '{parent_desc}'.

matonto:{child_class} a owl:Class;
dcterms:title '{child}';
dcterms:description '{child_desc}';
rdfs:subClassOf matonto:{parent_class}.

Answer:"""

        elif template_type == "context":
            parent_desc = clean_description(descriptions.get(parent, ""))
            child_desc = clean_description(descriptions.get(child, ""))
            return f"""Context:
- "{child}": {child_desc}
- "{parent}": {parent_desc}

Question:
Is "{child}" a subclass of "{parent}"? Answer with "true" or "false". Answer:"""

        elif template_type == "simple_question_subclass":
            return f'Is "{child}" a subclass of "{parent}"? Answer with "true" or "false". Answer:'
        
        elif template_type == "simple_question_child_class":
            return f'Is "{child}" a child class of "{parent}"? Answer with "true" or "false". Answer:'

        elif template_type == "simple_question_parent":
            return f'Is "{parent}" the parent class of "{child}"? Answer with "true" or "false". Answer:'

        else:
            raise ValueError("Unknown template type")

    def clean_description(desc):
        if not desc or "may refer to:" in desc:
            return ""
        return desc.replace("'", "’")

    # === 1. Generate TRUE examples ===
    true_examples = [
        {"prompt": format_prompt(p, c), "response": "true"}
        for p, c in true_pairs
    ]

    # === 2. Generate FALSE examples ===
    false_pairs = set()
    needed = len(true_examples) * false_to_true_ratio
    attempts = 0
    max_attempts = needed * 20

    while len(false_pairs) < needed and attempts < max_attempts:
        p, c = random.sample(types, 2)
        if (p, c) not in true_pairs and (p, c) not in false_pairs:
            false_pairs.add((p, c))
        attempts += 1

    false_examples = [
        {"prompt": format_prompt(p, c), "response": "false"}
        for p, c in false_pairs
    ]

    # === 3. Combine ===
    combined = []
    false_pool = false_examples.copy()
    random.shuffle(false_pool)
    false_index = 0

    for true_example in true_examples:
        combined.append(true_example)
        combined.extend(false_pool[false_index:false_index + false_to_true_ratio])
        false_index += false_to_true_ratio

    random.shuffle(combined)

    # === 4. Save ===
    with open(output_path / combined_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"✅ Generated {len(true_examples)} true, {len(false_examples)} false examples.")
    print(f"📁 Combined file saved to: {output_path / combined_file}")

# === Example Usage ===
generate_dataset(
    input_file="data/task_c/MatOnto/matonto_val_pairs.json", #change the name in validation or train
    output_dir="data/task_c/MatOnto_augmented",
    combined_file="val_6_simple_question_subclass.json", #change the name in validation or train
    false_to_true_ratio=6,
    template_type="simple_question_subclass",  # options: owl, owl_wiki, context, simple_question_subclass, simple_question_child_class, simple_question_parent
    descriptions_file="data/task_c/MatOnto/concepts_wikipedia.json"
)
