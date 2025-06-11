# llms4ol-SEMA

This repository contains scripts and data for experimenting with Large Language Models (LLMs) on ontology learning tasks. The code focuses on training lightweight LoRA adapters for the Llama 3 model and evaluating them on the **MatOnto** dataset of parent--child pairs.

## Repository Structure

- `data/` – Dataset files and inference outputs.
  - `task_c/MatOnto/` – Original train/val/test pairs.
  - `task_c/MatOnto_augmented/` – Augmented training data generated with `data_augmentation.py`.
- `lora_finetuned_models/` – Saved LoRA checkpoints and logs.
- `fine_tunning_lora.py` – Script used to fine‑tune a LoRA adapter on the formatted dataset.
- `inference.py` – Loads a trained adapter and predicts subclass relations for pairs of concepts.
- `evaluation.py` – Computes precision, recall and F1 against ground truth pairs.
- `data_augmentation.py` – Utility to create synthetic training data with different prompt templates.
- `add_descriptions_wiki.py` – Fetches short concept descriptions from Wikipedia.
- `filter_output.py` – Filters inference results that contain a clear `true` answer.
- `config_lora.json` – Lists which model layers to target when applying LoRA.
- `system_prompts.json` – Predefined system prompts used when formatting chat style data.

## Installation

1. Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Obtain a Hugging Face token with access to the Llama model and set it in the scripts where `TOKEN` is required.

## Training a LoRA Adapter

Edit the paths and hyper‑parameters at the top of `fine_tunning_lora.py` to point to the desired dataset files and output directory. Then run:

```bash
python fine_tunning_lora.py
```

The script tokenizes the prompts, applies the LoRA configuration from `config_lora.json`, and trains using the Hugging Face `Trainer`. Checkpoints and logs are saved under `lora_finetuned_models/`.

## Running Inference

Update `inference.py` with the path to a LoRA checkpoint (`PEFT_MODEL_PATH`) and a file containing the list of concept types. Then run:

```bash
python inference.py
```

The generated predictions are stored as JSON in `data/outputs/`.

Before running the evaluation step, you must filter these raw predictions to
extract only the pairs where the model explicitly answered `true`. Adjust the
paths inside `filter_output.py` to point to your inference output and run:

```bash
python filter_output.py
```

This will generate a new `*_true.json` file which contains the cases used for
scoring.

## Evaluation

After running inference you can compute precision, recall and F1 score using:

```bash
python evaluation.py
```

This script compares your predictions with the ground‑truth pairs in `data/task_c/MatOnto/matonto_test_pairs.json` and writes a summary Excel file.

## Data Augmentation

`data_augmentation.py` can generate negative examples and format prompts in several templates (OWL statements, simple questions, etc.). Adjust the parameters in the `generate_dataset` call at the bottom of the script to create new augmented sets.

`add_descriptions_wiki.py` helps enrich concepts with short Wikipedia summaries that can be injected into the prompts.

## Notes

`fine_tunning_lora.py` and `inference.py` are intentionally independent so you can train adapters in one folder and load them from another. When fine‑tuning, specify which model layers to adapt (`full` or `minimal`) using the options in `config_lora.json`.

