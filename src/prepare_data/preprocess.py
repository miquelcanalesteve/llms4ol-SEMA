import os
import pandas as pd
import datasets
import torch
import json
from prepare_data.prompts import PromptStyle
from functools import partial
from typing import Dict, List
from torch import Tensor


def tokenizer_dataset(dir, tokenizer, config, max_seq_length=2048):
    dir = os.path.join(os.getcwd(), dir)
    
    dataset = pd.DataFrame(columns=['system', 'instruction', 'output'])

    for file in os.listdir(dir):
        print(file)
        with open(os.path.join(dir, file)) as f:
            data = json.load(f)

        systemPrompt = data['system']
        auxData = pd.DataFrame(data['prompts'])
        auxData['system'] = systemPrompt
        auxData.rename(columns={'user': 'instruction', 'answer': 'output'}, inplace=True)
        dataset = pd.concat([dataset, auxData], axis=0)

    dataset = datasets.Dataset.from_pandas(dataset)

    dataset = dataset.map(
        tokenizer_dataset_with_prompt,
        remove_columns=['instruction', 'output', 'system'],
        fn_kwargs={'tokenizer': tokenizer, 'config': config, 'max_seq_length': max_seq_length}
    )
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def tokenizer_dataset_with_prompt(sample, tokenizer, config, max_seq_length):
    prompt_style = PromptStyle.from_name(config["prompt_style"])
    prompt = prompt_style.apply(prompt=sample, **sample)
    prompt_and_response = prompt + sample["output"]
    print("prompt_and_response:", prompt_and_response)
    encoded_prompt = tokenizer.encode(prompt, max_length=max_seq_length)
    encoded_prompt_and_response = tokenizer.encode(prompt_and_response, max_length=max_seq_length-1, return_tensors="pt")
    encoded_prompt_and_response = torch.cat((encoded_prompt_and_response.squeeze_(), torch.tensor([tokenizer.eos_token_id])))

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_prompt_and_response.clone()
    attention_mask = encoded_prompt_and_response.clone()

    if config["mask_prompt"]:
        labels[: len(encoded_prompt)] = config["ignore_index"]
    attention_mask[:(len(encoded_prompt_and_response))] = 1
    return {"input_ids": encoded_prompt_and_response.type(torch.int64), "labels": labels.type(torch.int64), "attention_mask": attention_mask.type(torch.int64)}

def get_sft_collate_fn(max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100):
    """Returns the collate function for supervised finetuning (needed in the DataLoader).

    The collate function gets a list of dicts with keys `input_ids` and `labels`.
    It returns a dict with batched `input_ids` and `labels`. Also pads short sequences to the longest element in
    the batch. Optionally truncates all sequences to the specified maximum length.
    """
    return partial(_sft_collate_fn, max_seq_length=max_seq_length, pad_id=pad_id, ignore_index=ignore_index)

def _sft_collate_fn(
    samples: List[Dict[str, Tensor]], max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100
) -> Dict[str, Tensor]:

    batched = {}
    for key in ("input_ids", "labels", "attention_mask"):
        pad_value = pad_id if key == "input_ids" else ignore_index

        # Pad right based on the longest sequence
        batched[key] = torch.nn.utils.rnn.pad_sequence(
            [sample[key] for sample in samples], batch_first=True, padding_value=pad_value
        )

        # Truncate if needed
        if max_seq_length > 0:
            batched[key] = batched[key][:, :max_seq_length]

    return batched