#!/bin/bash
set -e  # Exit on any error

# === Variables ===
DEVICES=3
CONFIG_PATH="config.json"
HF_TOKEN="your_huggingface_token_here"

# === Conditional FSDP usage ===
if [ "$DEVICES" -eq 1 ]; then
  USE_FSDP="false"
else
  USE_FSDP="true"
fi

echo "🛠️ Devices: $DEVICES"
echo "🧠 FSDP Enabled: $USE_FSDP"
echo "📄 Config Path: $CONFIG_PATH"

# === Run steps ===

echo "🔹 Step 1: Training with LoRA on $DEVICES devices..."
fabric run \
  --node-rank=0 \
  --accelerator=cuda \
  --devices=$DEVICES \
  --num-nodes=1 \
  train_lora.py \
  --config_path "$CONFIG_PATH" \
  --use_fsdp $USE_FSDP \
  --hf_token $HF_TOKEN

echo "🔹 Step 2: Inference..."
python3 inference.py --config_path "$CONFIG_PATH" --hf_token $HF_TOKEN

echo "🔹 Step 3: Filtering..."
python3 filter_output.py --config_path "$CONFIG_PATH" \

echo "🔹 Step 4: Evaluation..."
python3 evaluation.py --config_path "$CONFIG_PATH" \

# echo "✅ All done!"
