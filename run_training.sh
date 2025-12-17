#!/bin/bash
# Install CLIP package if not already installed
if ! python -c "import clip" &> /dev/null; then
    echo "Installing CLIP..."
    pip install git+https://github.com/openai/CLIP.git
fi

# Install other dependencies (optional, remove if unnecessary)
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    #pip install --upgrade -r requirements.txt

else
    echo "requirements.txt not found. Skipping dependency installation."
fi

echo "Starting training..."
python src/main.py \
    --dataset "harmeme" \
    --num_mapping_layers 2 \
    --map_dim 1024\
    --fusion "align" \
    --num_pre_output_layers 3 \
    --drop_probs 0.2 0.4 0.1 \
    --gpus "0" \
    --batch_size 64 \
    --lr 0.000013 \
    --max_epochs 25\
    --name "text-inv-comb" \
    --pretrained_proj_weights True\
    --freeze_proj_layers True\
    --proj_map True \
    --use_cross_attn True \
    --num_heads 8\
    --phi_inv_proj True \
    --text_inv_proj True \
    --post_inv_proj True \
    --enh_text True \
    --phi_freeze True\
    --fast_process True \
    --print_model False \
    --instruction_tuning True \
    --lora_rank 8 \
    --comb_skip True\
    --contrastive True \
    --contrastive_temp 0.07 \
    --label_smoothing 0.15 \
    --gradient_clip_val 0.5\
    --weight_decay 0.005 \
    --lr_scheduler True \
    --lr_patience 3 \
    --lr_factor 0.5

