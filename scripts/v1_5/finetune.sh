#!/bin/bash
# pip install -e ./
# pip install -e ".[train]"
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
# pip install flash-attn --no-build-isolation
# /mnt/raid5/weizhi/llava/mlm_filter_instruct_50k_gpt4v_cc12m_4k_checked.json
# /mnt/raid5/weizhi/llava/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k_checked.json

deepspeed --num_gpus=8 llava/train/train_mem.py  \
    --model_name_or_path /mnt/raid5/weizhi/models/Qwen2.5-7B-Instruct \
    --deepspeed ./scripts/zero3.json \
    --version qwen_instruct \
    --data_path /mnt/raid5/weizhi/data/llava/final_data/llava_video_mix.json \
    --image_folder /mnt/raid5/weizhi/data/llava/images/ \
    --vision_tower google/siglip-so400m-patch14-384 \
    --pretrain_mm_mlp_adapter /mnt/raid5/weizhi/checkpoints/lavia-qwen-2.5-instruct-pretrain-siglip-g-384-aapool-64/mm_projector.bin \
    --mm_projector_type aapool_mlp \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_num_image_tokens 64 \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /mnt/raid5/weizhi/checkpoints/LLaVA-Video-Qwen-2.5-7B \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none