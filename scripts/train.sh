# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=2 \
  --master_addr=localhost \
  --master_port=29503 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --vae_path flux/vae/ae.safetensors \
  --vit_path Huggingface/siglip-so400m-14-980-flash-attn2-navit \
  --llm_path Qwen/Qwen2.5-0.5B-Instruct \
  --use_flex True \
  --resume_from models/BAGEL-7B-MoT \
  --results_dir results \
  --checkpoint_dir results/checkpoints \
  --max_latent_size 64 \
  --num_workers 1