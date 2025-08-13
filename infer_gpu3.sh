#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python batch_inference.py --save-dir infer_results/opens2s_sft_bs1_acc4_lr2e-5_ep1_6691/wav --model-path /mnt/buffer/tuwenming/checkpoints/OpenS2S/OpenS2S_SFT_20250811_213511_bs1_acc4_lr2e-5_ep1/checkpoint-6691

# nohup bash infer_gpu3.sh > ../UltraVoice_dev/logs/opens2s_base_infer_gpu3_$(date +%Y%m%d%H%M%S).log 2>&1 &
