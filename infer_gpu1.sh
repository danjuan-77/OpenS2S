#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python batch_inference.py --save-dir infer_results/opens2s_base/wav --start 600 --end 1200

# nohup bash infer_gpu1.sh > ../UltraVoice_dev/logs/opens2s_base_infer_gpu1_$(date +%Y%m%d%H%M%S).log 2>&1 &
