#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python batch_inference.py --save-dir infer_results/opens2s_base/wav --start 0 --end 600

# nohup bash infer_gpu0.sh > ../UltraVoice_dev/logs/opens2s_base_infer_gpu0_$(date +%Y%m%d%H%M%S).log 2>&1 &
