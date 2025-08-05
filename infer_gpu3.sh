#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
python batch_inference.py --save-dir infer_results/opens2s_base/wav --start 1800 --end 2400

# nohup bash infer_gpu3.sh > ../UltraVoice_dev/logs/opens2s_base_infer_gpu3_$(date +%Y%m%d%H%M%S).log 2>&1 &
