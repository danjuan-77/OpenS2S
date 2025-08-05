#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python batch_inference.py --save-dir infer_results/opens2s_base/wav --start 1200 --end 1800

# nohup bash infer_gpu2.sh > ../UltraVoice_dev/logs/opens2s_base_infer_gpu2_$(date +%Y%m%d%H%M%S).log 2>&1 &
