#!/bin/bash

export omnispeech_path=/share/nlp/tuwenming/models/CASIA-LM/OpenS2S

python src/instruction_dataset.py offline \
    --dataroot /share/nlp/tuwenming/projects/GLM-4-Voice/opens2s_ultravoice \
    --manifest_files "*.jsonl" \
    --llm_path ${omnispeech_path} \
    --tts_path ${omnispeech_path}/tts/ \
    --save_dir ./opens2s_ultravoice_training_data \
    --num_proc 64