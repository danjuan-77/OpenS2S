omnispeech_path=/share/nlp/tuwenming/models/CASIA-LM/OpenS2S
data_dir=/share/nlp/tuwenming/projects/OpenS2S/opens2s_ultravoice_training_data
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_SIZE=1
GRAD_ACCUM=4
LEARNING_RATE=2e-5
EPOCHS=1
WARMUP_STEPS=500
SAVE_ROOT=/mnt/buffer/tuwenming/checkpoints/OpenS2S/OpenS2S_SFT_${TIMESTAMP}_bs${BATCH_SIZE}_acc${GRAD_ACCUM}_lr${LEARNING_RATE}_ep${EPOCHS}

python -m torch.distributed.run --nproc_per_node=4 train.py \
    --deepspeed ds_config/dp_config_zero2.json \
    \
    --dataset_dirs "${data_dir}" \
    \
    --output_dir ${SAVE_ROOT} \
    --remove_unused_columns False \
    --seed 42 \
    --do_train True \
    --bf16 True \
    \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps ${WARMUP_STEPS} \
    \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --num_train_epochs ${EPOCHS} \
    \
    --omnispeech_model $omnispeech_path \
    --unfreeze_adapter True \
    --unfreeze_llm True \
    --unfreeze_tts True \
    \
    --disable_tqdm True \
    --report_to "none" \
    \
    --logging_steps 1 \
    --save_steps 5000 \
    --save_total_limit 1

# nohup bash scripts/train_continue.sh > ./logs/train_continue_$(date +%Y%m%d%H%M%S).log 2>&1 &