omnispeech_path=/share/nlp/tuwenming/models/CASIA-LM/OpenS2S
data_dir=/share/nlp/tuwenming/projects/OpenS2S/opens2s_ultravoice_training_data
SAVE_ROOT=./ckpts

# 生成带时间戳和超参数的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_SIZE=1
GRAD_ACCUM=8
LEARNING_RATE=2e-5
EPOCHS=3
LOG_FILE="./logs/train_continue_${TIMESTAMP}_bs${BATCH_SIZE}_acc${GRAD_ACCUM}_lr${LEARNING_RATE}_ep${EPOCHS}.log"

# 确保logs目录存在
mkdir -p ./logs

python -m torch.distributed.run --nproc_per_node=4 train.py \
    --deepspeed ds_config/dp_config_zero1.json \
    \
    --dataset_dirs "${data_dir}" \
    \
    --output_dir ${SAVE_ROOT} \
    --remove_unused_columns False \
    --seed 42 \
    --do_train True \
    --bf16 True \
    \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.05 \
    --max_grad_norm 1.0 \
    --warmup_steps 500 \
    \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_epochs $EPOCHS \
    \
    --omnispeech_model $omnispeech_path \
    --unfreeze_adapter True \
    --unfreeze_llm True \
    --unfreeze_tts True \
    \
    --disable_tqdm True \
    --report_to "none" \
    \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 20

# nohup bash scripts/train_continue.sh > ${LOG_FILE} 2>&1 &
