#DATA_PATH="gift4code.tfrecord"
#OUTPUT_PATH="results"
#MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-instruct"
#MODEL_PATH="deepseek-ai/deepseek-coder-33b-instruct"
#MODEL_PATH="bigcode/starcoder2-3b"
#MODEL_PATH="bigcode/starcoder2-15b"

#CUDA_VISIBLE_DEVICES=0 python finetune.py \

#Convert tf record to huggingface dataset first
python convert_tfrecord_to_hfdataset.py

deepspeed finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 5 \
    --model_max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True
    #--report_to "tensorboard" \
