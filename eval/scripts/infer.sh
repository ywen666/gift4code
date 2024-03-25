MODE=base
TEMP=0.8

#CUDA_VISIBLE_DEVICES=3 python main.py \
#accelerate launch --multi_gpu --main_process_port 6666 --num_processes=5 --gpu_ids 3,4,5,6,7 main.py \
accelerate launch main.py \
  --model bigcode/starcoder2-15b \
  --tasks arcade-${MODE} \
  --max_length_generation 2560 \
  --max_new_tokens 128 \
  --temperature ${TEMP} \
  --do_sample True \
  --n_samples 40 \
  --batch_size 8 \
  --trust_remote_code \
  --allow_code_execution \
  --precision bf16 \
  --save_generations_path ${OUTPUT_PATH}/starcoder2_arcade${MODE}_temp${TEMP}.json \
  --generation_only \
  --save_generations
  #--limit 2 \
#Uncomment --limit 2 to run inference on only first 30 examples for prototyping
