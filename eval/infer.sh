MODE=base
TEMP=0.8

accelerate launch main.py \
--model bigcode/starcoder2-15b \
--tasks arcade-${MODE} \
--max_length_generation 2560 \
--temperature ${TEMP} \
--do_sample True \
--n_samples 10 \
--batch_size 1 \
--trust_remote_code \
--allow_code_execution \
--precision bf16 \
--save_generations_path ${OUTPUT_PATH}/starcoder2_arcade${MODE}_temp${TEMP}.json \
--generation_only \
--max_memory_per_gpu auto \
--limit 2 \
--save_generations
#Uncomment --limit 2 to run inference on only first 30 examples for prototyping