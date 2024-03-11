
Adding arcade evaluation based on [bigcode_eval](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main), haven't tested thoroughly.

### Generation only

```bash
accelerate launch  main.py \
  --model codellama/CodeLlama-70b-Python-hf \
  --tasks arcade \
  --max_length_generation 2560 \
  --temperature 0.8 \
  --do_sample True \
  --n_samples 10 \
  --batch_size 1 \
  --trust_remote_code \
  --allow_code_execution \
  --precision bf16 \
  --save_generations_path results/arcade_codellama70b_python_temp0.8.json \
  --generation_only \
  --max_memory_per_gpu auto \
  --save_generations
```

### Evaluation only

Need to convert the saved generation json file to the jsonl that can be used in the Arcade evaluation, [arcade-nl2code](https://github.com/google-research/arcade-nl2code.git)
by running the following
```bash
python scripts/format_predictions.py --lm_eval_result ${save_generations_path}
```

Then, following the evaluation instructions in [arcade-nl2code](https://github.com/google-research/arcade-nl2code.git), which looks similar to the following,

```bash
PROJECT_ROOT="$(dirname `pwd`)"
docker run -it --shm-size=2g \
  --mount type=bind,source=${PROJECT_ROOT}/evaluation/arcade_results_codellama,target=/data \
  --mount type=bind,source=${PROJECT_ROOT}/annotated_dataset/dataset/new_tasks/artifacts,target=/artifacts \
  -w / \
  --entrypoint /opt/conda/bin/python \
  notebook_evaluator:latest \
  -m arcade_nl2code.evaluation.execution_evaluation_main \
  --prediction_file /data/CodeLlama-34b-hf_zeroshot_predictions.jsonl \
  --output_path /data/eval_results/CodeLlama-34b-hf_zeroshot \
  --runtime_artifact_root /artifacts \
  --lm_output_postprocessor extract_first_cell_block \
  --split_episode \
  --noreuse_state \
  --timeout 30 \
  --num_workers 50
```
