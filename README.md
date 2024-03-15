
Adding arcade evaluation based on [bigcode_eval](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main), haven't tested thoroughly.

## Generation only
The folder structure looks like the following

### Folder Structure
```
bigcode_eval_arcade(root)
└── bigcode_eval/
    └── tasks/
        └── arcade_assets/
            └── arcade_nl2code/
                └── annotated_dataset/
                    ├── dataset/
                    │   ├── new_tasks/
                    │   │   ├── artifacts/
                    │   │   └── derived_datasets/
                    │   └── ... (other potential directories or files)
                    └── ... (other potential directories or files)
```

### Description of the Structure

- **arcade_assets/**: Root directory for all assets related to the arcade project. 
  
- **arcade_nl2code/**: Subdirectory within `arcade_assets` dedicated to the natural language to code conversion for arcade games. It serves as the main container for the project's annotated datasets and related files.

- **annotated_dataset/**: Contains all datasets annotated for the project, divided into different structures and types as detailed below.

  - **dataset/**:
  
    - **new_tasks/**: A directory containing specific tasks and their related datasets and artifacts for the project.
    
      - **artifacts/**: Stores additional files or artifacts related to the `new_tasks`. 
      
      - **derived_datasets/**: Contains datasets that have been derived or processed from the base dataset for specific uses, such as training in different modes (`base`, `iosummary`, `ioexample`, `iotype`).

***Each dataset under `derived_datasets` is named using the pattern `arcade_{mode}.json`, where `{mode}` represents the specific mode the dataset is intended for, such as `base`, `iosummary`, `ioexample`, or `iotype`***

```bash
MODE=base
TEMP=0.8
accelerate launch  main.py \
  --model bigcode/starcoder2 \
  --tasks arcade-${MODE} \
  --max_length_generation 2560 \
  --temperature ${TEMP} \
  --do_sample True \
  --n_samples 10 \
  --batch_size 1 \
  --trust_remote_code \
  --allow_code_execution \
  --precision bf16 \
  --save_generations_path results/starcoder2_temp${TEMP}.json \
  --generation_only \
  --max_memory_per_gpu auto \
  --save_generations
```

## Evaluation

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
