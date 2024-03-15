Adding arcade evaluation based on [bigcode_eval](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main), haven't tested thoroughly.

Run with torch2.1.2 + cuda12.1

## Setup

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt && pip install -e .
   ```

   This will install the required `bigcode_eval` package.

2. **Obtain Arcade Assets**
   - Create an `arcade_assets` directory under `bigcode_eval_arcade/bigcode_eval/tasks`.
   - Inside `arcade_assets`, clone the [arcade-nl2code](https://github.com/google-research/arcade-nl2code.git) repository by the following
   
   `git clone https://github.com/google-research/arcade-nl2code.git`

   The folder structure should look like this:

   ```
   bigcode_eval_arcade/
   ├── bigcode_eval/
   │   ├── tasks/
   │   │   ├── arcade_assets/
   │   │   │   ├── arcade-nl2code/
   │   │   │   │   ├── annotated_dataset/
   │   │   │   │   │   ├── dataset/
   │   │   │   │   │   │   ├── new_tasks/
   │   │   │   │   │   │   │   ├── artifacts/
   │   │   │   │   │   │   │   ├── derived_datasets/
   │   │   │   │   │   │   │   └── ...
   │   │   │   │   │   └── ...
   │   │   │   │   └── ...
   │   │   └── ...
   │   └── ...
   └── ...
   ```

**Description of the Structure:**

- **arcade_assets/**: Root directory for all assets related to the arcade project. 
  
- **arcade_nl2code/**: Clone from [arcade-nl2code](https://github.com/google-research/arcade-nl2code.git)

- **annotated_dataset/**: Contains all datasets, divided into different structures and types as detailed below.

  - **dataset/**:
  
    - **new_tasks/**: A directory containing specific tasks and their related datasets and artifacts for the project.
    
      - **artifacts/**: Stores additional files or artifacts related to the `new_tasks`. 
      
      - **derived_datasets/**: Contains datasets that have been derived or processed from the base dataset for specific uses, such as training in different modes (`base`, `iosummary`, `ioexample`, `iotype`).

**Datasets:**

Each dataset under `derived_datasets/` follows the naming convention `arcade_{mode}.json`, indicating the specific task it's designed for (`base`, `iosummary`, `ioexample`, `iotype`).

## Generation

To generate code using a pre-trained language model (e.g., `bigcode/starcoder2`), run the following command:

```bash
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
--save_generations_path results/starcoder2_temp${TEMP}.json \
--generation_only \
--max_memory_per_gpu auto \
--save_generations
#--limit 30 \ #Uncomment this to run inference on only first 30 examples for prototyping
```

This command will generate code samples for the specified `MODE` (e.g., `base`, `iosummary`, `ioexample`, `iotype`) using the `bigcode/starcoder2` model with a temperature of `0.8`. The generated samples will be saved in the `results/starcoder2_temp${TEMP}.json` file.

## Evaluation (Need docker)

To evaluate the generated code, follow these steps:

1. **Convert Predictions to Arcade Format**

   ```bash
   python scripts/format_predictions.py --lm_eval_result ${save_generations_path}
   ```

   This script will convert the generated predictions from the JSON format to the JSONL format required by the [arcade-nl2code](https://github.com/google-research/arcade-nl2code.git).

2. **Run Arcade Evaluation**

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

   This command runs the Arcade evaluation using Docker. Make sure to replace the `prediction_file` and `output_path` arguments with the appropriate paths for your generated predictions and desired output location, respectively.
