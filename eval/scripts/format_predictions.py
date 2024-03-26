import argparse
import copy
import json
import os
import sys

from pathlib import Path


arcade_root = ("/mnt/sdd2/ywen/code/arcade_lm_eval/bigcode-evaluation-harness/"
               "lm_eval/tasks/arcade_assets")
arcade_root = os.getenv("ARCADE_ROOT")
if not arcade_root:
    arcade_root = ("/mnt/sdd2/ywen/code/arcade_lm_eval/bigcode-evaluation-harness/"
                   "lm_eval/tasks/arcade_assets")
sys.path.insert(0, arcade_root)
from arcade_nl2code.annotated_dataset import dataset as dataset_module


def generate_dataset_predictions(dataset, predictions):
    final_predictions = []
    prediction_idx = 0

    for episode in dataset:
        episode_prediction = dict(
            metadata={k: v for k, v in episode.items() if k != 'turns'},
            turns=[]
        )

        for turn_example in episode['turns']:
            turn_pred_entry = dict(
                metadata=dict(example=turn_example),
                predictions=copy.deepcopy(predictions[prediction_idx]),
            )

            episode_prediction['turns'].append(turn_pred_entry)
            prediction_idx += 1

        final_predictions.append(episode_prediction)

    print(f"transfered {prediction_idx} predictions")
    return final_predictions


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--lm_generation_result', required=True)
    arg_parser.add_argument(
        "--few_shot",
        action="store_true",
        help="whether use the few-shot dataset"
    )
    args = arg_parser.parse_args()

    data_root = Path(arcade_root) / "arcade_nl2code/annotated_dataset/dataset/new_tasks/derived_datasets"
    if args.few_shot:
        data_name = "dataset.schema.originating_dfs.header_description.after_variable_cell.maxp2100.maxp_no_prefix-1.maxctxcell-1.json"
        print("use the few shot dataset")
    else:
        data_name = "dataset.schema.originating_dfs.header_description.after_variable_cell.maxp2100.maxp_no_prefix900.maxctxcell-1.e0_1_4_5.vanilla_prompting.json"
        print("use the NON few shot dataset")
    dataset = dataset_module.load_dataset(Path(data_root) / data_name)

    #result_path = "results/codellama13b_instruct_temp0.8.json"
    with open(args.lm_generation_result, "r") as f:
        lm_generation_predictions = json.load(f)
    predictions = generate_dataset_predictions(dataset, lm_generation_predictions)

    eval_folder = Path(
        arcade_root) / "arcade_nl2code/evaluation/results" / Path(
            args.lm_generation_result).stem

    Path(eval_folder).mkdir(parents=True, exist_ok=True)
    dataset_module.save_dataset(predictions, eval_folder / "raw_predictions.jsonl")
    print(f"save raw predictions to '{eval_folder}'/raw_predictions.jsonl")
