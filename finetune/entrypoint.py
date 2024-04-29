import argparse
import logging
import os
import pathlib
import time


logger = logging.getLogger(__name__)


def create_cmd(args):
    cmd = f"""deepspeed finetune.py \
        --model_name_or_path {args.model_name_or_path} \
        --data_path {args.data_path} \
        --output_dir {args.output_path} \
        --num_train_epochs 5 \
        --model_max_length 1024 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "steps" \
        --save_strategy "steps" \
        --save_steps 100 \
        --eval_steps 100 \
        --save_total_limit 100 \
        --learning_rate 2e-5 \
        --warmup_steps 10 \
        --logging_steps 1 \
        --lr_scheduler_type "cosine" \
        --gradient_checkpointing True \
        --deepspeed configs/ds_config_zero3.json \
        --bf16 True"""

    return cmd


def main():
  arg_parser = argparse.ArgumentParser('')
  arg_parser.add_argument(
      '--model_name_or_path',
      type=str,
      default='./checkpoints/bigcode-starcoder',
  )
  arg_parser.add_argument(
      '--data_path',
      type=str,
      required=True,
  )
  arg_parser.add_argument(
      '--output_path',
      type=str,
      required=True,
  )

  args = arg_parser.parse_args()

  logger.info('Args: %s', args)

  json_data_path = pathlib.Path(args.data_path).with_suffix('.json')
  os.system(f"python convert_tfrecord_to_hfdataset.py --tf_data_path {args.data_path} --json_path {json_data_path}")
  print(f'Converted data from {args.data_path} to {json_data_path}')
  args.data_path = json_data_path

  model_name_or_path = args.model_name_or_path
  if model_name_or_path.startswith('gs://'):
    logger.info('Copying model: %s', model_name_or_path)
    pathlib.Path('/tmp/checkpoints').mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    os.system(f'gsutil -m cp -r {model_name_or_path} /tmp/checkpoints')
    logger.info('Time taken to copy model: %ss', time.time() - t0)

    logger.info(
        'Content of local ckpt folder: %s', os.listdir('/tmp/checkpoints'))
    args.model_name_or_path = os.path.join(
        '/tmp/checkpoints', os.path.basename(model_name_or_path))

  cmd = create_cmd(args)
  logging.info('CMD: %s', cmd)
  os.system(cmd)


if __name__ == '__main__':
  main()
