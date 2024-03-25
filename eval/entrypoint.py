import argparse
import logging
import os
import pathlib
import time


logger = logging.getLogger(__name__)


def create_cmd(args):
    cmd = f"""accelerate launch main.py \
        --model {args.model_name_or_path} \
        --tasks arcade-{args.mode} \
        --max_length_generation 2560 \
        --max_new_tokens 128 \
        --temperature {args.temp} \
        --do_sample True \
        --n_samples 40 \
        --batch_size 8 \
        --trust_remote_code \
        --allow_code_execution \
        --precision bf16 \
        --save_generations_path {args.output_path}/starcoder2_arcade${args.mode}_temp${args.temp}.json \
        --generation_only \
        --save_generations"""
    # --limit 5 \
    return cmd


def main():
  arg_parser = argparse.ArgumentParser('')
  arg_parser.add_argument(
      '--model_name_or_path',
      type=str,
      default='./checkpoints/bigcode-starcoder',
  )
  available_modes = [f"{mode}-{context}" for mode in ["base", "iosummary", "ioexample", "iotype"] for context in ["withcontext", "nocontext"]]
  arg_parser.add_argument(
      '--mode',
      type=str,
      choices=available_modes,
      required=True,
  )
  arg_parser.add_argument(
      '--output_path',
      type=str,
      required=True,
  )
  arg_parser.add_argument(
      '--temp',
      type=float,
      required=True,
  )

  args = arg_parser.parse_args()

  logger.info('Args: %s', args)

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
