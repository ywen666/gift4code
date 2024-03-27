import argparse
import logging
import os
import pathlib
import sys
import time


def create_cmd(args):
    cmd = f"""accelerate launch main.py \
        --model {args.model_name_or_path} \
        --tasks arcade-{args.mode} \
        --max_length_generation 2560 \
        --max_new_tokens 128 \
        --temperature {args.temp} \
        --do_sample True \
        --n_samples 50 \
        --batch_size 8 \
        --trust_remote_code \
        --allow_code_execution \
        --precision bf16 \
        --save_generations_path {args.output_path}/starcoder2_arcade{args.mode}_temp{args.temp}.json \
        --generation_only \
        --save_generations"""

    if args.limit > 0:
       cmd += f' --limit {args.limit}'

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
      default=0.8,
  )
  arg_parser.add_argument(
      '--limit',
      type=int,
      default=0,
  )

  args = arg_parser.parse_args()

  print(f'Args: {args}', file=sys.stderr)

  model_name_or_path = args.model_name_or_path
  if model_name_or_path.startswith('gs://'):
    print(f'Copying model: {model_name_or_path}', file=sys.stderr)
    pathlib.Path('/tmp/checkpoints').mkdir(parents=True, exist_ok=True)
    
    os.system(f'gsutil ls {model_name_or_path}')
    t0 = time.time()
    os.system(f'gsutil -m cp -r {model_name_or_path} /tmp/checkpoints')

    print(f'Time taken to copy model: {time.time() - t0}s', file=sys.stderr)

    print(
        f'Content of local ckpt folder: {os.listdir("/tmp/checkpoints")}',
        file=sys.stderr)
    args.model_name_or_path = os.path.join(
        '/tmp/checkpoints', os.path.basename(model_name_or_path))

  cmd = create_cmd(args)
  print(f'CMD: {cmd}')
  sys.stdout.flush()
  sys.stderr.flush()

  os.system(cmd)


if __name__ == '__main__':
  main()
