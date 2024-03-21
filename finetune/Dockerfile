FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310

# Clone the specified repository
WORKDIR /workspace
RUN git clone https://github.com/ywen666/gift4code.git

WORKDIR /workspace/gift4code
RUN pip install -r requirements.txt

# Set the working directory to the 'finetune' directory within the cloned repo
WORKDIR /workspace/gift4code/finetune

# Set the required environment variables (replace <value> with actual paths/values)
ENV HF_HOME=/workspace/huggingface
ENV DATA_PATH=/workspace/gift4code/finetune/gift4code.tfrecord
ENV OUTPUT_PATH=/workspace/gift4code/finetune/results
ENV MODEL_PATH=bigcode/starcoder2-15b

RUN python3 -m pip uninstall -y torch_xla

# Execute the script
CMD ["bash", "ft.sh"]
