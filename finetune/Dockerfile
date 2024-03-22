#FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310
#FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-1.py310
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cu121.py310

ARG PYTORCH='2.1.2'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu121'

RUN apt -y update
RUN apt install -y libaio-dev git

RUN pip uninstall -y torch torchvision torchaudio && pip install --no-cache-dir -U torch==$PYTORCH torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA

# Uninstall `transformer-engine` shipped with the base image
RUN pip uninstall -y transformer-engine

# Uninstall `torch-tensorrt` shipped with the base image
RUN pip uninstall -y torch-tensorrt

RUN pip uninstall -y deepspeed
RUN pip install deepspeed

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

RUN pip uninstall -y torch_xla

# Execute the script
CMD ["bash", "ft.sh"]
