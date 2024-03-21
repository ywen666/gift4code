FROM nvcr.io/nvidia/pytorch:23.04-py3
ARG PYTORCH='2.1.2'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu121'

RUN apt -y update
RUN apt install -y libaio-dev git
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install latest release PyTorch
# (PyTorch must be installed before pre-compiling any DeepSpeed c++/cuda ops.)
# (https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops)
RUN python3 -m pip uninstall -y torch torchvision torchaudio && python3 -m pip install --no-cache-dir -U torch==$PYTORCH torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA

# Uninstall `transformer-engine` shipped with the base image
RUN python3 -m pip uninstall -y transformer-engine

# Uninstall `torch-tensorrt` shipped with the base image
RUN python3 -m pip uninstall -y torch-tensorrt

RUN python3 -m pip uninstall -y deepspeed

# Clone the specified repository
WORKDIR /workspace
RUN git clone https://github.com/ywen666/gift4code.git

WORKDIR /workspace/gift4code
RUN pip install -r requirements.txt

# Set the working directory to the 'eval' directory within the cloned repo
WORKDIR /workspace/gift4code/eval

RUN pip3 install .

ENV HF_HOME=/workspace/huggingface
ENV OUTPUT_PATH=/workspace/gift4code/eval/results
ENV MODEL_PATH=bigcode/starcoder2-15b

# Execute the script
CMD ["bash", "infer.sh"]
