FROM nvcr.io/nvidia/pytorch:23.04-py3
ARG PYTORCH='2.1.2'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu121'

RUN apt -y update
RUN apt install -y libaio-dev git
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
#RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

#RUN python3 -m pip install --no-cache-dir ./transformers[deepspeed-testing]

# Install latest release PyTorch
# (PyTorch must be installed before pre-compiling any DeepSpeed c++/cuda ops.)
# (https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops)
RUN python3 -m pip uninstall -y torch torchvision torchaudio && python3 -m pip install --no-cache-dir -U torch==$PYTORCH torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA

# Uninstall `transformer-engine` shipped with the base image
RUN python3 -m pip uninstall -y transformer-engine

# Uninstall `torch-tensorrt` shipped with the base image
RUN python3 -m pip uninstall -y torch-tensorrt

# Pre-build **latest** DeepSpeed, so it would be ready for testing (otherwise, the 1st deepspeed test will timeout)
RUN python3 -m pip uninstall -y deepspeed
# This has to be run (again) inside the GPU VMs running the tests.
# The installation works here, but some tests fail, if we don't pre-build deepspeed again in the VMs running the tests.
# TODO: Find out why test fail.
#RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 python3 -m pip install deepspeed --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1
#RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 
RUN python3 -m pip install deepspeed

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
ENV OUTPUT_PATH=/workspace/finetune/gift4code/results
ENV MODEL_PATH=bigcode/starcoder2-15b

# Execute the script
CMD ["bash", "ft.sh"]
