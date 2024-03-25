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

RUN gsutil version

# Clone the specified repository
WORKDIR /workspace
COPY . /workspace
RUN pip --no-cache-dir install -r /workspace/requirements.txt

# Make 1000 UID and root user as owners of the workspace.
RUN chown -R 1000:root /workspace && chmod -R 775 /workspace

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib/"
ENV HUGGINGFACE_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TRITON_CACHE_DIR="/tmp/triton"
# ENV TRANSFORMERS_CACHE="/tmp/transformer"
ENV HF_HOME="/tmp/transformer"

# Execute the script
ENTRYPOINT ["python3", "entrypoint.py"]
