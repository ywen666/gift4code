FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310

# Clone the specified repository
WORKDIR /workspace
RUN git clone https://github.com/ywen666/gift4code.git

# Set the working directory to the 'eval' directory within the cloned repo
WORKDIR /workspace/gift4code/eval

RUN pip install .
#RUN pip install -r requirements.txt

ENV HF_HOME=/workspace/huggingface
ENV OUTPUT_PATH=/workspace/gift4code/eval
ENV MODEL_PATH=bigcode/starcoder2-15b

RUN pip uninstall -y torch_xla
RUN pip install human-eval
RUN pip install evaluate accelerate datasets git+https://github.com/huggingface/transformers.git

# Execute the script
CMD ["bash", "scripts/infer.sh"]
