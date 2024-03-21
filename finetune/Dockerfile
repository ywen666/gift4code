# Use the specified PyTorch image with CUDA
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
#Use the following one if the above has deepspeed error
#FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Install git
RUN apt-get update && apt-get install -y git

# Clone the specified repository
RUN git clone https://github.com/ywen666/gift4code.git

WORKDIR /workspace/gift4code
RUN pip install -r requirements.txt

# Set the working directory to the 'finetune' directory within the cloned repo
WORKDIR /workspace/gift4code/finetune

# Set the required environment variables (replace <value> with actual paths/values)
ENV HF_HOME=<value>
ENV DATA_PATH=<value, not sure if a gcp bucket will work>
ENV OUTPUT_PATH=<value>
ENV MODEL_PATH=<value>

# Execute the script
CMD ["bash", "ft.sh"]
