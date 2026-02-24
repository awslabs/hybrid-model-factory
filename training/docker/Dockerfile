FROM nvcr.io/nvidia/pytorch:25.01-py3

# Pre-install PyTorch 2.10.0 with CUDA 12.8
RUN pip install --no-cache-dir \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install core packages with build isolation disabled for CUDA kernels
RUN pip install --no-cache-dir \
    mistral-common \
    tiktoken \
    numpy==1.26.4

RUN pip install --no-cache-dir --no-build-isolation \
    flash-attn==2.8.3 \
    mamba-ssm==2.3.0 \
    causal-conv1d==1.6.0 \
    flash-linear-attention==0.4.1 \
    ring-flash-attn==0.1.8

# Install HuggingFace packages
RUN pip install --no-cache-dir \
    yunchang==0.6.1 \
    transformers==5.3.0 \
    datasets==3.4.1 \
    huggingface_hub==1.3.7 \
    tokenizers==0.22.2

# Install accelerator packages
RUN pip install --no-cache-dir \
    deepspeed==0.17.6 \
    triton==3.6.0

# Install metrics and evaluation packages
RUN pip install --no-cache-dir \
    nltk \
    jieba \
    rouge-chinese \
    pytrec_eval \
    rouge_score \
    tensorboard

# Apply Qwen3.5 position IDs patch
COPY patches/qwen3_5_position_ids_fix.patch /tmp/
RUN patch -p1 -N -d /usr/local/lib/python3.12/dist-packages < /tmp/qwen3_5_position_ids_fix.patch
