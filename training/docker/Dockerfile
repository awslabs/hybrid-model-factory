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
    deepspeed==0.18.9 \
    triton==3.6.0

# Install metrics and evaluation packages
RUN pip install --no-cache-dir \
    nltk \
    jieba \
    rouge-chinese \
    pytrec_eval \
    rouge_score \
    tensorboard

# Apply DeepSpeed MiCS patches
COPY patches/ds_mics.patch /tmp/ds_mics.patch
COPY patches/ds_torch.patch /tmp/ds_torch.patch
RUN patch -N /usr/local/lib/python3.12/dist-packages/deepspeed/runtime/zero/mics.py /tmp/ds_mics.patch
RUN patch -N /usr/local/lib/python3.12/dist-packages/deepspeed/comm/torch.py /tmp/ds_torch.patch

# Apply patches for deepspeed async checkpointing
COPY plugins/async_torch_checkpoint_engine.py /usr/local/lib/python3.12/dist-packages/deepspeed/runtime/checkpoint_engine/
COPY patches/ds_config.patch /tmp/ds_config.patch
COPY patches/ds_utils.patch /tmp/ds_utils.patch
RUN patch -p1 -N /usr/local/lib/python3.12/dist-packages/deepspeed/runtime/checkpoint_engine/utils.py < /tmp/ds_utils.patch
RUN patch -p1 -N /usr/local/lib/python3.12/dist-packages/deepspeed/runtime/zero/config.py < /tmp/ds_config.patch

# Apply Qwen3.5 position IDs patch
COPY patches/qwen3_5_position_ids_fix.patch /tmp/
RUN patch -p1 -N -d /usr/local/lib/python3.12/dist-packages < /tmp/qwen3_5_position_ids_fix.patch

# Fix DeepSpeed zero_to_fp32.py shape.numel() bug for tuple shapes
COPY patches/ds_zero_to_fp32.patch /tmp/ds_zero_to_fp32.patch
RUN patch -p1 -N -d /usr/local/lib/python3.12/dist-packages < /tmp/ds_zero_to_fp32.patch
