FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    trl[vllm] \
    peft \
    accelerate \
    datasets \
    bitsandbytes \
    pandas \
    pyarrow \
    matplotlib \
    liger-kernel \
    scipy \
    scikit-learn \
    tqdm

RUN pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu130torch2.9-cp311-cp311-linux_x86_64.whl

CMD ["/bin/bash"]
