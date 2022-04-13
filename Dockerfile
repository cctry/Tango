FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
COPY . /workspace
WORKDIR /workspace
RUN apt-get update \
    && apt-get install -y build-essential python3-dev make cmake git \
    && git clone https://github.com/dmlc/dgl.git \
    && cd dgl \
    && mkdir build \
    && cd build \
    && cmake -DUSE_CUDA=ON .. \
    && make -j \
    && cd ../python \
    && python3 setup.py install \
    && cd /workspace \
    && git clone https://github.com/cctry/cutlass.git
    
