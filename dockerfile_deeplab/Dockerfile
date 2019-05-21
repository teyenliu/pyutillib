FROM nvidia/cuda:9.0-devel-ubuntu16.04

RUN apt-get update && apt-get install -y \
    autoconf \
    cmake \
    curl \
    git \
    libatlas-base-dev \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libcudnn7=7.4.1.5-1+cuda9.0 \
    libcudnn7-dev=7.4.1.5-1+cuda9.0 \
    libleveldb-dev \
    liblmdb-dev \
    libhdf5-serial-dev \
    libopenblas-dev \
    libopencv-dev \
    libopencv-highgui-dev \
    libturbojpeg \
    libsnappy-dev \
    openssh-server \
    python-dev \
    python-opencv \
    python-pip \
    python-tk \
    python-pil \
    python-numpy \
    sudo \
    vim \
    wget \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY . /build/

RUN cd /build
RUN git clone https://github.com/tensorflow/models.git 

RUN pip install --upgrade pip
RUN pip install tensorflow-gpu==1.12.0 \
    jupyter \
    matplotlib \
    PrettyTable 

WORKDIR /build/models/research

ENV PYTHONPATH "${PYTHONPATH}:/build/models/research:/build/models/research/slim"
ENTRYPOINT ["python", "deeplab/train.py"]

