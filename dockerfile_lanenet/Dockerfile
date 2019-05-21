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
    libturbojpeg \
    libsnappy-dev \
    openssh-server \
    python-dev \
    python-opencv \
    python-pip \
    python-tk \
    python3-dev \
    python3-pip \
    sudo \
    vim \
    wget \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY . /build/

RUN cd /build
RUN git clone https://github.com/MaybeShewill-CV/lanenet-lane-detection.git 
RUN cd /build/lanenet-lane-detection/data ; \
    wget https://gitlab.com/Daikon-Sun/MLDS2017_final_ver1/raw/master/Vgg16/vgg16.npy 

RUN /usr/bin/pip3 install tensorflow-gpu==1.12.0 \
    numpy \
    opencv-python \
    glog \
    config \
    easydict

WORKDIR /build/lanenet-lane-detection

# initial module
RUN touch /build/lanenet-lane-detection/config/__init__.py && \
    touch /build/lanenet-lane-detection/data_provider/__init__.py && \
    touch /build/lanenet-lane-detection/encoder_decoder_model/__init__.py && \
    touch /build/lanenet-lane-detection/lanenet_model/__init__.py && \
    touch /build/lanenet-lane-detection/tboard/__init__.py && \
    touch /build/lanenet-lane-detection/tools/__init__.py

ENV PYTHONPATH "${PYTHONPATH}:/build/lanenet-lane-detection"
ENTRYPOINT ["python3", "tools/train_lanenet.py"]

