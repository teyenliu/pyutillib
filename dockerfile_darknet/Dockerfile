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
    sudo \
    vim \
    wget \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY task.py /build/task.py
COPY darknet /build/darknet
#COPY Dataset /Dataset

RUN cd /build
#RUN git clone https://github.com/pjreddie/darknet ; \
#    cd darknet ; \
#    sed -i -- 's/GPU=0/GPU=1/g' Makefile ; \
#    sed -i -- 's/CUDNN=0/CUDNN=1/g' Makefile ; \
#    make -j8

#RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf
#RUN ldconfig

RUN cd darknet ; \
    make clean ; \
    make -j8

RUN pip install tensorflow-gpu==1.12.0

ENTRYPOINT ["python", "task.py"]

