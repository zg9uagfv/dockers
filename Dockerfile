FROM nvidia/cuda:8.0-devel-ubuntu16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 6.0.21
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN echo "deb http://mirrors.163.com/ubuntu/ xenial main restricted universe multiverse" > /etc/apt/sources.list
RUN echo "deb http://mirrors.163.com/ubuntu/ xenial-security main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb http://mirrors.163.com/ubuntu/ xenial-updates main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb http://mirrors.163.com/ubuntu/ xenial-proposed main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb http://mirrors.163.com/ubuntu/ xenial-backports main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb-src http://mirrors.163.com/ubuntu/ xenial main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb-src http://mirrors.163.com/ubuntu/ xenial-security main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb-src http://mirrors.163.com/ubuntu/ xenial-updates main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb-src http://mirrors.163.com/ubuntu/ xenial-proposed main restricted universe multiverse" >> /etc/apt/sources.list
RUN echo "deb-src http://mirrors.163.com/ubuntu/ xenial-backports main restricted universe multiverse" >> /etc/apt/sources.list

RUN apt-get update && apt-get install apt-utils -y 
RUN apt-get install -y --no-install-recommends \
            libcudnn6=$CUDNN_VERSION-1+cuda8.0 \
            libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0 && \
    rm -rf /var/lib/apt/lists/*

# Supress warnings about missing front-end. As recommended at:
# http://stackoverflow.com/questions/22466255/is-it-possibe-to-answer-dialog-questions-when-installing-under-docker
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils

# Developer Essentials
RUN apt-get install -y --no-install-recommends git-core curl vim unzip openssh-server wget

# Build tools
RUN apt-get install -y --no-install-recommends build-essential cmake

# OpenBLAS
RUN apt-get install -y --no-install-recommends libopenblas-dev

#
# Python 3.5
#
# For convenience, alisas (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems
RUN mkdir -p ~/.config/pip
RUN echo "[global]" >> ~/.config/pip/pip.conf
RUN echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> ~/.config/pip/pip.conf

RUN apt-get install -y --no-install-recommends python3.5 python3.5-dev python3-pip python3-tk
RUN pip3 install --no-cache-dir --upgrade pip setuptools
RUN echo "alias python='python3'" >> /root/.bash_aliases
RUN echo "alias pip='pip3'" >> /root/.bash_aliases
# Pillow and it's dependencies
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev
RUN pip3 --no-cache-dir install Pillow
# Common libraries
RUN pip3 --no-cache-dir install \
    numpy scipy sklearn scikit-image pandas matplotlib requests
# Cython
RUN pip3 --no-cache-dir install Cython

#
# Jupyter Notebook
#
RUN pip3 --no-cache-dir install jupyter
# Allow access from outside the container, and skip trying to open a browser.
# NOTE: disable authentication token for convenience. DON'T DO THIS ON A PUBLIC SERVER.
RUN mkdir /root/.jupyter
RUN echo "c.NotebookApp.ip = '*'" \
         "\nc.NotebookApp.open_browser = False" \
         "\nc.NotebookApp.token = ''" \
         > /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888

#
# Tensorflow 1.4.1 - CPU
#
RUN pip3 install futures==3.1.1
#RUN pip3 install --no-cache-dir --upgrade tensorflow 
RUN pip3 install --no-cache-dir tensorflow-gpu==1.5.0

# Expose port for TensorBoard
EXPOSE 6006

#
# OpenCV 3.4
#
# Dependencies
RUN apt-get update
RUN apt-get upgrade -y

#OS libraries
RUN apt-get remove x264 libx264-dev -y

RUN apt-get install build-essential checkinstall cmake pkg-config yasm -y
RUN apt-get install git gfortran -y
RUN apt-get install libjpeg8-dev libjasper-dev libpng12-dev -y
 
# If you are using Ubuntu 16.04
RUN apt-get install libtiff5-dev -y
 
RUN apt-get install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev -y
RUN apt-get install libxine2-dev libv4l-dev -y
RUN apt-get install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev -y
RUN apt-get install qt5-default libgtk2.0-dev libtbb-dev -y
RUN apt-get install libatlas-base-dev -y
RUN apt-get install libfaac-dev libmp3lame-dev libtheora-dev -y
RUN apt-get install libvorbis-dev libxvidcore-dev -y
RUN apt-get install libopencore-amrnb-dev libopencore-amrwb-dev -y
RUN apt-get install x264 v4l-utils -y
 
# Optional dependencies
RUN apt-get install libprotobuf-dev protobuf-compiler -y
RUN apt-get install libgoogle-glog-dev libgflags-dev -y
RUN apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev -y


RUN apt-get install -y --no-install-recommends \
    libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
    liblapacke-dev checkinstall -y

#Install Python libraries
RUN apt-get install python-dev python-pip python3-dev python3-pip -y
# Get source from github
RUN git clone -b 3.4.0 --depth 1 https://github.com/opencv/opencv.git /usr/local/src/opencv
RUN git clone -b 3.4.0 --depth 1 https://github.com/opencv/opencv_contrib.git /usr/local/src/opencv_contrib
# Compile
RUN cd /usr/local/src/opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D INSTALL_C_EXAMPLES=ON \
          -D INSTALL_PYTHON_EXAMPLES=ON \
          -D WITH_TBB=ON \
          -D WITH_V4L=ON \
          -D WITH_QT=ON \
          -D WITH_CUDA=ON \
          -D WITH_DNN=ON \
          -D WITH_OPENGL=ON \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -D BUILD_EXAMPLES=OFF .. && \
    make -j"$(nproc)" && \
    make install

#
# Caffe
#
# Dependencies
RUN apt-get install -y --no-install-recommends \
    cmake libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev \
    libhdf5-serial-dev protobuf-compiler liblmdb-dev libgoogle-glog-dev
RUN apt-get install -y --no-install-recommends libboost-all-dev
RUN pip3 install lmdb
# Get source. Use master branch because the latest stable release (rc3) misses critical fixes.
RUN git clone -b master --depth 1 https://github.com/BVLC/caffe.git /usr/local/src/caffe
# Python dependencies
RUN pip3 --no-cache-dir install -r /usr/local/src/caffe/python/requirements.txt
# Compile
RUN cd /usr/local/src/caffe && mkdir build && cd build && \
    cmake -D CPU_ONLY=ON -D python_version=3 -D BLAS=open -D USE_OPENCV=ON .. && \
    make -j"$(nproc)" all && \
    make install
# Enivronment variables
ENV PYTHONPATH=/usr/local/src/caffe/python:$PYTHONPATH \
	PATH=/usr/local/src/caffe/build/tools:$PATH
# Fix: old version of python-dateutil breaks caffe. Update it.
RUN pip3 install --no-cache-dir python-dateutil --upgrade

#
# Java
#
# Install JDK (Java Development Kit), which includes JRE (Java Runtime
# Environment). Or, if you just want to run Java apps, you can install
# JRE only using: apt install default-jre
RUN apt-get install -y --no-install-recommends default-jdk

#
# Keras 2.1.2
#
RUN pip3 install --no-cache-dir --upgrade h5py pydot_ng keras

#
# PyCocoTools
#
# Using a fork of the original that has a fix for Python 3.
# I submitted a PR to the original repo (https://github.com/cocodataset/cocoapi/pull/50)
# but it doesn't seem to be active anymore.
RUN pip3 install --no-cache-dir git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

#
# PyTorch 0.2
#
#RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
#RUN pip3 install torchvision

#
# Cleanup
#
RUN apt-get clean && \
    apt-get autoremove

WORKDIR "/root"
CMD ["/bin/bash"]
