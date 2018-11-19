# Use CUDA 9.0 with cudnn 
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
#FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

WORKDIR /deeplab

# Install some dependencies
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libpng12-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
		nano \
		pkg-config \
		python-dev \
		software-properties-common \
		unzip \
		vim \
		wget \
		zlib1g-dev \
		qt5-default \
		libvtk6-dev \
		zlib1g-dev \
		libjpeg-dev \
		libwebp-dev \
		libpng-dev \
		libtiff5-dev \
		libjasper-dev \
		libopenexr-dev \
		libgdal-dev \
		libdc1394-22-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libtheora-dev \
		libvorbis-dev \
		libxvidcore-dev \
		libx264-dev \
		yasm \
		libopencore-amrnb-dev \
		libopencore-amrwb-dev \
		libv4l-dev \
		libxine2-dev \
		libtbb-dev \
        libeigen3-dev \
        python3 \
        && \
	apt-get clean && \
	apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Get python3.5
#RUN add-apt-repository ppa:deadsnakes/ppa && \
#    apt-get update && \
#    apt-get install python3.5

# Get pip

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Get the git repo
RUN pip3 install tensorflow-gpu==1.10.1

RUN git clone https://github.com/ankur-gos/deeplab_v3.git

# Copy the data

COPY records.zip /deeplab/deeplab_v3/





