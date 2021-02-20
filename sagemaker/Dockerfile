#FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y --no-install-recommends \
		ca-certificates \
		curl \
		netbase \
		wget \
       build-essential \
        libopencv-dev\
        libopenblas-dev \
        nginx

ENV PYTHON_VERSION=3.6

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION 

RUN pip install tensorflow-gpu

#COPY ./tensorflow-2.3.0-cp38-cp38-linux_x86_64.whl /opt/
#RUN pip install /opt/tensorflow-2.3.0-cp38-cp38-linux_x86_64.whl 
RUN pip install scipy pillow gunicorn==19.9.0 gevent flask

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY ./scripts /opt/program

RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve

WORKDIR /opt/program



