# NVIDIA CUDA image as a base
FROM nvidia/cuda:11.6.0-base-ubuntu20.04

# Install Python and some utilities
RUN apt-get update && apt install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    sudo 2>&1

#RUN useradd -m ubuntu 
#RUN useradd -m /home/tkonuk -s /bin/bash -g root -G sudo -u 1001 tkonuk
RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers


RUN chown -R ubuntu:ubuntu /home/ubuntu/

COPY --chown=ubuntu . /home/ubuntu/app

USER ubuntu

# Install all the basic packages 
RUN pip3 install \
    # Numpy and Pandas 
    numpy pandas matplotlib \
    # PyTorch and torchvision
    torch torchvision torchtext --extra-index-url https://download.pytorch.org/whl/cu113 \
    # HuggingFace transformers
    transformers 2>&1


WORKDIR /home/ubuntu/app

#RUN pip3 install -r /home/ubuntu/app/requirements.txt & pip3 install -r /home/ubuntu/app/examples/flava/requirements.txt 2>&1
RUN pip3 install -r requirements.txt & pip3 install -r examples/flava/requirements.txt 2>&1

RUN sudo python3 setup.py install

# build with
# docker build --network=host -t multimodal:training .
# run with 

