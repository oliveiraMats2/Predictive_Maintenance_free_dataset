FROM ubuntu:20.04

LABEL maintainer="Mateus Oliveira da Silva <oliveira.mats.oo@gmail.com>"
ENV TZ=America/Sao_Paulo


RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get update && apt-get install -y \
    zip \
    python3-pip \
    libxml2-dev \
    libxmlsec1-dev \
    python3-dev \
    sox \
    ffmpeg \
    libcairo2 \
    libcairo2-dev \
    git \
    g++ \
    python \
    libeigen3-dev \
    zlib1g-dev \
    libgl1-mesa-dev \
    qt5-qmake \
    qtbase5-dev \
    libqt5svg5-dev

WORKDIR /app
# Clone o repositório
RUN git clone https://github.com/oliveiraMats2/Predictive_Maintenance_free_dataset.git

RUN pip3 install -r Predictive_Maintenance_free_dataset/requirements.txt

EXPOSE 5300
