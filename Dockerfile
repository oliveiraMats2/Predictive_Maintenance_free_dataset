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
    libqt5svg5-dev \
    curl  # Adiciona o pacote 'curl' para instalar o Miniconda

# Instala o Miniconda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda
RUN rm Miniconda3-latest-Linux-x86_64.sh

# Adiciona o Miniconda ao PATH
ENV PATH=/opt/miniconda/bin:$PATH

# Cria o ambiente 'wilec' com o Python 3.9
RUN conda create -n wilec python=3.9 -y

# Ativa o ambiente 'wilec'
SHELL ["conda", "run", "-n", "wilec", "/bin/bash", "-c"]



# Clone o repositório
WORKDIR /app
RUN git clone https://github.com/oliveiraMats2/Predictive_Maintenance_free_dataset.git
WORKDIR /app/Predictive_Maintenance_free_dataset
RUN git checkout inference
RUN pip install -r requirements.txt

EXPOSE 5300
