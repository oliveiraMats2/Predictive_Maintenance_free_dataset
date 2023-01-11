FROM ubuntu:20.04
MAINTAINER Mateus Oliveira da Silva "oliveira.mats.oo@gmail.com"

LABEL maintainer="wilec/ml_team:0.1"

RUN apt-get update

RUN apt install zip -y

RUN apt install python3-pip -y

RUN apt-get install libxml2-dev libxmlsec1-dev --fix-missing -y

RUN apt-get install python3-dev
RUN apt-get install sox ffmpeg libcairo2 libcairo2-dev --fix-missing -y
RUN apt-get pkg-config sox ffmpeg --fix-missing

WORKDIR /app

EXPOSE 5300

RUN ["git", "clone", "https://github.com/oliveiraMats2/Predictive_Maintenance_free_dataset.git"]

