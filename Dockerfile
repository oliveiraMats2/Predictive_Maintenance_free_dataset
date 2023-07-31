FROM oliveiramats/micro_service_math

LABEL maintainer="Mateus Oliveira da Silva <oliveira.mats.oo@gmail.com>"
ENV TZ=America/Sao_Paulo

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    cron \
    curl

RUN conda init bash

WORKDIR /app/Predictive_Maintenance_free_dataset
RUN git checkout inference

# Rode o comando: git pull --rebase
RUN git pull --rebase

# Rode o comando: pip install asyncua.
RUN pip install asyncua

RUN conda run -n wilec /bin/bash -c conda activate wilec

# Instale o crontab e configure para rodar a cada 1 minuto
RUN echo '* * * * * root conda run -n wilec python3 src/neural_prophet/inference_multi_variable.py src/neural_prophet/configs/inference_multi_variate.yaml' >> /etc/crontab

EXPOSE 5300
