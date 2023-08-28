#!/bin/bash

# Ativa o ambiente conda "wilec"
source activate wilec

# Executa o script Python
python3 src/neural_prophet/cron_execute_model.py src/neural_prophet/configs/inference_multi_variate.yaml
