from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from utils.read_dataset import ReadDatasets
import time



# dir_data = 'Datasets/sintetic_data/train_compressor_data.h5'

df = pd.read_csv(f"Datasets/saida/ite/payloadITE.csv")

# vector_series = ReadDatasets.read_h5(dir_data)
#
# df = pd.DataFrame({"temp_series": np.array(vector_series).astype(np.float),
#                    "ds": np.arange(len(vector_series)),
#                    "group_id": np.ones(len(vector_series)).astype(np.uint8)})

vector_series = df["temperature"].tolist()

#vector_series = vector_series[600:800]
vector_series = vector_series[:1500]

df = pd.DataFrame({"temp_series": np.array(vector_series).astype(np.float),
                   "ds": np.arange(len(vector_series)),
                   "group_id": np.ones(len(vector_series)).astype(np.uint8)})
# df = df.dropna()
df.index = pd.to_datetime(df.index)
earliest_time = df.index.min()
# df.sort_index(inplace=True)
# down sampling of the information

max_prediction_length = 1000
max_encoder_length = 30

training = TimeSeriesDataSet(
    df,
    time_idx="ds",
    target="temp_series",
    group_ids=["group_id"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    time_varying_known_reals=["ds"],
    time_varying_unknown_reals=["temp_series"],
    target_normalizer=GroupNormalizer(
        groups=["group_id"], transformation="softplus"
    ),  # we normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

batch_size = 64

# train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0, stop_randomization=True)
# validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
val_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
# val_dataloader = validation.to_dataloader(train=True, batch_size=batch_size * 10, num_workers=0)


# dir_model = "/mnt/arquivos_linux/wile_C/Predictive_Maintenance_free_dataset/lightning_logs/version_2/checkpoints/epoch=19-step=1540.ckpt"
# dir_model = "/mnt/arquivos_linux/wile_C/Predictive_Maintenance_free_dataset/lightning_logs/version_4/checkpoints/epoch=19-step=4800.ckpt"
# dir_model = "/mnt/arquivos_linux/wile_C/Predictive_Maintenance_free_dataset/lightning_logs/version_5/checkpoints/epoch=4-step=1600.ckpt"
dir_model = "/mnt/arquivos_linux/wile_C/Predictive_Maintenance_free_dataset/lightning_logs/version_8/checkpoints/epoch=1-step=7666.ckpt"

best_tft = TemporalFusionTransformer.load_from_checkpoint(dir_model)

actuals = torch.cat([y[0] for x, y in tqdm(iter(val_dataloader))])
predictions = best_tft.predict(val_dataloader, show_progress_bar=True)

x, y = next(iter(val_dataloader))

# dict_redefine_val = {}
# for key in x.keys():
#     if key == "decoder_target":
#         dict_redefine_val["decoder_target"] = torch.ones(len(x["decoder_target"][0]))
#
#     dict_redefine_val[key] = x[key]
#
# tuple(dict, None)

# average p50 loss overall
# print((actuals - predictions).abs().mean().item())

# average p50 loss per time series
# print((actuals - predictions).abs().mean(axis=1))

# raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)


print('\n')


# best_tft.plot_prediction(x, raw_predictions, idx=0, add_loss_to_title=True)
#predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
#best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
plt.show()
