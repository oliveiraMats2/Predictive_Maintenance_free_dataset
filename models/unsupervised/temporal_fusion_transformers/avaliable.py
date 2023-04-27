from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from configs import *
from pytorch_forecasting import TimeSeriesDataSet
from utils.read_dataset import ReadDatasets
from utils.save_numpy import SaveNumpy
from pytorch_forecasting.data.encoders import GroupNormalizer

weights_path = "../../../models_h5/lightning_logs/version_21/checkpoints/epoch=22-step=4048.ckpt"

dir_data = "../../../Datasets/sintetic_dataset/9100_points/test_compressor_data.h5"
dir_model = "../../../models_h5/"

vector_series = ReadDatasets.read_h5(dir_data)

length_vector_series = len(vector_series)

#vector_series = vector_series[:length_vector_series//2]

print(f"base test length: {length_vector_series}")

df = pd.DataFrame({"temp_series": np.array(vector_series).astype(np.float),
                   "ds": np.arange(len(vector_series)),
                   "group_id": np.ones(len(vector_series)).astype(np.uint8)})

df.index = pd.to_datetime(df.index)
earliest_time = df.index.min()

max_prediction_length = configs["max_prediction_length"]
max_encoder_length = configs["max_encoder_length"]

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

batch_size = 32

# train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0, stop_randomization=True)
# validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
test_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)


best_tft = TemporalFusionTransformer.load_from_checkpoint(weights_path)

actuals = torch.cat([y[0] for x, y in tqdm(iter(test_dataloader))])
# predictions = best_tft.predict(test_dataloader, show_progress_bar=True)

# print((actuals - torch.nan_to_num((predictions)).abs().mean().item()))
raw_predictions, x = best_tft.predict(test_dataloader, mode="raw", return_x=True, show_progress_bar=True)

torch.save(raw_predictions["prediction"],
           "../../../Datasets/sintetic_dataset/fusion_transformer_result/26000/predicted.pt")

torch.save(actuals,
           "../../../Datasets/sintetic_dataset/fusion_transformer_result/26000/actuals.pt")

# print('\n')
#
# for idx in range(5):  # plot all 5 consumers
#     fig, ax = plt.subplots(figsize=(10, 4))
#     best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True, ax=ax)
