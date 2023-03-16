import pandas as pd
import pytorch_lightning as pl
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from utils.read_dataset import ReadDatasets
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics.quantile import QuantileLoss

configs = {"max_prediction_length": 1000,
            "max_encoder_length": 30,
            "batch_size": 10,
           "transformation": "softplus",
           "trainer": {
               "max_epochs": 2,
               "accelerator": 'gpu',
               "devices": 1,
               "enable_model_summary": True,
               "gradient_clip_val": 0.1
           },
           "config_model": {
               "learning_rate": 0.0001,
               "hidden_size": 512,
               "attention_head_size": 8,
               "dropout": 0.1,
               "hidden_continuous_size": 160,
               "output_size": 7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
               "loss": QuantileLoss(),
               "log_interval": 10,
               "reduce_on_plateau_patience": 4
           },

           }

# dir_data = 'Datasets/sintetic_data/train_compressor_data.h5'

df = pd.read_csv(f"Datasets/saida/ite/payloadITE.csv")

# vector_series = ReadDatasets.read_h5(dir_data)
#
# df = pd.DataFrame({"temp_series": np.array(vector_series).astype(np.float),
#                    "ds": np.arange(len(vector_series)),
#                    "group_id": np.ones(len(vector_series)).astype(np.uint8)})

vector_series = df["temperature"].tolist()

df = pd.DataFrame({"temp_series": np.array(vector_series).astype(np.float),
                   "ds": np.arange(len(vector_series)),
                   "group_id": np.ones(len(vector_series)).astype(np.uint8)})
# df = df.dropna()
df.index = pd.to_datetime(df.index)
earliest_time = df.index.min()
# df.sort_index(inplace=True)

print(df.head(8))

# down sampling of the information

print("Define dataset")

training = TimeSeriesDataSet(
    df,
    time_idx="ds",
    target="temp_series",
    group_ids=["group_id"],
    min_encoder_length=configs["max_encoder_length"] // 2,
    max_encoder_length=configs["max_encoder_length"],
    min_prediction_length=1,
    max_prediction_length=configs["max_prediction_length"],
    static_categoricals=[],
    time_varying_known_reals=["ds"],
    time_varying_unknown_reals=["temp_series"],
    target_normalizer=GroupNormalizer(
        groups=["group_id"], transformation=configs["transformation"]
    ),  # we normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

# create dataloaders for  our model
# if you have a strong GPU, feel free to increase the number of workers
train_dataloader = training.to_dataloader(train=True,
                                          batch_size=configs["batch_size"],
                                          num_workers=0)

val_dataloader = validation.to_dataloader(train=False,
                                          batch_size=configs["batch_size"],
                                          num_workers=0)


trainer = pl.Trainer(
    max_epochs=configs["trainer"]["max_epochs"],
    accelerator=configs["trainer"]["accelerator"],
    devices=configs["trainer"]["devices"],
    enable_model_summary=configs["trainer"]["enable_model_summary"],
    gradient_clip_val=configs["trainer"]["gradient_clip_val"])

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=configs["config_model"]["learning_rate"],
    hidden_size=configs["config_model"]["hidden_size"],
    attention_head_size=configs["config_model"]["attention_head_size"],
    dropout=configs["config_model"]["dropout"],
    hidden_continuous_size=configs["config_model"]["hidden_continuous_size"],
    output_size=configs["config_model"]["output_size"],  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=configs["config_model"]["log_interval"],
    reduce_on_plateau_patience=configs["config_model"]["reduce_on_plateau_patience"])

trainer.fit(
    tft,
    train_dataloaders=train_dataloader)

best_model_path = trainer.checkpoint_callback.best_model_path
print(best_model_path)

