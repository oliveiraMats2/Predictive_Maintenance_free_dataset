import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from utils.read_dataset import ReadDatasets
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics.quantile import QuantileLoss
from configs import *


dir_data = "../../../Datasets/sintetic_dataset/train_compressor_data.h5"
dir_model = "../../../models_h5/"

vector_series = ReadDatasets.read_h5(dir_data)

df = pd.DataFrame({"temp_series": np.array(vector_series).astype(np.float),
                   "ds": np.arange(len(vector_series)),
                   "group_id": np.ones(len(vector_series)).astype(np.uint8)})

df.index = pd.to_datetime(df.index)
earliest_time = df.index.min()

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

train_dataloader = training.to_dataloader(train=True,
                                          batch_size=configs["batch_size"],
                                          num_workers=8)

val_dataloader = validation.to_dataloader(train=False,
                                          batch_size=configs["batch_size"],
                                          num_workers=8)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")

trainer = pl.Trainer(
    max_epochs=configs["trainer"]["max_epochs"],
    accelerator=configs["trainer"]["accelerator"],
    devices=configs["trainer"]["devices"],
    enable_model_summary=configs["trainer"]["enable_model_summary"],
    gradient_clip_val=configs["trainer"]["gradient_clip_val"],
    callbacks=[early_stop_callback],
    default_root_dir=dir_model)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=configs["config_model"]["learning_rate"],
    hidden_size=configs["config_model"]["hidden_size"],
    attention_head_size=configs["config_model"]["attention_head_size"],
    dropout=configs["config_model"]["dropout"],
    hidden_continuous_size=configs["config_model"]["hidden_continuous_size"],
    output_size=configs["config_model"]["output_size"],
    loss=QuantileLoss(),
    log_interval=configs["config_model"]["log_interval"],
    reduce_on_plateau_patience=configs["config_model"]["reduce_on_plateau_patience"])

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader)

best_model_path = trainer.checkpoint_callback.best_model_path
print(best_model_path)

