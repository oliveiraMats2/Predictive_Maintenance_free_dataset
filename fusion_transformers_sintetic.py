import pandas as pd
import pytorch_lightning as pl
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from utils.read_dataset import ReadDatasets

dir_data = 'Datasets/sintetic_data/train_compressor_data.h5'

vector_series = ReadDatasets.read_h5(dir_data)

df = pd.DataFrame({"temp_series": np.array(vector_series).astype(np.float),
                   "ds": np.arange(len(vector_series)),
                   "group_id": np.ones(len(vector_series)).astype(np.uint8)})

df.index = pd.to_datetime(df.index)
earliest_time = df.index.min()
df.sort_index(inplace=True)

print(df.head(20))

# down sampling of the information

df_list = []

for label in df:
    ts = df[label]

    start_date = min(ts.fillna(method='ffill').dropna().index)
    end_date = max(ts.fillna(method='bfill').dropna().index)

    active_range = (ts.index >= start_date) & (ts.index <= end_date)
    ts = ts[active_range].fillna(0.)

    tmp = pd.DataFrame({'power_usage': ts})
    date = tmp.index

    tmp['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (date - earliest_time).days * 24
    tmp['hours_from_start'] = tmp['hours_from_start'].astype('int')

    tmp['date'] = date
    tmp['consumer_id'] = label
    tmp['hour'] = date.hour
    tmp['day'] = date.day
    tmp['day_of_week'] = date.dayofweek
    tmp['month'] = date.month

time_df = tmp

max_prediction_length = 24
max_encoder_length = 7 * 24
training_cutoff = abs(time_df["hours_from_start"].max() - max_prediction_length)

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
    time_varying_known_reals=[],
    time_varying_unknown_reals=[],
    target_normalizer=GroupNormalizer(
        groups=["group_id"], transformation="softplus"
    ),  # we normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

# create dataloaders for  our model
batch_size = 64
# if you have a strong GPU, feel free to increase the number of workers
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics.quantile import QuantileLoss

trainer = pl.Trainer(
    max_epochs=45,
    accelerator='gpu',
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.1)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=160,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=160,
    output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4)

trainer.fit(
    tft,
    train_dataloaders=train_dataloader)
