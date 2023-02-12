import pandas as pd
import pytorch_lightning as pl
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer

# sample_data = pd.DataFrame(
#     dict(
#         time_idx=np.tile(np.arange(6), 3),
#         target=np.array([0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25, 40, 41, 42, 43, 44, 45]),
#         group=np.repeat(np.arange(3), 6),
#         holidays=np.tile(['X', 'Black Friday', 'X', 'Christmas', 'X', 'X'], 3),
#     )
# )
#
# dataset = TimeSeriesDataSet(
#     sample_data,
#     group_ids=["group"],
#     target="target",
#     time_idx="time_idx",
#     max_encoder_length=2,
#     max_prediction_length=3,
#     time_varying_unknown_reals=["target"],
#     static_categoricals=["holidays"],
#     target_normalizer=None
# )
#
#
# dataloader = dataset.to_dataloader(batch_size=5)
#
# #load the first batch
# x, y = next(iter(dataloader))
# print(x['encoder_target'])
# print(x['groups'])
# print('\n')
# print(x['decoder_target'])
dir_dataset = 'Datasets/dataset_fusion_transfomers/LD2011_2014.txt'
data = pd.read_csv(dir_dataset, index_col=0, sep=';', decimal=',')
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)

print(data.head(5))

# down sampling of the information
data = data.resample('1h').mean().replace(0., np.nan)
earliest_time = data.index.min()
#df = data[['MT_002', 'MT_004', 'MT_005', 'MT_006', 'MT_008']]
df = data[['MT_004']]

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

    tmp['days_from_start'] = (date - earliest_time).days
    tmp['date'] = date
    tmp['consumer_id'] = label
    tmp['hour'] = date.hour
    tmp['day'] = date.day
    tmp['day_of_week'] = date.dayofweek
    tmp['month'] = date.month

time_df = tmp

max_prediction_length = 24
max_encoder_length = 7 * 24
training_cutoff = time_df["hours_from_start"].max() - max_prediction_length

training = TimeSeriesDataSet(
    time_df[lambda x: x.hours_from_start <= training_cutoff],
    time_idx="hours_from_start",
    target="power_usage",
    group_ids=["consumer_id"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["consumer_id"],
    time_varying_known_reals=["hours_from_start", "day", "day_of_week", "month", 'hour'],
    time_varying_unknown_reals=['power_usage'],
    target_normalizer=GroupNormalizer(
        groups=["consumer_id"], transformation="softplus"
    ),  # we normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, time_df, predict=True, stop_randomization=True)

# create dataloaders for  our model
batch_size = 64
# if you have a strong GPU, feel free to increase the number of workers
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_forecasting.metrics.quantile import QuantileLoss

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min")

trainer = pl.Trainer(
    max_epochs=45,
    accelerator='gpu',
    devices=1,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback])

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
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
