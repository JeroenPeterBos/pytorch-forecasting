#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings

warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
from datetime import datetime

import torch
torch.cuda.empty_cache()
print('You: \"Torch, is cuda available?"')
print(f'Torch: \"{"Yes" if torch.cuda.is_available() else "No"}!\"')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss, SMAPE


# In[3]:




# In[4]:


data_path = '/home/jeroen/Leap/data-capacity-forecast/data/2021-04-20_advanced_preprocessing_2021-01-01_small/processed/advanced/intervals/'


# # Temporal Fusion Transformer
# ### Let's give it a go
# 
# In this notebook we load the interval data and try to train the model purely on the interval data.
# No forecasting performance yet, purely running it on the data we have in its most vanilla form.

# ### Loading the data

# In[5]:


data = pd.read_parquet(data_path)
data

# Filter to 3 meters
meters = data['meter_id'].drop_duplicates().tolist()[:3]
data = data[data['meter_id'].isin(meters)]

meters

# In[6]:


data.info(memory_usage="deep")
data['meter_id'].drop_duplicates().count()


# ### Preparing the data

# In[7]:


df = data.sort_values(['meter_id', 'local_interval_start']).copy()

df['is_commercial'] = df['is_commercial'].apply(lambda b: 'Yes' if b else 'No').astype('category')
df['is_event_interval'] = df['is_event_interval'].apply(lambda b: 'Yes' if b else 'No').astype('category')
df['is_business_day'] = df['is_business_day'].apply(lambda b: 'Yes' if b else 'No').astype('category')

df['time_idx'] = ((df['local_interval_start'] - datetime(1970, 1, 1)).dt.total_seconds() / (60 * 15)).round(0).astype(int)
df['time_idx'] -= df['time_idx'].min()

df['time_in_day'] = df['local_interval_start'].dt.hour * 4 + df['local_interval_start'].dt.minute // 15
df['day_in_week'] = df['local_interval_start'].dt.dayofweek
df['month_in_year'] = df['local_interval_start'].dt.month

# Put target in data to see how well the model works then
df['target_copy'] = df['generation_kw'] + np.random.normal(scale=0.1, size=df['generation_kw'].shape)

df = df.sort_values(['meter_id', 'local_interval_start']).reset_index(drop=True)

df


# In[8]:


df.info(memory_usage="deep")
df['meter_id'].drop_duplicates().count()


# In[9]:


max_prediction_length = 4*4
max_encoder_length = 4
training_cutoff = df["time_idx"].max() - max_prediction_length


# In[10]:


training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx='time_idx',
    target='generation_kw',
    group_ids=['meter_id'],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=['is_commercial'],
    static_reals=[],
    time_varying_known_categoricals=['is_business_day', 'is_event_interval'],
    time_varying_known_reals=['time_idx', 'time_in_day', 'day_in_week', 'month_in_year', 'target_copy'],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=['load_kw', 'baseline_kw', 'generation_kw'],
    target_normalizer=GroupNormalizer(
        groups=["meter_id"], transformation="softplus"
    ),  # use softplus and normalize by group
    lags={
            'load_kw': [(24 * 4), (24 * 4 * 7), (24 * 4 * 365)],
    },
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True
)


# In[11]:

x = training[0]


validation = TimeSeriesDataSet.from_dataset(
    training,
    df,
    predict=True,
    stop_randomization=True,
)


# In[12]:


batch_size = 32
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)


# ### Baseline model

# In[13]:


actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
(actuals - baseline_predictions).abs().mean().item()


# In[14]:


pl.seed_everything(42)
trainer = pl.Trainer(
    gpus=1,
    gradient_clip_val=0.1,
#     precision=16
)


# In[15]:


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.3,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss(),
    reduce_on_plateau_patience=4
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


# In[16]:


res = trainer.tuner.lr_find(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)

print(f"Suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()


# # Training the model

# In[17]:


early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4,
    patience=10,
    verbose=False,
    mode='min'
)
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger('lightning_logs')


# In[18]:


trainer = pl.Trainer(
    max_epochs=30,
    gpus=1,
    weights_summary='top',
    gradient_clip_val=0.1,
    limit_train_batches=30,
#    fast_dev_run=True,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)


# In[19]:


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.01,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.3,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


# In[20]:


trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader
)
