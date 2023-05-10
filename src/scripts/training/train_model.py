import argparse
import os

import pytorch_lightning as pl
import pandas as pd
import yaml
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from src.callbacks import SaveModelCheckpoint
from src.data.dataset import AudioDataset, collate_fn
from src.data.utils import get_base_dir_clips
from src.models.utils import get_model
from src.data.split import get_wav_file_split

parser = argparse.ArgumentParser(
    description='Creates a single dataframe containing all the information about the sound clips')

parser.add_argument('--dev', action='store_true', help='If we want to use a smaller dev set')

parser.add_argument('--config', type=str, help='Path to the config file',
                    default='./model_configs/simple_transformer.yaml')

args = parser.parse_args()

# Parse the config
config = yaml.safe_load(open(args.config, 'r'))

batch_size = config["training_hyperparameters"]["batch_size"]
max_epochs = config["training_hyperparameters"]["max_epochs"]

# TODO: choose the right path based on the args.dev
base_dir = get_base_dir_clips(dev=args.dev)
meta_data_path = os.path.join(base_dir, 'metadata.tsv')
metadata_df = pd.read_csv(meta_data_path, sep='\t',
                          names=['file_reference', 'start_time', 'end_time', 'label'])

train_wav_files, val_wav_files = get_wav_file_split(args.dev, split_percentage=0.8)

train_dataset = AudioDataset('', metadata_df, wav_files = train_wav_files)
val_dataset = AudioDataset('', metadata_df, wav_files = val_wav_files)

# Here we create a dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = get_model(config)

save_dir = os.path.join(os.getcwd(), 'models')


# Callbacks
callbacks = [
    SaveModelCheckpoint(config, model,  save_dir, monitor='val_loss', mode='min',)
]

wandb_logger = WandbLogger(project='elephants')
trainer = pl.Trainer(max_epochs=max_epochs, logger=wandb_logger, callbacks=callbacks)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
