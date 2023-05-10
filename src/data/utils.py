import yaml
from yaml.loader import SafeLoader
import boto3
# Utilities for managing save locations
from collections import defaultdict
from typing import Tuple

from src.data.preprocess import Clip
import os
import pandas as pd
LABELS = [
    'Rumble', 'Gunshot',
]

def get_s3_client_and_resource(config_ref: str = './config/connection_config.yaml'):
    '''
        Gets the s3 client and resource for used for accessing the data
    '''
    with open(config_ref, 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)

    client = boto3.client(
        **config
    )

    s3_resource = boto3.resource(
        **config
    )

    return client, s3_resource


def get_training_df_location(dev=False) -> str:
    df_location = ''
    if dev:
        df_location = './data/dev/training_info.csv'
    else:
        df_location = './data/training_info.csv'
    return df_location


def get_training_data_with_random_clips_location(dev=False) -> str:
    df_location = ''
    if dev:
        df_location = './data/dev/training_info_with_random_clips.csv'
    else:
        df_location = './data/training_info_with_random_clips.csv'
    return df_location


def create_dirs_for_file(file_ref: str):
    dirs = '/'.join(file_ref.split('/', )[:-1])
    os.makedirs(dirs, exist_ok=True)


def get_base_dir_clips(dev=False) -> str:
    location = ''
    if dev:
        location = './data/dev/training'
    else:
        location = './data/training'
    return location


def get_base_dir_resampled_clips(dev=False) -> str:
    location = ''
    if dev:
        location = './data/dev/resampled/training'
    else:
        location = './data/dev/resampled/training'
    return location


def get_metadata(dev=False) -> pd.DataFrame:
    base_dir = get_base_dir_clips(dev)

    metadata_path = os.path.join(base_dir, 'metadata.tsv')
    events_df = pd.read_csv(metadata_path, sep='\t', names=['file_reference', 'start_time', 'end_time', 'label'])

    return events_df

class ClipManager:

    def __init__(self, base_location='./data/training', classes=None, max_clips_in_dir: int = 1000):
        if classes is None:
            classes = LABELS
        self.base_location = base_location
        self.classes = classes

        self.map_counter = defaultdict(lambda: 0)

        self.current_dir = {c: 0 for c in classes}

        self.max_clips_in_dir = max_clips_in_dir

    def get_loc(self, clip: Clip) -> Tuple[str, str]:
        '''
            Gets the location for a given clips, the location depends on the previous calls
            This is pure for easy management of the sounds clips, not the most elegant solution :)
        '''
        target_dir = self.current_dir[clip.label]
        nth_file = self.map_counter[(clip.label, target_dir)]

        # Update counter
        self.map_counter[(clip.label, target_dir)] += 1

        if self.map_counter[(clip.label, target_dir)] > self.max_clips_in_dir:
            self.current_dir[clip.label] += 1
        dir_loc = f'{self.base_location}/{clip.label}/{target_dir}'
        save_loc = f'{dir_loc}/{nth_file}.wav'
        return dir_loc, save_loc
