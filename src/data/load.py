import boto3
import io
from scipy.io import wavfile
import numpy as np
from typing import List, Tuple, Any
import pandas as pd
import os

# Some helpfull typings

bucket_list = List[Any] # Hard to do type checking with s3


COLUMNS_TO_KEEP = [
    'Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)',
       'Begin Path', 'Begin File', 'File Offset (s)',
]



def get_all_files_with_extension(buckets: bucket_list, extension: str='.txt'):
    '''
        A helper function to get files with a certain extention
    '''
    return [bucket for bucket in buckets if bucket.key[-len(extension):] == extension]



def get_buckets(s3_bucket: Any) -> Tuple[bucket_list, bucket_list]:
    
    
    all_buckets = [my_bucket_object for my_bucket_object in s3_bucket.objects.all()]
    text_buckets = get_all_files_with_extension(all_buckets)
    wav_buckets = get_all_files_with_extension(all_buckets, '.wav')
    return text_buckets, wav_buckets

def load_wav_file(s3: Any, aws_key: str, bucket_name: str = 'data-ai-for-forest-elephants') -> Tuple[int, np.array]:
    s3_object = s3.Object(
        bucket_name=bucket_name,
        key=aws_key,
    ).get()
    audio_data = s3_object['Body'].read()
    sample_rate, data = wavfile.read(io.BytesIO(audio_data))
    
    return sample_rate, data


def load_rumble_clip_data(s3_bucket: Any) -> pd.DataFrame:
    rumble_training_texts = [
    'Rumble/Training/Clearings/rumble_clearing_00-24hr_56days.txt', 
    #'Rumble/Training/pnnn/nn_ele_hb_00-24hr_TrainingSet.txt',
    #'Rumble/Training/Overlap/Select_HH_forOverlap_simple.txt', # This has something to do with overlapping files
    # 'Rumble/Training/Spectrograms/wavs/spects.txt', # THis is a weird file, don't know the purpose :( 
    ]

    dfs = {}

    # Create a data directory if it doesn't already exist
    if not os.path.exists("data"):
        os.makedirs("data")

    for text_ref in rumble_training_texts:
        name = text_ref.split('/')[-1]
        save_name = 'data/' + name

        s3_bucket.download_file(text_ref, save_name)
        dfs[name] = pd.read_csv(save_name, sep="\t")



    rumble_training_df = pd.concat([v[COLUMNS_TO_KEEP] for _, v in dfs.items()])
    return rumble_training_df

def load_gunshot_clip_data(s3_bucket: Any) -> pd.DataFrame:
    gunshot_testing_texts = [
     'Gunshot/Training/ecoguns/Guns_Training_ecoGuns_SST.txt',
     #'Gunshot/Training/pnnn_dep1-7/nn_Grid50_guns_dep1-7_Guns_Training.txt',
    ]

    # Create a data directory if it doesn't already exist
    if not os.path.exists("data"):
        os.makedirs("data")

    gunshot_dfs = {}
    for text_ref in gunshot_testing_texts:
        name = text_ref.split('/')[-1]
        save_name = 'data/' + name

        s3_bucket.download_file(text_ref, save_name)
        gunshot_dfs[name] = pd.read_csv(save_name, sep="\t")

    rename_columns = {
        'view': 'View', 
        'channel': 'Channel', 
        'begin time': 'Begin Time (s)' , 
        'end time': 'End Time (s)', 
        'begin path': 'Begin Path', 

        
    }
    gunshot_dfs['Guns_Training_ecoGuns_SST.txt'] = gunshot_dfs['Guns_Training_ecoGuns_SST.txt'] .rename(rename_columns, axis=1)
    gunshot_training_df = pd.concat([v[COLUMNS_TO_KEEP] for _, v in gunshot_dfs.items()])
    return gunshot_training_df