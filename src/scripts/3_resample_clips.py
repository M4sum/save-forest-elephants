import argparse

import numpy as np
import pandas as pd
import os

import torch
import librosa
import soundfile as sf
from torchaudio.transforms import Resample
from tqdm import tqdm

from src.data.dataset import get_file_names
from src.data.utils import ClipManager, get_base_dir_clips, get_base_dir_resampled_clips

parser = argparse.ArgumentParser(
    description='Resamples the clips ')

parser.add_argument('--dev',
                    action='store_true', help='If we want to use a smaller dev set')

parser.add_argument('--target_freq',
                    type=int, default=4000, help='The target frequency for the resampling')

parser.add_argument('--target_clip_length',
                    type=int, default=20, help='The target clip length in seconds')
args = parser.parse_args()

# Very naive implementation for resampling (no parallel processing)

resampled_dir = get_base_dir_resampled_clips(args.dev)

original_dir = get_base_dir_clips(args.dev)

clip_manager = ClipManager(base_location=resampled_dir)

# Load the clips
wav_files, _ = get_file_names(original_dir)

# Load the tsv file
for i, wav_file_ref in tqdm(enumerate(wav_files), total=len(wav_files)):

    # Load the input wav file
    data, sample_rate = librosa.load(wav_file_ref.file_ref, sr=None)

    # Resample the input signal to 4kHz
    data_resampled = librosa.resample(data, orig_sr=sample_rate, target_sr=args.target_freq)

    # Make sure it is the right length
    if len(data) > args.target_freq * args.target_clip_length:
        print('WARNING, AUDIO FILE TOO long', wav_file_ref.file_ref)
        data = data[:args.target_freq * args.target_clip_length]

    if len(data) < args.target_freq * args.target_clip_length:
        # Pad with zeros
        print('WARNING, AUDIO FILE TOO short', wav_file_ref.file_ref)
        print('quick fix: remove this file from your dataset')

    sf.write(wav_file_ref.file_ref, data_resampled, args.target_freq)
