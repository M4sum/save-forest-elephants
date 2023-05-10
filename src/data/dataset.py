from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch.nn
from scipy.io import wavfile
import os

from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram, MelScale

from src.data.api import WavFileRef, Event
from src.data.event_logic import event_overlaps_with_window, fit_event_to_window
from src.data.utils import LABELS


def file_ref_to_name(ref: str) -> str:
    return '_'.join(ref.split('/')[-3:])


def df_to_event_matrix(df: pd.DataFrame, length_in_seconds=10.0, event_bins=10,
                       starting_sec=0.0, labels=LABELS) -> np.ndarray:
    '''
        Converts a dataframe to an event matrix
    '''
    event_ar = np.zeros((len(labels), event_bins))

    events_within_window, overlapping = get_events_within_window(df, length_in_seconds, starting_sec)

    # Keep the labels that are from overlapping clips
    labels_of_events = df['label'][overlapping]
    label_indices = [labels.index(label) for label in labels_of_events]

    for i, (start_time, end_time) in zip(label_indices, events_within_window):
        event_ar[i,
        int(start_time / length_in_seconds * event_bins): int(end_time / length_in_seconds * event_bins)] = 1

    return event_ar


def get_events_within_window(df: pd.DataFrame, length_in_seconds=10.0, starting_sec=0.0) -> Tuple[
    List[Event], List[bool]]:
    '''
        Returns the events that are within the window and a boolean array of which events are in the window
        Also makes the events relative to the window start

    '''
    starting_times = np.floor(df["start_time"])
    ending_times = np.ceil(df["end_time"]) + 1  # + 1 to make it inclusive

    # Make the times relative
    starting_times -= starting_sec
    ending_times -= starting_sec

    # Check if they are contained in the window
    start_window = 0
    end_window = length_in_seconds

    overlapping = [event_overlaps_with_window((s, e), start_window, end_window) for s, e in
                   zip(starting_times, ending_times)]

    starting_times = starting_times[overlapping]
    ending_times = ending_times[overlapping]

    events_within_window = [fit_event_to_window(event, start_window, end_window) for event in
                            zip(starting_times, ending_times)]

    return events_within_window, overlapping


def get_file_names(base_dir, labels: List[str] = LABELS) -> Tuple[List[WavFileRef], List[str]]:
    '''
        Generates a list of all the wav files
    '''
    wav_files = []

    class_dirs = [f'{base_dir}/{label}' for label in labels]

    for sound_class, class_dir in zip(labels, class_dirs):
        subdirs = [d for d in os.listdir(class_dir)]

        for subdir in subdirs:
            subdir_ref = f'{class_dir}/{subdir}'

            wav_files += [
                WavFileRef(f'{subdir_ref}/{file}', sound_class) for file in os.listdir(subdir_ref)
            ]
    return wav_files, labels


class AudioDataset(Dataset):

    def __init__(self, base_dir, metadata_df, batch_size=32, event_bins=10,
                 max_start_offset=10.0, desired_clip_length=10.0, shuffle=True,
                wav_files =None
                ):
        'Initialization'
        self.base_dir = base_dir
        self.batch_size = batch_size
        
        if wav_files:
            self.wav_files = wav_files
        else:
            print('searching for wav_files')
            self.wav_files, _ = get_file_names(base_dir)
        self.metadata_df = metadata_df
        self.event_bins = event_bins
        # Add file names
        self.metadata_df['file_name'] = metadata_df['file_reference'].apply(file_ref_to_name, )

        self.shuffle = shuffle
        self.indices = np.arange(len(self.wav_files))

        # By how much we can offset the starting point of a clip
        # e.g. if saved clips are 15s long and we want to extract 10s, this can't be higher than 5
        self.max_start_offset = max_start_offset
        self.desired_clip_length = desired_clip_length

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, index) -> Tuple[float, np.ndarray, np.ndarray]:
        'Generate one batch of data'
        wav_file = self.wav_files[index]

        # Generate data
        sample_rate, data, y = self.__data_generation(wav_file)

        return sample_rate, data, y

    def __data_generation(self, wav_file):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        sample_rate, data = wavfile.read(wav_file.file_ref)

        file_name = file_ref_to_name(wav_file.file_ref)

        events = self.metadata_df[self.metadata_df['file_name'] == file_name]

        # Sample starting time of the clip
        starting_sec = np.random.uniform(low=0.0, high=self.max_start_offset, size=(1,))[0]

        events = df_to_event_matrix(events, length_in_seconds=self.desired_clip_length,
                                    event_bins=self.event_bins, starting_sec=starting_sec)
        starting_index = int(starting_sec * sample_rate)
        end_index = int((starting_sec + self.desired_clip_length) * sample_rate)
        data = data[starting_index:end_index]

        return sample_rate, data, events


class MelSpectogramPipeline(torch.nn.Module):
    def __init__(self, sample_rate=4000, n_fft=1024, n_mels=128, pad_to=80, pad=True, do_normalize=True):
        super(MelSpectogramPipeline, self).__init__()
        self.spec = Spectrogram(n_fft=n_fft, power=2)
        self.mel_scale = MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=n_fft // 2 + 1
        )

        self.pad_to = pad_to
        self.pad = pad

        self.do_normalize = do_normalize

        # Taken from: https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer#transformers.ASTFeatureExtractor
        # Might use other values depending on the dataset
        self.mean = 15
        self.std = 1.5

    def forward(self, data):
        specs = self.spec(data)
        mels = torch.log(self.mel_scale(specs))

        if self.do_normalize:
            mels = self.normalize(mels)

        # Pad to 80
        # Pad zeros to the end of the sequence
        if self.pad and mels.shape[2] < self.pad_to:
            zero_padding = torch.zeros(
                (mels.shape[0], mels.shape[1], self.pad_to - mels.shape[2])).to(
                mels.device)
            mels = torch.cat((mels, zero_padding), dim=2)

        return mels

    def normalize(self, mel_scale):
        return (mel_scale - self.mean) / (self.std * 2)


def pad(audio_clips: List[np.ndarray]) -> List[np.ndarray]:
    max_length = max([len(clip) for clip in audio_clips])
    padded_audio_clips = [np.pad(clip, (0, max_length - len(clip)), 'constant') for clip in audio_clips]

    return padded_audio_clips


def collate_fn(items: List[Tuple[float, np.ndarray, np.ndarray]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    '''
        Collates the items into a batch
    '''

    # Pad items to the same length
    audio_clips = [item[1] for item in items]

    # print the shape of the audio clips
    labels = [item[2] for item in items]

    x = torch.FloatTensor(np.array(audio_clips, dtype=np.int16))
    y = torch.FloatTensor(np.array(labels))

    return x, y
