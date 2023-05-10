import pandas as pd
from typing import List, Any
from src.data.api import Clip
from src.data.event_logic import event_overlaps_with_window, fit_event_to_window
from src.data.load import load_wav_file
import numpy as np


def aggregate_groups(grouped_df: Any) -> pd.DataFrame:
    new_df = grouped_df.agg({'Begin Time (s)': list})
    new_df['End Time (s)'] = grouped_df.agg({'End Time (s)': list})['End Time (s)']
    new_df['File Offset (s)'] = grouped_df.agg({'File Offset (s)': list})['File Offset (s)']
    new_df['Duration (s)'] = grouped_df.agg({'Duration (s)': list})['Duration (s)']
    new_df.reset_index(inplace=True, )
    return new_df


def preprocess_training_df(training_df: pd.DataFrame, wav_buckets: List[Any]) -> pd.DataFrame:
    '''
        Preprocesses the dataframe such that we can use it for extracting the rumble clips
    '''

    def _get_keys(wav_keys, file_name):
        for key in wav_keys:
            wav_name = key.split('/')[-1]
            if file_name == wav_name:
                return key
        return None

    wav_keys = [summary.key for summary in wav_buckets]
    training_df['aws key'] = training_df['Begin File'].apply(lambda file_name: _get_keys(wav_keys, file_name))
    # Add the duration
    training_df["Duration (s)"] = training_df['End Time (s)'] - training_df['Begin Time (s)']

    training_df = training_df.groupby(['Begin File', 'aws key'])

    result = aggregate_groups(training_df)

    return result


def merge_rumble_and_gunshots(preprocessed_rumble_df: pd.DataFrame,
                              preprocessed_gunshot_df: pd.DataFrame, ) -> pd.DataFrame:
    '''
        Preprocesses the dataframe such that we can use it for extracting the rumble clips
    '''

    cols_to_rename = [
        'Begin Time (s)', 'End Time (s)',
        'File Offset (s)', 'Duration (s)'

    ]
    rumble_rename = {col: f'Rumble {col}' for col in cols_to_rename}
    gunshot_rename = {col: f'Gunshot {col}' for col in cols_to_rename}

    preprocessed_rumble_df = preprocessed_rumble_df.rename(rumble_rename, axis=1)
    preprocessed_gunshot_df = preprocessed_gunshot_df.rename(gunshot_rename, axis=1)

    merged_result = preprocessed_rumble_df.merge(preprocessed_gunshot_df, how='outer', on=['Begin File', 'aws key'])

    return merged_result


def extract_clips(wav_data: np.array, sample_rate: int, row: any, sound_classes: list, clip_length=20.0,
                  start_shift=10.0) -> List[
    Clip]:
    '''
        Extract clips from wav data based on the rows and classes.
        Args:
            wav_data: numpy array containing the wav data
            sample_rate: sample rate of the wav file
            row: dataframe row containing the file offsets and durations for the given classes
            sound_classes: the different sound classes e.g. Rumble, Gunshot, Other
            clip_length: the length of the resulting clips. Any clip that has a longer duration will not be returned
                clip_length is the number of clips that
            start_shift: time (in seconds) by which start of the clips will be offset from the event start
        Returns:
            A list of clips
    '''
    clips = [
    ]
    for sound_class in sound_classes:
        start_times = row[f'{sound_class} File Offset (s)']
        durations = row[f'{sound_class} Duration (s)']

        for idx, (start_time_clip, duration_clip) in enumerate(zip(start_times, durations)):

            # Offset the start time of the clip
            shifted_start_time_clip = max(0, start_time_clip - start_shift)
            shifted_end_time_clip = shifted_start_time_clip + clip_length

            # Check if the clip is within the wav file
            if shifted_end_time_clip < len(wav_data) / sample_rate:
                # Get the events in the clip
                events = get_events(start_times, durations, shifted_start_time_clip, shifted_end_time_clip)

                # Add the clip
                start_index = int(shifted_start_time_clip * sample_rate)
                end_index = int(shifted_end_time_clip * sample_rate)
                clips.append(Clip(wav_data[start_index: end_index], sample_rate, sound_class, events))
    return clips


def get_events(start_time_events, duration_event, clip_start_time, clip_end_time, relative=True):
    events = []

    for start_time_event, duration_event in zip(start_time_events, duration_event):

        end_time_event = start_time_event + duration_event

        if event_overlaps_with_window((start_time_event, end_time_event), clip_start_time, clip_end_time):
            event_start, event_end = fit_event_to_window((start_time_event, end_time_event), clip_start_time, clip_end_time)


            events.append((event_start, event_end))

    # Make relative
    if relative:
        events = [(event_start - clip_start_time, event_end - clip_start_time) for event_start, event_end in events]
    return events




def get_clips_from_file(s3: Any, row: Any, sound_classes) -> List[Clip]:
    '''
        Get the clips from a given file
    '''
    # Load the datafile
    sample_rate, data = load_wav_file(s3, row['aws key'])

    return extract_clips(data, sample_rate, row, sound_classes)


def add_random_clips(df: pd.DataFrame) -> pd.DataFrame:
    def __add_random_intervals(x: any, clip_length=10):
        n_seconds_day = 60 * 60 * 24

        rumble_lengths = len(x['Rumble File Offset (s)'])
        gunshot_lenghts = len(x['Gunshot File Offset (s)'])

        size = max(rumble_lengths, gunshot_lenghts)
        x["Other File Offset (s)"] = list(np.random.uniform(low=0, high=n_seconds_day, size=size))
        x["Other Duration (s)"] = list([clip_length] * size)

        return x

    df = df.apply(__add_random_intervals, axis=1)

    return df
