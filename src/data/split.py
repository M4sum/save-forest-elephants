from typing import Tuple
from src.data.utils import get_base_dir_clips
from src.data.dataset import get_file_names
import functools
import math
def extract_dir_and_nr(ref: str, split_character='/') -> Tuple[int, int]:

    '''
        Extracts the directory and file number from a file reference
    '''
    split = ref.split(split_character)

    dir = int(split[-2])
    nr = int(split[-1].split('.')[0])

    return (dir, nr)


def compare_wav_file_refs(wav_file_ref_1, wav_file_ref_2, split_character='/') -> int:
    return compare(wav_file_ref_1.file_ref, wav_file_ref_2.file_ref, split_character=split_character)
    

def compare(ref_1: str, ref_2: str, split_character='/') -> int:
    '''
        Compares two file references gives the following order:
        1. Directory
        2. File number

        return -1 if ref_1 < ref_2 and 1 if ref_1 > ref_2 and 0 if they are equal
    '''

    (dir_1, nr_1) = extract_dir_and_nr(ref_1, split_character)
    (dir_2, nr_2) = extract_dir_and_nr(ref_2, split_character)

    if dir_1 < dir_2:
        return -1
    elif dir_1 > dir_2:
        return 1
    else:
        if nr_1 < nr_2:
            return -1
        elif nr_1 > nr_2:
            return 1
        else:
            return 0
def split_list(lst, percentage):
    split_index = math.ceil(percentage * len(lst))
    return lst[:split_index], lst[split_index:]
        
def get_wav_file_split(dev=False, split_percentage=0.8):
    
    # First we list all the files
    base_dir = get_base_dir_clips(dev)
    wav_refs, labels = get_file_names(base_dir)

    gunshots = [wav_ref for wav_ref in wav_refs if wav_ref.label == 'Gunshot']
    rumbles = [wav_ref for wav_ref in wav_refs if wav_ref.label == 'Rumble']

    gunshots_sorted = sorted(gunshots, key=functools.cmp_to_key(compare_wav_file_refs))
    rumbles_sorted = sorted(rumbles, key=functools.cmp_to_key(compare_wav_file_refs))
    
    # Split the data
    train_gunshots, val_gunshots = split_list(gunshots_sorted, split_percentage)
    train_rumbles, val_rumbles = split_list(rumbles_sorted, split_percentage)
    
    # Merge the data
    training_wav_files = train_gunshots + train_rumbles
    val_wav_files = val_gunshots + val_rumbles
    
    return training_wav_files, val_wav_files


    