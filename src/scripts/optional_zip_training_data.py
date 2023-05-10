# This zips the training data to have an easy way to share the results. 

import shutil

import os

dirs = [
    'Gunshot',
    'Other',
    'Rumble'
    ]

for d in dirs:
    path = f'./data/training/{d}'
    
    subdirs = os.listdir(path)
    
    for subdir in subdirs:
        sub_path = f'{path}/{subdir}'
        print(sub_path)
        shutil.make_archive(f'./data/training/{d}_{subdir}', 'zip', sub_path, )