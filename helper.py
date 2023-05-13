import asyncio
import numpy as np
import os

def load_spectrograms(filename, file_id):
    with open(filename, 'rb') as f:
        spectrogram = np.load(f)
    return spectrogram, file_id

