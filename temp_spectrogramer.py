import generate_spectrograms
import argparse
import numpy as np
from scipy.io import wavfile
from os import path
import os
from time import time
import librosa
# import miniaudio
# import resampy
import soundfile as sf
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from matplotlib import mlab as ml

TARGET_SR = 4000

"""
Update this file!!
This file is just temporary to generate spectrograms without labels for testing on 
Peter's new data! Assumes we only want to process .wav files.

Let us also create a file at the end with all of the specs createdddd
"""

def audio_slice_to_spectogram(audio_chunk, idx, nfft, sr, noverlap, pad, max_f):
    [spectrum_chunk, freqs_chunk, t_chunk] = ml.specgram(audio_chunk,
                                             NFFT=nfft, Fs=sr, noverlap=noverlap,
                                             window=ml.window_hanning, pad_to=pad)

    # Cutout the high frequencies that are not of interest
    spectrum_chunk = spectrum_chunk[(freqs_chunk <= max_f)]

    return spectrum_chunk, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # For use on quatro
    parser.add_argument('--data_dirs', dest='data_dirs', nargs='+', type=str,
        help='Provide the data_dirs with the files that you want to be processed')
    parser.add_argument('--out', dest='outputDir', default='/home/data/elephants/rawdata/Spectrograms/',
         help='The output directory')

    parser.add_argument('--NFFT', type=int, default=4096, help='Window size used for creating spectrograms') 
    parser.add_argument('--hop', type=int, default=800, help='Hop size used for creating spectrograms')
    parser.add_argument('--window', type=int, default=256, 
        help='Deterimes the window size in frames of the resulting spectrogram') # Default corresponds to 21s
    parser.add_argument('--max_f', dest='max_freq', type=int, default=150, help='Deterimes the maximum frequency band')
    parser.add_argument('--pad', dest='pad_to', type=int, default=4096, 
        help='Deterimes the padded window size that we want to give a particular grid spacing (i.e. 1.95hz')


    args = parser.parse_args()
    data_dirs = args.data_dirs
    outputDir = args.outputDir
    # Make sure this exists
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    spectrogram_info = {'NFFT': args.NFFT,
                        'hop': args.hop,
                        'max_freq': args.max_freq,
                        'window': args.window,
                        'pad_to': args.pad_to}
    NFFT = args.NFFT
    hop = args.hop
    max_freq = args.max_freq
    window = args.window
    pad_to = args.pad_to

    # Loop through the directory and process the .wav files
    for currentDir in data_dirs:
        # Get the final name of the directory with the spect files
        current_path, tail = os.path.split(currentDir)
        # Create the output directory
        spect_dir = path.join(outputDir, tail)
        if not path.exists(spect_dir):
            os.mkdir(spect_dir)

        # Create and save a file of the names of each of the spectrograms produced
        spect_files = []
        for(dirpath, dirnames, filenames) in os.walk(currentDir):
            # Iterate through the .wav spectrogram files to generate them!
            for audio_file in filenames:
                tags = audio_file.split('.')
                data_id = tags[0]
                file_type = tags[1]

                if (file_type not in ['wav']):
                    continue

                # We need to read the audio file so that we can use generate_spectrogram
                audio_path = path.join(dirpath, audio_file)
                # audio_path = "input_data\\split\\dz_20120206_000000.wav"          # 8kHz 24h file
                # audio_path = "input_data\\split\\kp16_20150416_000000.wav"            # 4kHz 24h file
                # audio_path = "input_data\\split\\tree3_20180517_095900.wav"          # 14h file
                try:
                    time_start_load_file = time()
                    raw_audio, samplerate = librosa.load(audio_path, sr=TARGET_SR, mono=True)
                    if (samplerate != 4000):
                        print("Sample Rate Error!", samplerate)
                    print("File size", raw_audio.shape)
                    print(f'samplerate = {samplerate}')
                    time_end_load_file = time()
                    print(f'librosa took {time_end_load_file - time_start_load_file} to load, shape and resample wav')

                except:
                    print("FILE Failed", audio_file)
                    # Let us try this for now to see if it stops the failing
                    continue

                spectrogram_info['samplerate'] = samplerate
                # spectrogram = generate_spectrograms.generate_spectogram(raw_audio, spectrogram_info, data_id)     # Replace this with spectogram generation that is parallelized
                start_time = time()

                # divide the audio into equal parts, each part will be processed in a separate process
                # Remember that we want to start as if we are doing one continuous sliding window
                chunk_size = 1000
                len_chunk = (chunk_size - 1) * hop + NFFT
                # Calculate number of chunks i.e ((total_length - window_size) / jump_size) +1
                # num_chunks = np.ceil((raw_audio.shape[0] - len_chunk) / (len_chunk - NFFT + hop)) #+ 1
                num_chunks = np.ceil((raw_audio.shape[0]) / (len_chunk - NFFT + hop))
                # total number of windows in sliding window: np.ceil((total_length - window_size) / hop_size) -1
                processes = []
                with ProcessPoolExecutor() as executor:
                    for audio_idx in range(0, raw_audio.shape[0], len_chunk - NFFT + hop):
                        print(f'Chunk number {audio_idx}: {data_id}')
                        # pdb.set_trace()
                        audio_slice = raw_audio[audio_idx: audio_idx + len_chunk]
                        processes.append(executor.submit(audio_slice_to_spectogram, audio_slice, audio_idx,
                                                         NFFT, samplerate, NFFT-hop, pad_to, max_freq))

                    # wait for all processes to finish and collect spectrogram chunks
                    i = 0
                    final_spec = None
                    process_spec = None
                    first_chunk = None
                    last_chunk = None
                    for process in as_completed(processes):
                        # pdb.set_trace()
                        spect_chunk, spect_idx = process.result()
                        idx_number = int(spect_idx/len_chunk)       # results not returned in correct order so need to use this
                        if i == 0:
                            first_chunk = spect_chunk
                            process_spec = np.zeros((spect_chunk.shape[0], int(num_chunks * chunk_size)))
                        # print(f'size of spect_chunk = {spect_chunk.shape[0]}')
                        if spect_chunk.shape[1] != first_chunk.shape[1]:
                            last_chunk = spect_chunk
                            process_spec[:, idx_number * chunk_size:(idx_number * chunk_size) + last_chunk.shape[1]] = spect_chunk
                        else:
                            process_spec[:, idx_number * chunk_size:(idx_number + 1) * chunk_size] = spect_chunk
                        i +=1
                        # final_spec[idx_number*chunk_spec_dim: (idx_number+1)*chunk_spec_dim, :] += spect_chunk

                final_spec = np.zeros((spect_chunk.shape[0], int(((num_chunks-1) * chunk_size) + last_chunk.shape[1])))
                final_spec = final_spec.T

                print("Finished making one 24 hour spectogram")
                end_time = time()
                print(f'Spectrogram generation for one file took {end_time-start_time} seconds')

                # Want to save the corresponding label_file with the spectrogram!!
                np.save(path.join(spect_dir, data_id + "_spec.npy"), final_spec)
                print("processed " + data_id)
                spect_files.append(data_id)

        # Save the spect ids
        with open(path.join(spect_dir, 'spects.txt'), 'w') as f:
            for spect_id in spect_files:
                f.write(spect_id + "\n")