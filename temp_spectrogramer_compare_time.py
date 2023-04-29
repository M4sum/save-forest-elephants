import generate_spectrograms
import argparse
import numpy as np
from scipy.io import wavfile
from os import path
import os
from time import time
import librosa
import miniaudio
import resampy
import soundfile as sf

TARGET_SR = 8000


"""
Update this file!!
This file is just temporary to generate spectrograms without labels for testing on 
Peter's new data! Assumes we only want to process .wav files.

Let us also create a file at the end with all of the specs createdddd
"""

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

    # Loop through the directory and process the .wav files
    for currentDir in data_dirs:
        # Get the final name of the directory with the spect files
        files_dirs = currentDir.split('/')
        file_dir_name = files_dirs[-2] if files_dirs[-1] == '' else files_dirs[-1]
        # Create the output directory
        spect_dir = path.join(outputDir, file_dir_name)
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
                audio_path = "input_data\\split\\dz_20120206_000000.wav"          # 8kHz file
                # audio_path = "input_data\\split\\kp16_20150416_000000.wav"            # 4kHz file
                try:
                    time_start_load_file = time()
                    # raw_audio_librosa, samplerate_librosa = librosa.load(audio_path, sr=TARGET_SR, mono=True)
                    # raw_audio_librosa, samplerate_librosa = librosa.load(audio_path)
                    # samplerate_wavfile, raw_audio_wavfile = wavfile.read(audio_path)
                    raw_audio, samplerate = librosa.load(audio_path, sr=TARGET_SR, mono=True)
                    if (samplerate < 4000):
                        print("Sample Rate Unexpectadly low!", samplerate)
                    print("File size", raw_audio.shape)
                    print(f'samplerate = {samplerate}')

                    '''
                    raw_audio, sr = sf.read(audio_path, always_2d=True, dtype=np.float32)
                    assert raw_audio.dtype == np.float32, 'Bad sample type: %r' % raw_audio.dtype
                    # waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
                    # waveform = waveform.astype('float32')
                    # Add normalization to deal with differences between wmv and mp4 amplitudes
                    # waveform = waveform / np.max(np.abs(waveform))

                    # Convert to mono and the expected sample rate.
                    if len(raw_audio.shape) > 1:
                        raw_audio = np.mean(raw_audio, axis=1)
                    if sr != TARGET_SR:
                        raw_audio = resampy.resample(raw_audio, sr, TARGET_SR)  # this takes crazy long when wav_data is long
                    print("File size", raw_audio.shape)

                    '''
                    '''
                    target_sampling_rate = TARGET_SR  # the input audio will be resampled a this sampling rate 44100
                    n_channels = 1  # either 1 or 2
                    waveform_duration = 60*60  # in seconds 30
                    # offset = 10  # this means that we read only in the interval [15s, duration of file] 15
                    waveform_generator = miniaudio.stream_file(
                        filename=audio_path,
                        sample_rate=target_sampling_rate,
                        seek_frame=0,  # seek_frame = int(offset * target_sampling_rate),
                        frames_to_read=int(waveform_duration * target_sampling_rate),
                        output_format=miniaudio.SampleFormat.SIGNED16,  # miniaudio.SampleFormat.FLOAT32,
                        nchannels=n_channels)

                    raw_audio = None

                    for i, waveform in enumerate(waveform_generator):
                        # do something with the waveform....
                        # print(f'{i}th waveform size = {len(waveform)}')
                        if i == 0:
                            raw_audio = waveform
                        else:
                            raw_audio = np.concatenate((raw_audio, waveform), axis=0)

                    print("File size", raw_audio.shape)
                    #print(f'samplerate = {samplerate}')
                    samplerate = TARGET_SR
                    '''

                    '''
                    samplerate, raw_audio = wavfile.read(audio_path)
                    if (samplerate < 4000):
                        print ("Sample Rate Unexpectadly low!", samplerate)
                    print ("File size", raw_audio.shape)
                    print(f'samplerate = {samplerate}')
                    '''

                    time_end_load_file = time()
                    # print(f'wavfile took {time_end_load_file - time_start_load_file} to load wav')
                    print(f'librosa took {time_end_load_file - time_start_load_file} to load, shape and resample wav')
                    # print(f'miniaudio took {time_end_load_file - time_start_load_file} to load, shape and resample wav')
                    # print(f'sf and resampy took {time_end_load_file - time_start_load_file} to load, shape and resample wav')

                except:
                    print("FILE Failed", audio_file)
                    # Let us try this for now to see if it stops the failing
                    continue

                spectrogram_info['samplerate'] = samplerate
                start_time = time()
                spectrogram = generate_spectrograms.generate_spectogram(raw_audio, spectrogram_info, data_id)
                end_time = time()
                print(f'Spectrogram generation for one file took {end_time - start_time} seconds')

                # Want to save the corresponding label_file with the spectrogram!!
                np.save(path.join(spect_dir, data_id + "_spec.npy"), spectrogram)
                print ("processed " + data_id)
                spect_files.append(data_id)

        # Save the spect ids
        with open(path.join(spect_dir, 'spects.txt'), 'w') as f:
            for spect_id in spect_files:
                f.write(spect_id + "\n")