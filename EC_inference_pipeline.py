import os
import asyncio
from aiomultiprocess import Pool
import sys
import sounddevice as sd
import soundfile as sf
from math import ceil
import numpy as np
import resampy
from time import time
import librosa

AUDIO_FILE_PATH = "C:\\Users\\estin\\PycharmProjects\\ElephantsEfficientProcessingTeam\\finding_manny\\input_data\\tree3_20180517_095900.wav"
#AUDIO_FILE_PATH = "C:\\Users\\estin\\PycharmProjects\\ElephantsEfficientProcessingTeam\\finding_manny\\input_data\\testmp3.mp3"

BLOCK_SIZE = 2496
FIXED_SR = 4000
current_frame = 0

# settings for spectrograms
NFFT = 4096                 # Window size used for creating spectrograms
HOP = 800                   # Hop size used for creating spectrograms
WINDOW = 256                # Deterimes the window size in frames of the resulting spectrogram (Default corresponds to 21s)
MAX_FREQ = 150              # Deterimes the maximum frequency band
PAD_TO = 4096               # Deterimes the padded window size that we want to give a particular grid spacing (i.e. 1.95hz'


async def generate_spectrogram_of_chunk(idx, processor_chunk):
    # print(f'generate_spectrogram_of_chunk in {os.getpid()} with chunk {idx}')
    '''
    nfft = NFFT
    hop = HOP
    max_freq = MAX_FREQ
    window = WINDOW
    pad_to = PAD_TO
    samplerate = FIXED_SR
    chunk_size = 1000
    # Generate the spectogram in chunks of 1000 frames.
    len_chunk = (chunk_size - 1) * hop + NFFT       # = 803296

    [spectrum, freqs, t] = ml.specgram(raw_audio[start_chunk: start_chunk + len_chunk],
                                       NFFT=NFFT, Fs=samplerate, noverlap=(NFFT - hop), window=ml.window_hanning,
                                       pad_to=pad_to)
    # Cutout the high frequencies that are not of interest
    spectrum = spectrum[(freqs <= max_freq)]

    final_spec = None
    start_chunk = 0
    i = 0
    while start_chunk + len_chunk < raw_audio.shape[0]:
        if (i % 100 == 0):
            print("Chunk number " + str(i) + ": " + id)


        if i == 0:
            final_spec = spectrum
        else:
            final_spec = np.concatenate((final_spec, spectrum), axis=1)

        # Remember that we want to start as if we are doing one continuous sliding window
        start_chunk += len_chunk - NFFT + hop               # = 800000
        i += 1

    # Do one final chunk for whatever remains at the end
    [spectrum, freqs, t] = ml.specgram(raw_audio[start_chunk: start_chunk + len_chunk],
                                       NFFT=NFFT, Fs=samplerate, noverlap=(NFFT - hop), window=ml.window_hanning,
                                       pad_to=pad_to)
    # Cutout the high frequencies that are not of interest
    spectrum = spectrum[(freqs <= max_freq)]
    final_spec = np.concatenate((final_spec, spectrum), axis=1)

    print("Finished making one 24 hour spectogram")
    '''
    await asyncio.sleep(1)      # simulates generating spectrogram
    print(f'generate_spectrogram_of_chunk in {os.getpid()} woke up with chunk {idx}')



async def predict_spec_sliding_window(chunkID, chunk):
    # print(f'predict_spec_sliding_window in {os.getpid()} with chunk {chunkID}')
    await asyncio.sleep(1)      # simulates prediction
    print(f'predict_spec_sliding_window in {os.getpid()} woke up with chunk {chunkID}')

async def process_batch(processor_chunk):
    # print(f'process_batch function in process {os.getpid()} with args {input}')
    # Need to split processor_chunk into applicable chunks and can run each chunk concurrently -> I think use async for loop?

    idx = 0
    async for outdata, status in data_stream_generator(processor_chunk):
        print(f'processing {idx}th chunk in data_stream_generatorsize = {len(outdata)}')
        await generate_spectrogram_of_chunk(idx, outdata)  # simulates generating spectrogram
        await predict_spec_sliding_window(idx, outdata)  # simulates prediction
        idx += 1

    return idx


async def data_stream_generator(indata_processor_chunk, blocksize=BLOCK_SIZE, *, channels=1, dtype='float32',
                           pre_fill_blocks=10, **kwargs):
    """Generator that yields blocks of output data from larger chunk passed (from wav file)
     as NumPy arrays. The output blocks are uninitialized and have to be filled with
        appropriate audio signals.
        """
    assert blocksize != 0
    q_out = asyncio.Queue()
    loop = asyncio.get_event_loop()

    print(f'indata_processor_chunk.shape = {indata_processor_chunk.shape}')

    def callback(indata, frames, time, status):
        global current_frame
        if status:
            print(status)
        chunksize = min(len(indata_processor_chunk) - current_frame, frames)
        indata[:chunksize] = indata_processor_chunk[current_frame:current_frame + chunksize]
        if chunksize < frames:
            indata[chunksize:] = 0
            raise sd.CallbackStop()
        current_frame += chunksize
        loop.call_soon_threadsafe(q_out.put_nowait, (indata.copy(), status))
        print(f'outputstream_generator callback executed with frames {frames}')

    stream = sd.OutputStream(samplerate=FIXED_SR, device=sd.default.device, blocksize=BLOCK_SIZE, channels=indata_processor_chunk.shape[1], callback=callback)

    with stream:
        while True:
            outdata, status = await q_out.get()
            yield outdata, status
            if q_out.empty():
                stream.stop()  # ???
                break



async def main(datadir):
    # Need to loop for all files in the directory
    print(f'argument passed to async main {datadir}')
    print(f'number of available cores = {os.cpu_count()}')
    # Use input arguments for dataloader to get raw audio batches
    # must return in batches with asynchronous iterator to split batch into chunks - create class AudioIterator
    '''
    dataloader = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
                  (23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (33, 34), (35, 36), (37, 38), (39, 40), (41, 42),
                  (43, 44), (45, 46), (47, 48)]
    '''
    # Decode the WAV file.
    '''
    time_start_sf = time()
    wav_data, sr = sf.read(AUDIO_FILE_PATH, always_2d=True, dtype=np.float32)
    assert wav_data.dtype == np.float32, 'Bad sample type: %r' % wav_data.dtype
    #waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    #waveform = waveform.astype('float32')
    # Add normalization to deal with differences between wmv and mp4 amplitudes
    #waveform = waveform / np.max(np.abs(waveform))

    # Convert to mono and the sample rate expected by YAMNet.
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
    if sr != FIXED_SR:
        wav_data = resampy.resample(wav_data, sr, FIXED_SR)  # this takes crazy long when wav_data is long

    #wav_data = np.reshape(wav_data, [1, -1]).astype(np.float32)
    wav_data = wav_data[..., np.newaxis]
    #_, patches = features_lib.waveform_to_log_mel_spectrogram_patches(tf.squeeze(waveform, axis=0), params)
    time_end_sf = time()
    print(f'soundfile took {time_end_sf - time_start_sf} to load, shape and resample wav')
    '''

    try:
        time_start_librosa = time()
        wav_data, sr = librosa.load(AUDIO_FILE_PATH, sr=4000, mono=True)
        if (sr < 4000):
            print("Sample Rate Unexpectadly low!", sr)
        print("File size", wav_data.shape)
        wav_data = wav_data[..., np.newaxis]
        time_end_librosa = time()
        print(f'librosa took {time_end_librosa - time_start_librosa} to load, shape and resample wav')
    except:
        print("FILE Failed", AUDIO_FILE_PATH)

    processor_chunk_size = ceil(len(wav_data) / os.cpu_count())
    dataloader = []
    current_chunk_position = 0
    while current_chunk_position < len(wav_data):
        print(f'current_chunk_position = {current_chunk_position} is << len(data) {len(wav_data)}')
        current_chunk = wav_data[current_chunk_position:current_chunk_position + processor_chunk_size]
        dataloader.append(current_chunk)
        current_chunk_position += processor_chunk_size

    final_results = []
    # create a pool with the number of available cores and distribute batches to separate processes
    async with Pool() as pool:
        print(f'Processes used = {list(pool.processes)}')
        async for result in pool.map(process_batch, dataloader):
            print(f'result returned = {result}')
            final_results.append(result)
    # 34 result comes from processor_chunk_size = 42528, divided by frames = 1248 in generator = 34.07692
    await pool.join()
    pool.close()

    # Can add part to extract elephant calls here for this entire file

    return final_results


if __name__ == '__main__':
    data_dir = 'test_data_dir'
    try:
        final = asyncio.run(main(data_dir))
        print(f'final in __main__ = {final}')
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')


