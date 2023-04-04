import os
import asyncio
from aiomultiprocess import Pool
import miniaudio


async def generate_spectrogram_of_chunk(chunkID, chunk):
    print(f'generate_spectrogram_of_chunk in {os.getpid()} with chunk {chunkID}')
    await asyncio.sleep(2)      # simulates generating spectrogram
    print(f'generate_spectrogram_of_chunk in {os.getpid()} woke up with chunk {chunkID}')


async def predict_spec_sliding_window(chunkID, chunk):
    print(f'predict_spec_sliding_window in {os.getpid()} with chunk {chunkID}')
    await asyncio.sleep(2)      # simulates prediction
    print(f'predict_spec_sliding_window in {os.getpid()} woke up with chunk {chunkID}')

async def process_batch(waveform_generator):
    print(f'process_batch function in process {os.getpid()} with args {input}')
    # Need to split batch into chunks and can run each chunk concurrently -> I think use async for loop?

    i = 0
    async for waveform in waveform_generator:
        # do something with the waveform....
        print(f'{i}th waveform size = {len(waveform)}')
        i += 1
        await generate_spectrogram_of_chunk(i, waveform)  # simulates generating spectrogram
        await predict_spec_sliding_window(i, waveform)  # simulates prediction

    return input


async def main(in_args):
    print(f'argument passed to async main {in_args}')
    print(f'number of available cores = {os.cpu_count()}')
    # Use input arguments for dataloader to get raw audio batches
    # must return in batches with asynchronous iterator to split batch into chunks - create class AudioIterator
    '''
    dataloader = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
                  (23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (33, 34), (35, 36), (37, 38), (39, 40), (41, 42),
                  (43, 44), (45, 46), (47, 48)]
    '''
    audio_path = "C:\\Users\\estin\\PycharmProjects\\ElephantsDetector\\test_EC\\data\\nn09a_20201025_120105.wav"
    target_sampling_rate = 8000  # the input audio will be resampled at this sampling rate 44100
    n_channels = 1  # either 1 or 2
    waveform_duration = 100  # in seconds 30
    offset = 100  # this means that we read only in the interval [15s, duration of file] 15
    dataloader = miniaudio.wav_stream_file(
        filename=audio_path,
        seek_frame=0,  # seek_frame = int(offset * target_sampling_rate),
        frames_to_read=int(waveform_duration * target_sampling_rate)
    )
    final_results = []
    # create a pool with the number of available cores and distribute batches to separate processes
    async with Pool() as pool:
        print(f'Processes used = {list(pool.processes)}')
        async for result in pool.map(process_batch, dataloader):
            print(f'result returned = {result}')
            final_results.append(result)

    await pool.join()
    pool.close()

    return final_results

if __name__ == '__main__':
    data_dir = 'test_data_dir'
    final = asyncio.run(main(data_dir))
    print(f'final in __main__ = {final}')

