import os
import asyncio
from aiomultiprocess import Pool
import sys
import sounddevice as sd
import soundfile as sf
from math import ceil

AUDIO_FILE_PATH = "C:\\Users\\estin\\PycharmProjects\\ElephantsEfficientProcessingTeam\\finding_manny\\input_data\\testmp3.mp3"

BLOCK_SIZE = 2496
current_frame = 0

async def generate_spectrogram_of_chunk(chunkID, chunk):
    # print(f'generate_spectrogram_of_chunk in {os.getpid()} with chunk {chunkID}')
    await asyncio.sleep(1)      # simulates generating spectrogram
    print(f'generate_spectrogram_of_chunk in {os.getpid()} woke up with chunk {chunkID}')


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
    fs = 48000

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

    stream = sd.OutputStream(samplerate=fs, device=sd.default.device, blocksize=BLOCK_SIZE, channels=indata_processor_chunk.shape[1], callback=callback)
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
    data, fs = sf.read(AUDIO_FILE_PATH, always_2d=True)
    print(f'sampling rate = {fs}')
    processor_chunk_size = ceil(len(data) / os.cpu_count())
    dataloader = []
    current_chunk_position = 0
    while current_chunk_position < len(data):
        print(f'current_chunk_position = {current_chunk_position} is << len(data) {len(data)}')
        current_chunk = data[current_chunk_position:current_chunk_position + processor_chunk_size]
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


