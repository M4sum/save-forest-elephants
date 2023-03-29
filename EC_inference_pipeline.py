import os
import asyncio
from aiomultiprocess import Pool


async def generate_spectrogram_of_chunk():
    await asyncio.sleep(2)      # simulates generating spectrogram
    print(f'{os.getpid()} woke up once')


async def predict_spec_sliding_window():
    await asyncio.sleep(2)      # simulates prediction
    print(f'{os.getpid()} woke up twice')

async def process_batch(input):
    print(f'process_batch function in process {os.getpid()} with args {input}')
    # Need to split into chunks and can run each chunk concurrently -> I think async for?
    await generate_spectrogram_of_chunk()      # simulates generating spectrogram
    await predict_spec_sliding_window()         # simulates prediction
    return input


async def main(in_args):
    print(f'argument passed to async main {in_args}')
    print(f'number of available cores = {os.cpu_count()}')
    # Use input arguments for dataloader
    dataloader = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
                  (23, 24), (25, 26), (27, 28), (29, 30), (31, 32), (33, 34), (35, 36), (37, 38), (39, 40), (41, 42),
                  (43, 44), (45, 46), (47, 48)]
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

