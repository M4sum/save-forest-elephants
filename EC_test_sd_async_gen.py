# https://github.com/spatialaudio/python-sounddevice/blob/master/examples/asyncio_generators.py
#!/usr/bin/env python3
"""Creating an asyncio generator for blocks of audio data.

This example shows how a generator can be used to analyze audio input blocks.
In addition, it shows how a generator can be created that yields not only input
blocks but also output blocks where audio data can be written to.

You need Python 3.7 or newer to run this.

"""
import asyncio
import queue
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf

AUDIO_FILE_PATH = "C:\\Users\\estin\\PycharmProjects\\ElephantsEfficientProcessingTeam\\finding_manny\\input_data\\testmp3.mp3"


async def inputstream_generator(channels=1, **kwargs):
    """Generator that yields blocks of input data as NumPy arrays."""
    q_in = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

    stream = sd.InputStream(callback=callback, channels=channels, **kwargs)
    with stream:
        while True:
            indata, status = await q_in.get()
            yield indata, status


async def stream_generator(blocksize, *, channels=1, dtype='float32',
                           pre_fill_blocks=10, **kwargs):
    """Generator that yields blocks of input/output data as NumPy arrays.
    The output blocks are uninitialized and have to be filled with
    appropriate audio signals.
    """
    assert blocksize != 0
    q_in = asyncio.Queue()
    q_out = queue.Queue()
    loop = asyncio.get_event_loop()

    def callback(indata, outdata, frame_count, time_info, status):
        loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))
        print('stream_generator callback triggered')
        outdata[:] = q_out.get_nowait()

    # pre-fill output queue
    for _ in range(pre_fill_blocks):
        q_out.put(np.zeros((blocksize, channels), dtype=dtype))

    stream = sd.Stream(blocksize=blocksize, callback=callback, dtype=dtype,
                       channels=channels, **kwargs)
    with stream:
        while True:
            indata, status = await q_in.get()
            outdata = np.empty((blocksize, channels), dtype=dtype)
            yield indata, outdata, status
            q_out.put_nowait(outdata)


async def outputstream_generator(blocksize, *, channels=1, dtype='float32',
                           pre_fill_blocks=10, **kwargs):
    """Generator that yields blocks of output data from wav file as NumPy arrays.
        The output blocks are uninitialized and have to be filled with
        appropriate audio signals.
        """
    assert blocksize != 0
    q_out = queue.Queue()
    loop = asyncio.get_event_loop()
    def callback(indata, outdata, frame_count, time_info, status):
        # loop.call_soon_threadsafe(q_out.put_nowait, (indata.copy(), status))
        # outdata[:] = q_out.get_nowait()
        assert frame_count == 2048
        if status.output_underflow:
            print('Output underflow: increase blocksize?', file=sys.stderr)
            raise sd.CallbackAbort
        assert not status
        try:
            data = q_out.get_nowait()
            data = data[..., np.newaxis]
        except queue.Empty as e:
            print('Buffer is empty: increase buffersize?', file=sys.stderr)
            raise sd.CallbackAbort from e
        if len(data) < len(outdata):
            outdata[:len(data)] = data
            outdata[len(data):].fill(0)
            raise sd.CallbackStop
        else:
            outdata[:] = data

    '''
    # pre-fill output queue
    for _ in range(pre_fill_blocks):
        q_out.put(np.zeros((blocksize, channels), dtype=dtype))
    '''
    with sf.SoundFile(AUDIO_FILE_PATH) as f:
        for _ in range(20):
            data = f.read(2048)
            if not len(data):
                break
            q_out.put_nowait(data)  # Pre-fill queue
        # stream = sd.OutputStream(samplerate=f.samplerate, blocksize=2048, device=sd.default.device, channels=f.channels, callback=callback)
        stream = sd.Stream(samplerate=f.samplerate, blocksize=2048, device=sd.default.device, callback=callback, dtype=dtype, channels=f.channels, **kwargs)
        with stream:
            while True:
                timeout = 2048 * 20 / f.samplerate
                while len(data):
                    data = f.read(2048)
                    q_out.put(data, timeout=timeout)
                # outdata = np.empty((blocksize, channels), dtype=dtype)
                # outdata, status = await q_out.get()
                # outdata[:] = q_out.get_nowait()
                outdata = q_out.get()
                yield outdata


async def print_input_infos(**kwargs):
    """Show minimum and maximum value of each incoming audio block."""
    async for indata, status in inputstream_generator(**kwargs):
        if status:
            print(status)
        print('min:', indata.min(), '\t', 'max:', indata.max())


async def wire_coro(**kwargs):
    """Create a connection between audio inputs and outputs.

    Asynchronously iterates over a stream generator and for each block
    simply copies the input data into the output block.

    """
    async for indata, outdata, status in stream_generator(**kwargs):
       if status:
           print(status)
       outdata[:] = indata
    # i = 0
    # async for outdata in outputstream_generator(**kwargs):
    #     print(f'processing {i}th chunk in outputstream_generator')
    #     i += 1


async def main(**kwargs):
    print('Some informations about the input signal:')
    try:
        await asyncio.wait_for(print_input_infos(), timeout=2)
    except asyncio.TimeoutError:
        pass
    print('\nEnough of that, activating wire ...\n')
    audio_task = asyncio.create_task(wire_coro(**kwargs))
    for i in range(10, 0, -1):
        print(i)
        await asyncio.sleep(1)
    audio_task.cancel()
    try:
        await audio_task
    except asyncio.CancelledError:
        print('\nwire was cancelled')


if __name__ == "__main__":
    try:
        asyncio.run(main(blocksize=2048))
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')