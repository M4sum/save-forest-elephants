#!/usr/bin/env python3
"""Play an audio file using a limited amount of memory.

The soundfile module (https://python-soundfile.readthedocs.io/) must be
installed for this to work.

In contrast to play_file.py, which loads the whole file into memory
before starting playback, this example program only holds a given number
of audio blocks in memory and is therefore able to play files that are
larger than the available RAM.

This example is implemented using NumPy, see play_long_file_raw.py
for a version that doesn't need NumPy.

"""
import argparse
import queue
import sys
import threading
import numpy as np

import sounddevice as sd
import soundfile as sf

# AUDIO_FILE_PATH = "C:\\Users\\estin\\PycharmProjects\\ElephantsEfficientProcessingTeam\\finding_manny\\input_data\\tree3_20180517_095900.wav"
AUDIO_FILE_PATH = "C:\\Users\\estin\\PycharmProjects\\ElephantsEfficientProcessingTeam\\finding_manny\\input_data\\testmp3.mp3"

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


'''
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'filename', metavar='FILENAME',
    help='audio file to be played back')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='output device (numeric ID or substring)')
parser.add_argument(
    '-b', '--blocksize', type=int, default=2048,
    help='block size (default: %(default)s)')
parser.add_argument(
    '-q', '--buffersize', type=int, default=20,
    help='number of blocks used for buffering (default: %(default)s)')
args = parser.parse_args(remaining)
if args.blocksize == 0:
    parser.error('blocksize must not be zero')
if args.buffersize < 1:
    parser.error('buffersize must be at least 1')
'''

# q = queue.Queue(maxsize=args.buffersize)
q = queue.Queue(maxsize=20)
event = threading.Event()


def callback(outdata, frames, time, status):
    # assert frames == args.blocksize
    assert frames == 2048
    if status.output_underflow:
        print('Output underflow: increase blocksize?', file=sys.stderr)
        raise sd.CallbackAbort
    assert not status
    try:
        data = q.get_nowait()
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


try:
    # with sf.SoundFile(args.filename) as f:
    with sf.SoundFile(AUDIO_FILE_PATH) as f:
        for _ in range(20):
            data = f.read(2048)
            if not len(data):
                break
            q.put_nowait(data)  # Pre-fill queue
        #stream = sd.OutputStream(samplerate=f.samplerate, blocksize=args.blocksize,device=args.device, channels=f.channels,callback=callback, finished_callback=event.set)
        stream = sd.OutputStream(samplerate=f.samplerate, blocksize=2048, device=sd.default.device,channels=f.channels, callback=callback, finished_callback=event.set)
        with stream:
            # timeout = args.blocksize * args.buffersize / f.samplerate
            timeout = 2048 * 20 / f.samplerate
            while len(data):
                # data = f.read(args.blocksize)
                data = f.read(2048)
                q.put(data, timeout=timeout)
            event.wait()  # Wait until playback is finished
except KeyboardInterrupt:
    #parser.exit('\nInterrupted by user')
    print('\nInterrupted by user')
except queue.Full:
    # A timeout occurred, i.e. there was an error in the callback
    # parser.exit(1)
    print('\nA timeout occurred, i.e. there was an error in the callback, so queue is full')
    exit(1)
except Exception as e:
    # parser.exit(type(e).__name__ + ': ' + str(e))
    exit(type(e).__name__ + ': ' + str(e))