import boto3
import os
from time import time
from scipy.io import wavfile
import numpy as np
import miniaudio

audio_path = "C:\\Users\\estin\\PycharmProjects\\ElephantsDetector\\test_EC\\data\\nn09a_20201025_120105.wav"
##### What stanford has done
start_time_audio_read = time()
samplerate, raw_audio = wavfile.read(audio_path)
if (samplerate < 4000):
     print ("Sample Rate Unexpectadly low!", samplerate)
print ("File size", raw_audio.shape)
end_time_audio_read = time()
#timing_audio_read = np.round(end_time_audio_read - start_time_audio_read, 4)
timing_audio_read = np.round(end_time_audio_read - start_time_audio_read, 10)
print(f'READING WAV FILE IS TAKING {timing_audio_read} seconds')
duration = float(raw_audio.size / samplerate)
print(f'audio file is {duration} seconds long')


#audio_path = "my_audio_file.mp3"
target_sampling_rate = 8000  #the input audio will be resampled a this sampling rate 44100
n_channels = 1  #either 1 or 2
waveform_duration = 10 #in seconds 30
offset = 10 #this means that we read only in the interval [15s, duration of file] 15

''''
function stream_file (filename: str, output_format: miniaudio.SampleFormat = <SampleFormat.SIGNED16: 2>, 
nchannels: int = 2, sample_rate: int = 44100, frames_to_read: int = 1024, 
dither: miniaudio.DitherMode = <DitherMode.NONE: 0>, seek_frame: int = 0) 
-> Generator[array.array, int, NoneType]
'''

start_time_audio_read_stream = time()
waveform_generator = miniaudio.stream_file(
     filename = audio_path,
     sample_rate = target_sampling_rate,
     seek_frame = 0,       #seek_frame = int(offset * target_sampling_rate),
     frames_to_read = int(waveform_duration * target_sampling_rate),
     output_format = miniaudio.SampleFormat.SIGNED16,          #miniaudio.SampleFormat.FLOAT32,
     nchannels = n_channels)


for i, waveform in enumerate(waveform_generator):
    #do something with the waveform....
    print(f'{i}th waveform size = {len(waveform)}')
    # we can pass i and waveform to separate thread/subprocess to generate spectrograms of that section

end_time_audio_read_stream = time()
timing_audio_read_stream = np.round(end_time_audio_read_stream - start_time_audio_read_stream, 10)
print(f'READING WAV FILE AS STREAM AND DISTRIBUTING CHUNKS IS TAKING {timing_audio_read_stream} seconds')
'''
# Create an S3 resource
s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-2',
    aws_access_key_id='AKIAR2Q47EXTL23Q7RK4',
    aws_secret_access_key='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'
)

# Define the S3 bucket and object key
bucket_name = "data-ai-for-forest-elephants"
object_key = "list_of_sounds_final.txt"
# Retrieve the object from S3
obj = s3.Object(bucket_name, object_key)
# Read the contents of the file
#file_content = obj.get()[‘Body’].read().decode(‘utf-8’)
file_content=obj.get()['Body'].read().decode(encoding="utf-8",errors="ignore")
# Print the contents of the file
print(file_content)
#print(type(file_content))
'''