import wave
import contextlib
import os
import glob
from datetime import datetime
import sys

subdirs = [x[0] for x in os.walk('F:/Program Files/BirdCall-Recognition/train_audio/')]

minimum_duration=max_duration= 0
short_audio = 0
long_audio = 0
min_name=max_name = ''

for sub in subdirs:
    for file in glob.iglob(sub + '/*.wav'):
        with contextlib.closing(wave.open(file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            name = os.path.splitext(os.path.basename(file))[0]
            dst = os.path.dirname(file) + '/' + name + '.wav'

            if minimum_duration > duration or minimum_duration == 0:
                minimum_duration = duration
                min_name = dst
            if max_duration < duration or max_duration == 0:
                max_duration = duration
                max_name = dst
                print(rate)
                print(frames)
                print(frames/rate)

            if 1 < duration < 120:
                short_audio += 1
            if duration > 120:
                long_audio += 1

            
            
print("The shortest wav file is: ", minimum_duration)
print("The shortest wav file is called: ", min_name)
print("The longest wav file is: ", max_duration)
print("The longest wav file is called: ", max_name)

print("Audio Files between 1s and 2m: ", short_audio)
print("Audio Files longer than 2m: ", long_audio)
