
import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import glob
import time
import sys
max_columns = 0

def extract_features(file):
    global max_columns

    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        if max_columns < mfccs.shape[1]:
            max_columns = mfccs.shape[1]
            print("Current max # of columns: ",max_columns,end='\r')
            
    except Exception as e:
        print("Error while parsing file: ", file)
        return None
    
    return mfccs_scaled

train_data = pd.read_csv("../train.csv")

subdirs = [x[0] for x in os.walk('F:/Program Files/BirdCall-Recognition/train_audio/')]
features = []
count = 0

for sub in subdirs:   
    class_label = os.path.splitext(os.path.basename(sub))[0]
    for file in glob.iglob(sub + '/*.wav'):
        name = os.path.splitext(os.path.basename(file))[0]
        dst = os.path.dirname(file) + '/' + name + '.wav'
        
        data = extract_features(dst)
        features.append([data, class_label])
    count += 1
    print('Folders done: ',count, '/266',end='\r')
        

features_df = pd.DataFrame(features, columns=['feature','class_label'])
features_df.to_csv('features2.csv', index=False, header=True)
   


