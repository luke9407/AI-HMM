import csv
import os

import librosa
import numpy as np

from model import HMMModel

TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'urban-sound-classification/train')
TRAIN_INFO_FILE = os.path.join(TRAIN_PATH, 'train.csv')
TRAIN_AUDIO_DIR = os.path.join(TRAIN_PATH, 'Train')

f = open(TRAIN_INFO_FILE, 'r')

train_info = csv.reader(f)
next(train_info, None)

train_data = {}
first_shape = test_info = None

print 'Extracting mfcc...'

for idx, line in enumerate(train_info):
    if idx == 500:
        break
    audio_id, audio_class = line

    audio_file_path = os.path.join(TRAIN_AUDIO_DIR, str(audio_id) + '.wav')
    if not os.path.isfile(audio_file_path):
        continue

    y, sr = librosa.load(audio_file_path)
    mfcc = librosa.feature.mfcc(y, sr=sr)
    if first_shape is not None:
        mfcc = np.resize(mfcc, first_shape)

    if idx == 0:
        test_info = [audio_class, mfcc]
        first_shape = mfcc.shape

    if audio_class not in train_data:
        train_data[audio_class] = mfcc
    else:
        np.append(train_data[audio_class], mfcc, axis=0)

    if idx % 100 == 0:
        print 'Line {0} finished!'.format(idx)


print 'Train start!'
models = {}

for audio_class in train_data:
    models[audio_class] = HMMModel()
    models[audio_class].train(train_data[audio_class])

print 'Test start! Test class : {0}'.format(test_info[0])

for audio_class in models:
    score = models[audio_class].evaluate(test_info[1])
    print 'Class : {0}, Score : {1}'.format(audio_class, score)
