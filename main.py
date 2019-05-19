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
    if first_shape is not None and mfcc.shape != first_shape:
        mfcc = np.resize(mfcc, first_shape)

    if idx == 0:
        first_shape = mfcc.shape

    if audio_class not in train_data:
        train_data[audio_class] = mfcc
    else:
        train_data[audio_class] = np.concatenate((train_data[audio_class], mfcc))

    if idx % 100 == 0:
        print 'Line {0} finished!'.format(idx)

print 'Train start!'
models = {}
test_data = {}
train_portion = 0.9

for audio_class in train_data:
    portion = int(int(train_data[audio_class].shape[0] / first_shape[0]) * train_portion)
    test_index = portion * first_shape[0]
    test_data[audio_class] = train_data[audio_class][test_index:]

    models[audio_class] = HMMModel()
    models[audio_class].train(train_data[audio_class][:test_index])

accuracy = {}

for real_class in test_data:
    accuracy[real_class] = {'total': 0, 'correct': 0}
    start = 0
    while True:
        test_mfcc = test_data[real_class][start:(start + first_shape[0])]
        if test_mfcc.size == 0:
            break
        score = None
        hmm_result = None
        for train_class in models:
            evaluated = models[train_class].evaluate(test_mfcc)
            if score is None or score < evaluated:
                score = evaluated
                hmm_result = train_class
        if hmm_result is not None:
            print 'Real class : {0}, HMM result : {1}'.format(real_class, hmm_result)
        accuracy[real_class]['total'] += 1
        if real_class == hmm_result:
            accuracy[real_class]['correct'] += 1
        start += first_shape[0]

total = 0
correct = 0

for audio_class in accuracy:
    t = accuracy[audio_class]['total']
    c = accuracy[audio_class]['correct']
    p = (float(c) / float(t)) * 100.0

    total += t
    correct += c

    print 'Class : {0}, Total test : {1}, Correct : {2}, Accuracy : {3}'.format(audio_class, t, c, p)

print '===================================================================='
p = (float(correct) / float(total)) * 100.0
print 'Total : {0}, Correct : {1}, Accuracy : {2}'.format(total, correct, p)
