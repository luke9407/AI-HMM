import csv
import os

import librosa
import numpy as np

from model import HMMModel

# train.csv is csv file in form of 'audio_file_id, audio_class'
TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'urban-sound-classification/train')
TRAIN_INFO_FILE = os.path.join(TRAIN_PATH, 'train.csv')
TRAIN_AUDIO_DIR = os.path.join(TRAIN_PATH, 'Train')

f = open(TRAIN_INFO_FILE, 'r')

train_info = csv.reader(f)
next(train_info, None)

train_data = {}
train_lengths = {}
first_shape = None

print 'Extracting mfcc...'

for idx, line in enumerate(train_info):
    audio_id, audio_class = line

    audio_file_path = os.path.join(TRAIN_AUDIO_DIR, str(audio_id) + '.wav')
    if not os.path.isfile(audio_file_path):
        continue

    y, sr = librosa.load(audio_file_path)  # audio file load
    mfcc = librosa.feature.mfcc(y, sr=sr)  # MFCC feature extraction
    spectral_centroid = librosa.feature.spectral_centroid(y, sr=sr)[0]  # Spectral centroid extraction for comparison
    spectral_rolloff = librosa.feature.spectral_rolloff(y, sr=sr)[0]  # Spectral rolloff extraction for comparison
    if first_shape is not None and mfcc.shape != first_shape:
        # Each audio files have different mfcc shape so resize it
        mfcc = np.resize(mfcc, first_shape)
        spectral_centroid = np.resize(spectral_centroid, first_shape[1])
        spectral_rolloff = np.resize(spectral_rolloff, first_shape[1])

    # For training, just use mfcc.
    # I ran some tests with other features like spectral_centroid, but accuracy was worse.
    feature = mfcc

    if idx == 0:
        first_shape = mfcc.shape

    if audio_class not in train_data:
        train_data[audio_class] = feature
        train_lengths[audio_class] = []
    else:
        train_data[audio_class] = np.concatenate([train_data[audio_class], feature])

    # Model fitting in hmmlearn library, we need to give multiple sequences' length as input
    train_lengths[audio_class].append(len(feature))

    if idx % 100 == 0:
        print 'Line {0} finished!'.format(idx)

print 'Train start!'
models = {}
test_data = {}
test_lengths = {}
train_portion = 0.9  # Use 90% of data as training and 10% as testing

for audio_class in train_lengths:
    test_index = int(len(train_lengths[audio_class]) * train_portion)
    length_sum = sum(train_lengths[audio_class][:test_index])

    test_lengths[audio_class] = train_lengths[audio_class][test_index:]
    test_data[audio_class] = train_data[audio_class][length_sum:]

    # Make HMM model for each audio class
    models[audio_class] = HMMModel()
    models[audio_class].train(train_data[audio_class][:length_sum], train_lengths[audio_class][:test_index])

accuracy = {}

for real_class in test_data:
    accuracy[real_class] = {'total': 0, 'correct': 0}
    start = 0

    for test_length in test_lengths[audio_class]:
        test_mfcc = test_data[real_class][start:(start + test_length)]
        scores = {}
        for train_class in models:
            evaluated = models[train_class].evaluate(test_mfcc)
            scores[train_class] = evaluated
        # Score the test audio file with each audio HMM class and sort by score
        sorted_score = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        rank = 1
        for ss in sorted_score:
            if ss[0] == real_class:
                break
            rank += 1
        print 'Real class : {0}, Rank : {1}'.format(real_class, rank)
        accuracy[real_class]['total'] += 1
        # If real class's rank is less than 3, consider it as right answer.
        if rank <= 3:
            accuracy[real_class]['correct'] += 1
        start += test_length

total = 0
correct = 0

# Print summary
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
