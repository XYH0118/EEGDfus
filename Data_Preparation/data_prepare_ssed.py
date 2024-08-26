import numpy as np
from sklearn import preprocessing

def prepare_data(X_train_path, y_train_path):

    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)

    eeg_train = []
    noise_train = []

    for i in range(X_train.shape[0]):

        eeg = y_train[i]
        eeg = preprocessing.scale(eeg, axis=1)

        eog = X_train[i] - y_train[i]
        eog = preprocessing.scale(eog, axis=1)

        noise = eeg + eog

        eeg_train.extend(eeg)
        noise_train.extend(noise)


    eeg_train = np.array(eeg_train)
    noise_train = np.array(noise_train)
    print(eeg_train.shape)

    dataset = [eeg_train, noise_train]

    return dataset