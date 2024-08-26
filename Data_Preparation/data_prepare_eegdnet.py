import numpy as np
from sklearn import preprocessing

def get_rms(records, multi_channels):
    """
    均方根值 反映的是有效值而不是平均值
    """
    if multi_channels == 1:
        n = records.shape[0]
        rms = 0
        for i in range(n):
            rms_t = np.sum([records[i]**2]) / len(records[i])
            rms = rms + rms_t
        return rms / n

    if multi_channels == 0:
        rms = np.sum([records**2]) / len(records)
        return rms


def get_SNR(signal, noisy):
    snr = 10 * np.log10(signal / noisy)
    return snr

def random_signal(signal, combin_num):
    random_result = []

    for i in range(combin_num):
        random_num = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[random_num, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0], signal.shape[1])
        random_result.append(shuffled_dataset)

    random_result = np.array(random_result)

    return random_result


def prepare_data(combin_num, train_per, noise_type):

    file_location = './data/'  ############ change it to your own location #########
    if noise_type == 'EOG':
        EEG_all = np.load(file_location + 'EEG_all_epochs.npy')
        noise_all = np.load(file_location + 'EOG_all_epochs.npy')
    elif noise_type == 'EMG':
        EEG_all = np.load(file_location + 'EEG_all_epochs.npy')
        noise_all = np.load(file_location + 'EMG_all_epochs.npy')


    EEG_all_random = np.squeeze(random_signal(signal=EEG_all, combin_num=1))
    noise_all_random = np.squeeze(random_signal(signal=noise_all, combin_num=1))

    if noise_type == 'EMG':
        reuse_num = noise_all_random.shape[0] - EEG_all_random.shape[0]
        EEG_reuse = EEG_all_random[0: reuse_num, :]
        EEG_all_random = np.vstack([EEG_reuse, EEG_all_random])
        print('EEG segments after reuse: ', EEG_all_random.shape[0])

    elif noise_type == 'EOG':
        EEG_all_random = EEG_all_random[0:noise_all_random.shape[0]]
        print('EEG segments after drop: ', EEG_all_random.shape[0])

    timepoint = noise_all_random.shape[1]
    train_num = round(train_per * EEG_all_random.shape[0])
    test_num = round(EEG_all_random.shape[0] - train_num)

    train_eeg = EEG_all_random[0: train_num, :]
    test_eeg = EEG_all_random[train_num: train_num + test_num, :]

    train_noise = noise_all_random[0: train_num, :]
    test_noise = noise_all_random[train_num: train_num + test_num, :]

    EEG_train = random_signal(signal=train_eeg, combin_num=combin_num).reshape(combin_num * train_eeg.shape[0],
                                                                               timepoint)
    NOISE_train = random_signal(signal=train_noise, combin_num=combin_num).reshape(combin_num * train_noise.shape[0],
                                                                                   timepoint)

    EEG_test = random_signal(signal=test_eeg, combin_num=combin_num).reshape(combin_num * test_eeg.shape[0],
                                                                             timepoint)
    NOISE_test = random_signal(signal=test_noise, combin_num=combin_num).reshape(combin_num * test_noise.shape[0],
                                                                                 timepoint)

    print(EEG_train.shape)
    print(NOISE_train.shape)

    sn_train = []
    eeg_train = []
    all_sn_test = []
    all_eeg_test = []

    # Adding noise to train
    SNR_train_dB = np.random.uniform(-5, 5, (EEG_train.shape[0]))
    print(SNR_train_dB.shape)
    SNR_train = np.sqrt(10 ** (0.1 * (SNR_train_dB)))

    for i in range(EEG_train.shape[0]):

        noise = preprocessing.scale(NOISE_train[i])
        EEG = preprocessing.scale(EEG_train[i])

        coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNR_train[i])
        noise = noise * coe
        signal_noise = EEG + noise

        sn_train.append(signal_noise)
        eeg_train.append(EEG)

    SNR_test_dB = np.linspace(-5.0, 5.0, num=(11))
    SNR_test = np.sqrt(10 ** (0.1 * (SNR_test_dB)))


    for i in range(11):

        sn_test = []
        eeg_test = []

        for j in range(EEG_test.shape[0]):
            noise = preprocessing.scale(NOISE_test[j])
            EEG = preprocessing.scale(EEG_test[j])

            coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNR_test[i])
            noise = noise * coe
            signal_noise = EEG + noise

            sn_test.append(signal_noise)
            eeg_test.append(EEG)

        sn_test = np.array(sn_test)
        eeg_test = np.array(eeg_test)

        all_sn_test.append(sn_test)
        all_eeg_test.append(eeg_test)

    X_train = np.array(sn_train)
    y_train = np.array(eeg_train)

    X_test = np.array(all_sn_test)
    y_test = np.array(all_eeg_test)

    X_train = np.expand_dims(X_train, axis=1)
    y_train = np.expand_dims(y_train, axis=1)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    np.save('./data/data_for_test/X_test.npy', X_test)
    np.save('./data/data_for_test/y_test.npy', y_test)

    Dataset = [X_train, y_train, X_test, y_test]

    print('Dataset ready to use.')

    return Dataset