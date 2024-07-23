import argparse

import numpy as np
import torch
import datetime
import json
import yaml
import os

from Data_Preparation.data_prepare_ssed import prepare_data

from DDPM import DDPM
from denoising_model_seed import DualBranchDenoisingModel
from utils import train, evaluate

from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--n_type', type=str, default='EOG', help='noise version')
    args = parser.parse_args()
    print(args)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    foldername = "./check_points/noise_type_" + args.n_type + "/"
    print('folder:', foldername)
    os.makedirs(foldername, exist_ok=True)

    [X_train, y_train] = prepare_data(r'data/ssed_noise.npy', r'data/ssed_eeg.npy')

    # X_train = np.load(r'./data/noise_train.npy')
    # y_train = np.load(r'./data/eeg_train.npy')

    X_train = torch.FloatTensor(X_train).unsqueeze(dim=1)
    y_train = torch.FloatTensor(y_train).unsqueeze(dim=1)

    print(X_train.shape)

    train_val_set = TensorDataset(y_train, X_train)
    train_idx, val_test_idx = train_test_split(list(range(len(train_val_set))), test_size=0.2, random_state=666)
    test_idx, val_idx = train_test_split(list(range(len(val_test_idx))), test_size=0.5, random_state=666)

    train_set = Subset(train_val_set, train_idx)
    val_set = Subset(train_val_set, val_idx)
    test_set = Subset(train_val_set, test_idx)

    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=64, num_workers=8)


    base_model = DualBranchDenoisingModel(config['train']['feats']).to(args.device)
    model = DDPM(base_model, config, args.device)

    train(model, config['train'], train_loader, args.device,
          valid_loader=val_loader, valid_epoch_interval=10, foldername=foldername)

    #eval best
    print('eval')
    foldername = "./check_points/noise_type_" + args.n_type + "/"
    output_path = foldername + "/model.pth"
    model.load_state_dict(torch.load(output_path))
    evaluate(model, val_loader, args.device)










