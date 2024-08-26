import argparse
import torch
import datetime
import json
import yaml
import os

from Data_Preparation.data_prepare_eegdnet import prepare_data

from DDPM import DDPM
from denoising_model_eegdnet import DualBranchDenoisingModel
from utils import train, evaluate

from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

from sklearn.model_selection import train_test_split

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

    [X_train, y_train, X_test, y_test] = prepare_data(combin_num=11, train_per=0.9, noise_type=args.n_type)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    train_val_set = TensorDataset(y_train, X_train)
    test_set = TensorDataset(y_test, X_test)

    train_idx, val_idx = train_test_split(list(range(len(train_val_set))), test_size=0.2)
    train_set = Subset(train_val_set, train_idx)
    val_set = Subset(train_val_set, val_idx)

    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=64, num_workers=8)

    # base_model = ConditionalModel(64,8,4).to(args.device)
    base_model = DualBranchDenoisingModel(config['train']['feats']).to(args.device)
    model = DDPM(base_model, config, args.device)

    train(model, config['train'], train_loader, args.device,
          valid_loader=val_loader, valid_epoch_interval=10, foldername=foldername)

    # eval best
    print('eval best')
    foldername = "./check_points/noise_type_" + args.n_type + "/"
    output_path = foldername + "/model.pth"
    model.load_state_dict(torch.load(output_path))
    evaluate(model, val_loader, args.device)










