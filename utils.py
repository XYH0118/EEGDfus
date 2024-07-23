import numpy as np
import torch
from torch.optim import Adam
import metrics


def train(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername=""):
    optimizer = Adam(model.parameters(), lr=config["lr"])


    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1500, gamma=.1, verbose=True
    )

    best_valid_loss = 1e10

    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        count = 0
        model.train()

        for batch_no, (clean_batch, noisy_batch) in enumerate(train_loader, start=1):
            clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
            optimizer.zero_grad()

            loss = model(clean_batch, noisy_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            avg_loss += loss.item()
            count += 1


        lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            evaluate(model, valid_loader, 'cuda:0')
            avg_loss_valid = 0
            with torch.no_grad():

                for batch_no, (clean_batch, noisy_batch) in enumerate(valid_loader, start=1):
                    clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                    loss = model(clean_batch, noisy_batch)
                    avg_loss_valid += loss.item()

            if best_valid_loss > avg_loss_valid / batch_no:
                best_valid_loss = avg_loss_valid / batch_no
                print("\n best loss is updated to ", avg_loss_valid / batch_no, "at", epoch_no, )

                if foldername != "":
                    torch.save(model.state_dict(), output_path)

    torch.save(model.state_dict(), final_path)


def evaluate(model, test_loader, device):
    rrmse_t = 0
    rrmse_s = 0
    cc = 0
    p_value = 0
    snr_noise = 0
    snr_recon = 0
    snr_improvement = 0
    count = 0

    for batch_no, (clean_batch, noisy_batch) in enumerate(test_loader, start=1):
        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)

        output = model.denoising(noisy_batch)  # B,1,L
        clean_batch = clean_batch.permute(0, 2, 1)
        noisy_batch = noisy_batch.permute(0, 2, 1)
        output = output.permute(0, 2, 1)  # B,L,1
        out_numpy = output.cpu().detach().numpy()
        clean_numpy = clean_batch.cpu().detach().numpy()
        noisy_numpy = noisy_batch.cpu().detach().numpy()

        count += 1

        rrmse_t += np.sum(metrics.RRMSE(clean_numpy, out_numpy))
        rrmse_s = np.sum(metrics.RRMSE_s(clean_numpy, out_numpy))
        cc += np.sum(metrics.CC(clean_numpy, out_numpy))
        p_value = np.sum(metrics.Pvalue(clean_numpy, out_numpy))
        snr_noise += np.sum(metrics.SNR(clean_numpy, noisy_numpy))
        snr_recon += np.sum(metrics.SNR(clean_numpy, out_numpy))
        snr_improvement += np.sum(metrics.SNR_improvement(noisy_numpy, out_numpy, clean_numpy))

    print("rrmse_t: ", rrmse_t / count, )
    print("rrmse_s: ", rrmse_s / count, )
    print("cc: ", cc / count, )
    print("p_value: ", p_value / count, )
    print("snr_in: ", snr_noise / count, )
    print("snr_out: ", snr_recon / count, )
    print("snr_improve: ", snr_improvement / count, )