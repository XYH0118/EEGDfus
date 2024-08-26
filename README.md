# EEGDfus

## 1. Download the Dataset
Download the EEG denoising dataset from the following link:
- [https://drive.google.com/drive/folders/1n25Kx7BApBFZX3JX_b1g4V2BM0lAw-Zp?usp=drive_link](#)

## 2. Prepare the Data
Place all downloaded data into the `data` folder located in the root directory of the project.

## 3. Configure Parameters
Modify the parameters in the `base.yaml` file located in the `config` folder according to the settings provided in the article.

## 4. Train the Denoising Models
Run the following commands in the terminal to train the denoising models:

```bash
python train_eegdnet.py
python train_ssed.py
```

## 5. Save the Trained Models
After training, the models will be saved in the `check_points` folder.
