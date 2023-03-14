import os

import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
from torch.utils.data import DataLoader
import numpy as np
from tqdm.asyncio import tqdm
import feature_util
from SignatureDev.ContrastiveLearning.ContrastiveLoss import ContrastiveLoss
from SignatureDev.ContrastiveLearning.Model import CNVExtractorContrastive
from SignatureDev.ContrastiveLearning.contrastive_dataloader import CNVImagesContrastive
from SignatureDev.training_util import visualize_latent
# Linux rtx3090, pytorch 1.13.1, cublas 11.11.6,cudnn 8.5.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
lr = 1e-2
batch_size = 128
wd = 1e-3
L_to_try = [14]
dataset = CNVImagesContrastive("{}/data/output/make_square_images/".format(feature_util.mac_path))
dataset_size = len(dataset.annotation)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size,num_workers=0, shuffle=True)

for l in L_to_try:
    for temp in [0.02]:

        best_loss = np.inf

        print(l, temp)
        if not os.path.exists("{}/SignatureDev/Plots/{}/".format(feature_util.mac_path, l)):
            os.mkdir("{}/SignatureDev/Plots/{}/".format(feature_util.mac_path, l))
        if not os.path.exists("{}/SignatureDev/Plots/{}/{}/".format(feature_util.mac_path, l,temp)):
            os.mkdir("{}/SignatureDev/Plots/{}/{}/".format(feature_util.mac_path, l,temp))
        env_e = CNVExtractorContrastive(encoded_space_dim=l)
        env_e.to(device)

        wandb.init(
            name="{}_{}".format(l,temp),
            project="CNVSigs",
            config={
                "learning_rate": lr,
                "architecture": "CNN-Contrastive",
                "dataset": "make_square_images",
                "batch_size": batch_size,
                "weight_decay": wd,
                "L_size": l
            }
        )

        loss_fn = ContrastiveLoss(batch_size, temperature=temp, device=device).to(device)
        optim = torch.optim.Adam(env_e.parameters(), lr=lr, weight_decay=wd)
        scheduler = ReduceLROnPlateau(optim, 'min', factor=0.5, patience=3)
        num_epochs = 150
        for epoch in tqdm(range(num_epochs)):
            # Train:
            env_e.train()
            train_loss = []
            for batch_index, (X1, X2, id) in enumerate(trainLoader):
                optim.zero_grad()
                recon1, L1 = env_e(X1.float().to(device))
                recon2, L2 = env_e(X2.float().to(device))
                total_loss = loss_fn(L1.to(device), L2.to(device))
                train_loss.append(total_loss.item())
                total_loss.backward()
                optim.step()
            wandb.log({"Train/loss": np.mean(train_loss),
                       "Epoch": epoch})
            # Valid
            env_e.eval()
            val_loss,val_kl,val_log_pxz = [],[],[]
            val_Ls,ids = [],[]
            with torch.no_grad():
                for batch_index, (X1,X2,id) in enumerate(valLoader):
                    recon1, L1 = env_e(X1.float().to(device))
                    recon2, L2 = env_e(X2.float().to(device))
                    val_Ls.append(L1.float().cpu().numpy())
                    ids.append(id)
                    total_loss = loss_fn(L1.to(device), L2.to(device))
                    train_loss.append(total_loss.item())
                    val_loss.append(total_loss.item())
                wandb.log({"Valid/loss": np.mean(val_loss),
                           "Epoch":epoch})
                scheduler.step(np.mean(val_loss))
            #Plot validation data from last epoch
            visualize_latent(val_Ls, ids, l, epoch,temp)
            if np.mean(val_loss) < best_loss and epoch > 30:
                best_loss = np.mean(val_loss)
                print("current best loss: ", best_loss)
                print("saving model...")
                torch.save(env_e.state_dict(), "{}/{}_best_model_contrastive_temp_{}.pth".format(feature_util.mac_path, l,temp))
        wandb.finish()