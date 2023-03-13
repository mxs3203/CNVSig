import os

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from umap import UMAP

import wandb
from torch.utils.data import DataLoader
import numpy as np
from tqdm.asyncio import tqdm
import feature_util
from SignatureDev.VariationalAutoEncoder.Model import CNVExtractorVAE
from SignatureDev.dataloader import CNVImages
from SignatureDev.training_util import visualize_latent


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#from feature_util import readPickle
print(device)
lr = 1e-2
batch_size = 128
wd = 1e-3
L_to_try = [7]#range(24, 5, -1)
dataset = CNVImages("{}/data/output/make_square_images/".format(feature_util.mac_path))
dataset_size = len(dataset.annotation)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size,num_workers=0, shuffle=True)

best_loss = np.inf

for l in L_to_try:
    print(l)
    if not os.path.exists("{}/SignatureDev/Plots/{}/".format(feature_util.mac_path, l)):
        os.mkdir("{}/SignatureDev/Plots/{}/".format(feature_util.mac_path, l))
    env_e = CNVExtractorVAE(encoded_space_dim=l)
    env_e.to(device)

    wandb.init(
        name="{}".format(l),
        project="CNVSigs",
        config={
            "learning_rate": lr,
            "architecture": "CNN-AE",
            "dataset": "make_square_images",
            "batch_size": batch_size,
            "weight_decay": wd,
            "L_size":l
        }
    )

    loss_fn = torch.nn.MSELoss(reduction="sum")
    optim = torch.optim.Adam(env_e.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optim, 'min', factor=0.5, patience=3)
    num_epochs = 500
    for epoch in tqdm(range(num_epochs)):
        # Train:
        env_e.train()
        train_loss = []
        for batch_index, (X,id) in enumerate(trainLoader):
            optim.zero_grad()
            recon, L,kl = env_e(X.float())
            # total_loss = 0
            # for depth in range(np.shape(X)[1]): # until 9
            #     layer_loss = loss_fn(recon[:,depth,:,:], X.float()[:,depth,:,:])
            #     total_loss += layer_loss
            total_loss = loss_fn(recon, X.float()) + kl
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
            for batch_index, (X,id) in enumerate(valLoader):
                recon, L,kl = env_e(X.float())
                val_Ls.append(L.float().numpy())
                ids.append(id)
                # total_loss = 0
                # for depth in range(np.shape(X)[1]):  # until 9
                #     layer_loss = loss_fn(recon[:, depth, :, :], X.float()[:, depth, :, :])
                #     total_loss += layer_loss
                total_loss = loss_fn(recon, X.float()) + kl
                train_loss.append(total_loss.item())
                val_loss.append(total_loss.item())
            wandb.log({"Valid/loss": np.mean(val_loss),
                       "Epoch":epoch})
            scheduler.step(np.mean(val_loss))
        #Plot validation data from last epoch
        visualize_latent(val_Ls, ids, l, epoch)
        if np.mean(val_loss) < best_loss:
            best_loss = np.mean(val_loss)
            print("current best loss: ", best_loss)
            print("saving model...")
            torch.save(env_e.state_dict(), "{}/best_model_vae.pth".format(feature_util.mac_path))
    wandb.finish()