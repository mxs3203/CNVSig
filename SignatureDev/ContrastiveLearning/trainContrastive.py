import os

import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from pytorch_metric_learning.losses import NTXentLoss
import wandb
from torch.utils.data import DataLoader
import numpy as np
from tqdm.asyncio import tqdm
import feature_util
from SignatureDev.ContrastiveLearning.Model import CNVExtractorContrastive
from SignatureDev.ContrastiveLearning.contrastive_dataloader import CNVImagesContrastive
from SignatureDev.training_util import visualize_latent
# Linux rtx3090, pytorch 1.13.1, cublas 11.11.6,cudnn 8.5.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
lr = 1e-3
batch_size = 256
wd = 1e-3
L_to_try = [8,10,12,14,16,18]
wGII_groups = 3
dataset = CNVImagesContrastive("{}/data/output/make_square_images/".format(feature_util.linux_path),num_quantiles_for_wgii=wGII_groups )
dataset_size = len(dataset.annotation)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size,num_workers=0, shuffle=True)

for l in L_to_try:
    for temp in [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

        best_loss = np.inf

        print(l, temp)
        if not os.path.exists("{}/SignatureDev/Plots/{}/".format(feature_util.linux_path, l)):
            os.mkdir("{}/SignatureDev/Plots/{}/".format(feature_util.linux_path, l))
        if not os.path.exists("{}/SignatureDev/Plots/{}/{}/".format(feature_util.linux_path, l,temp)):
            os.mkdir("{}/SignatureDev/Plots/{}/{}/".format(feature_util.linux_path, l,temp))
        env_e = CNVExtractorContrastive(encoded_space_dim=l)
        env_e.to(device)
        #
        wandb.init(
            name="{}_{}_{}".format(l,temp, wGII_groups),
            project="CNVSigs_Contrastive_CancerType",
            config={
                "learning_rate": lr,
                "architecture": "CNN-SupContrastive",
                "dataset": "make_square_images",
                "batch_size": batch_size,
                "weight_decay": wd,
                "L_size": l
            }
        )

        loss_fn = NTXentLoss(temperature=temp)
        optim = torch.optim.Adam(env_e.parameters(), lr=lr, weight_decay=wd)
        scheduler = StepLR(optim, step_size=10, gamma=0.9)
        num_epochs = 50
        for epoch in tqdm(range(num_epochs)):
            #print('Epoch-{0} lr: {1}'.format(epoch, optim.param_groups[0]['lr']))
            # Train:
            env_e.train()
            train_loss = []
            for batch_index, (X1, X2, id, wGII_group,c_type) in enumerate(trainLoader):
                optim.zero_grad()
                #print(np.shape(wGII_group))
                recon1, L1 = env_e(X1.float().to(device))
                #recon2, L2 = env_e(X2.float().to(device))
                total_loss = loss_fn(L1, c_type, None)
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
                for batch_index, (X1,X2,id,wGII_group, c_type) in enumerate(valLoader):
                    recon1, L1 = env_e(X1.float().to(device))
                    recon2, L2 = env_e(X2.float().to(device))
                    val_Ls.append(L1.float().cpu().numpy())
                    ids.append(id)
                    total_loss = loss_fn(L1, c_type, None)
                    train_loss.append(total_loss.item())
                    val_loss.append(total_loss.item())
                wandb.log({"Valid/loss": np.mean(val_loss),
                           "Epoch":epoch})
            scheduler.step()
            #Plot validation data from last epoch
            visualize_latent(val_Ls, ids, l, epoch,temp)
            if np.mean(val_loss) < best_loss and epoch > 30:
                best_loss = np.mean(val_loss)
                print("current best loss: ", best_loss)
                print("saving model...")
                #torch.save(env_e.state_dict(), "{}/{}_best_model_sup_wgii_3_contrastive_temp_{}.pth".format(feature_util.linux_path, l,temp))
        wandb.finish()