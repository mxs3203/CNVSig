import pandas as pd
import torch
from umap import UMAP

import wandb
from torch.utils.data import DataLoader
import numpy as np
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from SignatureDev.AutoEncoder.Model import CNVExtractorAE
from SignatureDev.dataloader import CNVImages
from feature_util import readPickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
lr = 8e-3
batch_size = 128
wd = 1e-6
L_to_try = range(5, 16, 1)
dataset = CNVImages("/home/mateo/pytorch_docker/CNVSig/data/output/make_square_images/")
dataset_size = len(dataset.annotation)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size,num_workers=10, shuffle=True)


for l in L_to_try:
    print(l)

    env_e = CNVExtractorAE(encoded_space_dim=l)
    env_e.to(device)

    wandb.init(
        name="{}".format(l),
        project="CNVSigs",
        config={
            "learning_rate": lr,
            "architecture": "CNN",
            "dataset": "make_square_images",
            "batch_size": batch_size,
            "weight_decay": wd,
            "L_size":l
        }
    )

    loss_fn = torch.nn.MSELoss(reduction="mean")
    optim = torch.optim.Adam(env_e.parameters(), lr=lr, weight_decay=wd)

    num_epochs = 15
    for epoch in tqdm(range(num_epochs)):
        # Train:
        env_e.train()
        train_loss = []
        for batch_index, (X,id) in enumerate(trainLoader):
            recon, L = env_e(X.float())
            total_loss = 0
            for depth in range(np.shape(X)[1]): # until 9
                layer_loss = loss_fn(recon[:,depth,:,:], X.float()[:,depth,:,:])
                total_loss += layer_loss
            #total_loss = loss_fn(recon, X.float())
            train_loss.append(total_loss.item())
            optim.zero_grad()
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
                recon, L = env_e(X.float())
                val_Ls.append(L.float().numpy())
                ids.append(id)
                total_loss = 0
                for depth in range(np.shape(X)[1]):  # until 9
                    layer_loss = loss_fn(recon[:, depth, :, :], X.float()[:, depth, :, :])
                    total_loss += layer_loss
                #total_loss = loss_fn(recon, X.float())
                train_loss.append(total_loss.item())
                val_loss.append(total_loss.item())
            wandb.log({"Valid/loss": np.mean(val_loss),
                       "Epoch":epoch})
    #Plot validation data from last epoch
    dim_reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, spread=1.0, metric="cosine")
    val_Ls = np.concatenate(val_Ls)
    ids = np.concatenate(ids)
    tsne_results = pd.DataFrame(dim_reducer.fit_transform(val_Ls),columns=["Dim1","Dim2"])
    tsne_results['ID'] = ids
    #features = readPickle("/home/mateo/pytorch_docker/CNVSig/data/output/merged_features.pickle")
    wgii = pd.read_csv("/home/mateo/pytorch_docker/CNVSig/data/input/hmf_wgii.csv")
    tsne_results = pd.merge(tsne_results,wgii, on="ID", copy=False)
    print(np.shape(tsne_results))
    print(tsne_results)
    fix, ax = plt.subplots(2,1, figsize=(10, 10))
    sns.scatterplot(
        x="Dim1", y="Dim2",hue="wGII",
        data=tsne_results,
        legend="auto",
        alpha=0.9,ax=ax[0]
    ).set_title("UMAP of L = {}".format(l),fontsize=20)
    sns.scatterplot(
        x="Dim1", y="Dim2",hue="wFLOH",
        data=tsne_results,
        legend="auto",
        alpha=0.9,ax=ax[1]
    ).set_title("UMAP of L = {}".format(l),fontsize=20)
    plt.show()




    wandb.finish()