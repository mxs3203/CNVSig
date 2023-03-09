import pandas as pd
import torch
import wandb
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import numpy as np
from tqdm.asyncio import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from SignatureDev.dataloader import CNVImages
from SignatureDev.VariationalAutoEncoder.Model import CNVExtractorVAE


def gaussian_likelihood(x_hat, logscale, x):
    scale = torch.exp(logscale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, scale)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
lr = 1e-4
batch_size = 128
wd = 1e-6
L_to_try = range(4, 15, 1)
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

    env_e = CNVExtractorVAE(encoded_space_dim=l)
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

    num_epochs = 10
    for epoch in tqdm(range(num_epochs)):
        # Train:
        env_e.train()
        train_loss = []
        for batch_index, X in enumerate(trainLoader):
            recon, L, log_scale, kl = env_e(X.float())
            recon_loss = gaussian_likelihood(recon, log_scale, X)
            elbo = (kl - recon_loss).mean()
            total_loss = elbo
            train_loss.append(total_loss.item())
            optim.zero_grad()
            total_loss.backward()
            optim.step()
        wandb.log({"Train/loss": np.mean(train_loss),
                   "Epoch": epoch})
        # Valid
        env_e.eval()
        val_loss,val_kl,val_log_pxz = [],[],[]
        val_Ls = []
        with torch.no_grad():
            for batch_index, X in enumerate(valLoader):
                recon, L,log_scale,kl = env_e(X.float())
                val_Ls.append(L.float().numpy())
                recon_loss = gaussian_likelihood(recon, log_scale, X)
                elbo = (kl - recon_loss).mean()
                total_loss = elbo
                train_loss.append(total_loss.item())
                val_loss.append(total_loss.item())
            wandb.log({"Valid/loss": np.mean(val_loss),
                       "Epoch":epoch})

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    val_Ls = np.concatenate(val_Ls)
    tsne_results = pd.DataFrame(tsne.fit_transform(val_Ls),columns=["Dim1","Dim2"])
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="Dim1", y="Dim2",
        data=tsne_results,
        legend="full",
        alpha=0.9
    ).set_title("TSNE of L = {}".format(l),fontsize=30)
    plt.show()

    wandb.finish()