import torch
import wandb
from torch import Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from tqdm.asyncio import tqdm

from SignatureDev.AutoEncoder.Model import CNVExtractor
from SignatureDev.AutoEncoder.dataloader import CNVImage

lr = 1e-4
batch_size = 512
wd = 1e-2
L_size = 7

dataset = CNVImage("/home/mateo/pytorch_docker/CNVSig/data/output/make_square_images_/")
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = DataLoader(dataset, batch_size=batch_size,sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=batch_size,sampler=valid_sampler)

env_e = CNVExtractor(encoded_space_dim=L_size)
wandb.init(
    project="CNVSigs",
    config={
        "learning_rate": lr,
        "architecture": "CNN",
        "dataset": "make_square_images_",
        "batch_size": batch_size,
        "weight_decay": wd,
        "L_size":L_size
    }
)

loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(env_e.parameters(), lr=lr, weight_decay=wd)

num_epochs = 1000
for epoch in tqdm(range(num_epochs)):
    # Train:
    env_e.train()
    for batch_index, X in enumerate(train_loader):
        recon, L = env_e(X.float())
        #loss = loss_fn(recon, X.float())
        loss = ((X - recon)**2).sum() + env_e.kl
        optim.zero_grad()
        loss.backward()
        optim.step()
        wandb.log({"Train/loss": loss.item(),
                   "Train/KL": env_e.kl})
    # Valid
    env_e.eval()
    with torch.no_grad():
        for batch_index, X in enumerate(validation_loader):
            recon, L = env_e(X.float())
            #print(L)
            #loss = loss_fn(recon, X.float())
            loss = ((X - recon) ** 2).sum() + env_e.kl
            wandb.log({"Valid/loss": loss.item(),
                       "Valid/KL": env_e.kl})
wandb.finish()