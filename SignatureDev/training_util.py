import pandas as pd
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import feature_util


def visualize_latent(val_Ls,ids,l,epoch):
    dim_reducer = UMAP(n_components=2, n_neighbors=70, min_dist=1.9, spread=2.0)
    val_Ls = pd.DataFrame(np.concatenate(val_Ls), columns=["L{}".format(i) for i in range(0,l)])
    ids = np.concatenate(ids)
    tsne_results = pd.DataFrame(dim_reducer.fit_transform(val_Ls),columns=["Dim1","Dim2"])
    tsne_results['ID'] = ids
    val_Ls['ID'] = ids
    val_Ls['UMAP1'] = tsne_results['Dim1']
    val_Ls['UMAP2'] = tsne_results['Dim2']
    features = pd.read_csv("{}/data/output/all_features_grouped_by_id.csv".format(feature_util.mac_path))
    wgii = pd.read_csv("{}/data/input/hmf_wgii.csv".format(feature_util.mac_path))
    val_Ls = pd.merge(val_Ls, wgii, on="ID", copy=False)
    val_Ls = pd.merge(val_Ls, features, on="ID", copy=False)
    #print(np.shape(tsne_results))
    #print(tsne_results)
    fig, ax = plt.subplots(nrows=3,ncols=4, figsize=(20, 20))
    sns.scatterplot(
        x="UMAP1", y="UMAP2",hue="wGII",
        data=val_Ls,
        legend="auto",
        alpha=0.9,ax=ax[0,0]
    ).set_title("UMAP of L = {}".format(l),fontsize=10)
    sns.scatterplot(
        x="UMAP1", y="UMAP2",hue="wFLOH",
        data=val_Ls,
        legend="auto",
        alpha=0.9,ax=ax[0,1]
    ).set_title("UMAP of L = {}".format(l),fontsize=10)
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="cn",
        data=val_Ls,
        legend="auto",
        alpha=0.9, ax=ax[0,2]
    ).set_title("UMAP of L = {}".format(l), fontsize=10)
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="logR",
        data=val_Ls,
        legend="auto",
        alpha=0.9, ax=ax[0,3]
    ).set_title("UMAP of L = {}".format(l), fontsize=10)
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="log10_distanceToNearestCNV",
        data=val_Ls,
        legend="auto",
        alpha=0.9, ax=ax[1, 0]
    ).set_title("UMAP of L = {}".format(l), fontsize=10)
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="changepoint",
        data=val_Ls,
        legend="auto",
        alpha=0.9, ax=ax[1, 1]
    ).set_title("UMAP of L = {}".format(l), fontsize=10)
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="log10_segmentSize",
        data=val_Ls,
        legend="auto",
        alpha=0.9, ax=ax[1, 2]
    ).set_title("UMAP of L = {}".format(l), fontsize=10)
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="loh",
        data=val_Ls,
        legend="auto",
        alpha=0.9, ax=ax[1, 3]
    ).set_title("UMAP of L = {}".format(l), fontsize=10)
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="allelicImbalance",
        data=val_Ls,
        legend="auto",
        alpha=0.9, ax=ax[2, 0]
    ).set_title("UMAP of L = {}".format(l), fontsize=10)
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="log10_distToCentromere",
        data=val_Ls,
        legend="auto",
        alpha=0.9, ax=ax[2, 1]
    ).set_title("UMAP of L = {}".format(l), fontsize=10)
    sns.scatterplot(
        x="UMAP1", y="UMAP2", hue="replication_timing",
        data=val_Ls,
        legend="auto",
        alpha=0.9, ax=ax[2, 2]
    ).set_title("UMAP of L = {}".format(l), fontsize=10)
    plt.savefig("{}/SignatureDev/Plots/{}/{}.png".format(feature_util.mac_path, l, epoch))
    # plt.show()