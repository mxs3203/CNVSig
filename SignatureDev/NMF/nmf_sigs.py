import glob

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import NMF
import seaborn as sb
import matplotlib.pyplot as plt

from feature_util import readPickle

features = ['cn', 'log10_distanceToNearestCNV', 'logR', 'changepoint', 'log10_segmentSize',
            'loh', 'allelicImbalance', 'log10_distToCentromere', 'replication_timing']
#IDs = IDs[0:25]

valid_IDs = []
for file in glob.glob("/home/mateo/pytorch_docker/CNVSig/data/output/make_square_images_/*.pickle"):
    df = readPickle(file)
    if np.shape(df) == (9, 9, 22):
        # split path by / and take the last which is ID.pickle and then split by . to get ID only
        valid_IDs.append(file.split("/")[-1].split(".")[0])
print(len(valid_IDs))




all_data = readPickle("/home/mateo/pytorch_docker/CNVSig/data/output/merged_features.pickle")
all_data = all_data[all_data['ID'].isin(valid_IDs)]  # remove the ones that do not have images
all_data = all_data[features]  # take only features
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(all_data)
for k in range(3, 20):
    model = NMF(n_components=k, init='random', verbose=0, max_iter=50)
    W = model.fit_transform(X)
    print(k,model.reconstruction_err_)
    H = pd.DataFrame(model.components_)
    H.columns = features
    H.index = range(1,k+1)
    H['S'] = H.index
    #print(H)
    H = H.melt(id_vars='S')
    #print(H)
    fig, ax1 = plt.subplots(figsize=(10, 10))
    sb.barplot(data=H, x="S", y="value", hue='variable', ax=ax1)
    sb.despine(fig)
    plt.show()
