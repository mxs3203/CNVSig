import os
import warnings

from tqdm.asyncio import tqdm
import numpy as np
np.seterr(divide='ignore')
import pandas as pd
import pickle as pk

hg19 = pd.read_csv("data/input/hg19.chrom.sizes.txt", sep=",")
rep_time = pd.read_csv("data/input/Encode_replication_timing.tsv", sep="\t")

def readAscatSavePickle(input, output):
    df = pd.read_csv(input, delimiter=",")
    print("Before filtering: ",np.shape(df))
    df = df[~df.Chr.isin([23,24])]
    df = df.query("ACF < 0.99 and ACF > 0.1")
    # Do we want to remove 'normal' segments?
    df = df.query("cn != 2 and Ploidy != 2.0")

    df.loc[df["nAraw"] < 0, "nAraw"] = 0.0
    df.loc[df["nBraw"] < 0, "nBraw"] = 0.0
    print("After filtering: ", np.shape(df))
    with open(output, 'wb') as f:
        pk.dump(df, f, pk.HIGHEST_PROTOCOL)
        f.close()

def readPickle(input):
    return pd.read_pickle(input)

def savePickle(df, output):
    with open(output, 'wb') as f:
        pk.dump(df, f, pk.HIGHEST_PROTOCOL)
        f.close()

def label_loh(row):
   if row['nA'] == 0 and row['nB'] == 1 and row['cn'] == 1:
       return 1
   if row['nA'] == 1 and row['nB'] == 0 and row['cn'] == 1:
       return 1
   else:
       return 0

def makeFilesForEachSampleAndChr(df,IDs, out_folder):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    for id in tqdm(IDs):
        if not os.path.exists("{}/{}".format(out_folder, id)):
            os.mkdir("{}/{}".format(out_folder, id))
        for c in range(1, 23, 1):
            tmp_df = df.loc[df["ID"] == id]
            tmp_df = tmp_df.loc[df["Chr"] == c]
            savePickle(tmp_df, "{}/{}/{}.pickle".format(out_folder,id,c))

def distanceToCentromere(row):
    location = hg19[hg19['Chr'] == row['Chr']]['location'].values[0]
    if row['Start'] <= location: # long arm
        return np.log10((location - row['End'] + 1))
    else: #short arm
        return np.log10((location - row['Start'] + 1))

def distanceToClosestCNV(df):
    n_row = np.shape(df)[0]
    distance = np.array(df.iloc[range(1, n_row), [3]].values - df.iloc[range(0, n_row - 1), [4]].values)
    distance = np.insert(distance, 0, 0)
    df['log10_distanceToNearestCNV'] = np.log10(np.abs(distance+0.0001))
    return df


def changePoint(df):
    n_row = np.shape(df)[0]
    changepoint = np.array(df.iloc[range(1, n_row), [15]].values - df.iloc[range(0, n_row - 1), [15]].values)
    changepoint = np.insert(changepoint, 0, 0)
    return changepoint

def allelicImbalance(row):
    if row['nA'] != row['nB']:
        return 1
    else:
        return 0

def replicationTiming(row):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        start,end = row['Start'], row['End']
        chr = "chr{}".format(row['Chr'])
        rep_time_tmp = rep_time[rep_time['Chromosome'] == chr]
        rep_time_tmp = rep_time_tmp[rep_time['Position'].between(start, end)]
        if np.shape(rep_time_tmp)[0] == 0:
            return 0
        else:
            return rep_time_tmp['Scaled'].mean()
def computeFeatures(df):
    with np.errstate(all="ignore"):
        df['replication_timing'] = df.apply(lambda row: replicationTiming(row), axis=1)
        df['log10_distToCentromere'] = df.apply(lambda row: distanceToCentromere(row), axis=1)
        df = distanceToClosestCNV(df)
        df['logR'] = np.log2(((df['nAraw'] + df['nBraw']+0.00001) / (df['Ploidy']+0.0001)))
        df['changepoint'] = changePoint(df)
        df['log10_segmentSize'] = np.log10(df['End'] - df['Start'] + 1)
        df['loh'] = df.apply(lambda row: label_loh(row), axis=1)
        df['allelicImbalance'] = df.apply(lambda row: allelicImbalance(row), axis=1)
    return df
