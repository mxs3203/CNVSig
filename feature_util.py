import os
import warnings

from tqdm.asyncio import tqdm
import numpy as np

np.seterr(divide='ignore')
import pandas as pd
import pickle as pk

hg19 = pd.read_csv("data/input/hg19.chrom.sizes.txt", sep=",")
chrom_centromere = pd.read_csv("data/input/chrom_centromere_info.csv", sep=",")
rep_time = pd.read_csv("data/input/Encode_replication_timing.tsv", sep="\t")


def readAscatSavePickle(input, output):
    df = pd.read_csv(input, delimiter=",")
    print("Before filtering: ", np.shape(df))
    df = df[~df.Chr.isin([23, 24])]
    df = df.query("ACF < 0.99 and ACF > 0.1")
    # Do we want to remove 'normal' segments?
    # df = df.query("cn != 2 and Ploidy != 2.0")

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


def makeFilesForEachSampleAndChr(df, IDs, out_folder):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    for id in tqdm(IDs):
        if not os.path.exists("{}/{}".format(out_folder, id)):
            os.mkdir("{}/{}".format(out_folder, id))
        for c in range(1, 23, 1):  # up to 22
            tmp_df = df.loc[df["ID"] == id]
            tmp_df = tmp_df.loc[df["Chr"] == c]
            savePickle(tmp_df, "{}/{}/{}.pickle".format(out_folder, id, c))


def distanceToCentromere(row):
    chrom_centromere_start = chrom_centromere[chrom_centromere['chrom'] == row['Chr']]['chromStart'].values
    chrom_centromere_end = chrom_centromere[chrom_centromere['chrom'] == row['Chr']]['chromEnd'].values
    middle_centromere = (chrom_centromere_end + chrom_centromere_start) / 2
    middle_segment = (row['End'] + row['Start']) / 2
    if middle_segment < middle_centromere:  # long arm
        return np.log10(np.abs(chrom_centromere_start - row['End'] + 1))[0]
    else:  # short arm
        return np.log10(np.abs(row['Start'] - chrom_centromere_end + 1))[0]


def distanceToClosestCNV(df):
    n_row = np.shape(df)[0]
    distances = [0]
    for i in range(1, n_row):  # skip first, distance is 0
        if df.iloc[i]['cn'] == 2:  # if cn is normal:
            distances.append(0)
        else:  # find distance to closes CNV
            for j in range(i, n_row):  # iter from i until next cnv
                if df.iloc[j]['cn'] != 2:  # if cn is not normal
                    distance = np.log10(np.abs(df.iloc[j]['Start'] - df.iloc[i]['End'] + 0.0001))
                    distances.append(distance)
                    break
    # print(distances)
    df['log10_distanceToNearestCNV'] = distances
    print(df)
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
        start, end = row['Start'], row['End']
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
        df['logR'] = np.log2(((df['nAraw'] + df['nBraw'] + 0.00001) / (df['Ploidy'] + 0.0001)))
        df['changepoint'] = changePoint(df)
        df['log10_segmentSize'] = np.log10(df['End'] - df['Start'] + 1)
        df['loh'] = df.apply(lambda row: label_loh(row), axis=1)
        df['allelicImbalance'] = df.apply(lambda row: allelicImbalance(row), axis=1)
    return df


def computeMeanValueBasedOnBinAndFeature(chr_specific_features, quantiles, f, bin, num_bins):
    if bin == 0:
        tmp = chr_specific_features[chr_specific_features[f] < quantiles.iloc[bin][f]]
    elif bin == (num_bins - 1):  # zero indexed
        tmp = chr_specific_features[chr_specific_features[f] > quantiles.iloc[bin][f]]
    else:  # feature f is in between two quantiles
        tmp = chr_specific_features[
            chr_specific_features[f].between(quantiles.iloc[bin][f], quantiles.iloc[bin + 1][f])]

    if not tmp.empty:
        return tmp[f].mean()
    else:
        return 0


def makeLongImage(chr_specific_features, quantiles, features, num_bins):
    names = []
    values = []
    for f in features:  # for all features
        for bin in range(num_bins):  # for all bins,9
            # find chr spec df which is within the quantile range(bin)
            values.append(computeMeanValueBasedOnBinAndFeature(chr_specific_features, quantiles, f, bin, num_bins))
            names.append("{}_{}".format(f, bin))

    binned_features = pd.DataFrame(values)
    binned_features.index = names
    return binned_features


def makeSquareImage(chr_specific_features, quantiles, features, num_bins):
    chr_matrix = []
    for bin in range(num_bins):
        row = []
        for f in features:
            val = computeMeanValueBasedOnBinAndFeature(chr_specific_features, quantiles, f, bin, num_bins)
            row.append(val)
        chr_matrix.append(row)
    chr_matrix = np.array(chr_matrix)
    assert np.shape(chr_matrix) == (num_bins, len(features))
    return np.array(chr_matrix)
