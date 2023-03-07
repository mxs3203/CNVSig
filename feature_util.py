import glob
import os
import warnings

import numpy as np
from tqdm.asyncio import tqdm

np.seterr(divide='ignore')
import pandas as pd
import pickle as pk

hg19 = pd.read_csv("data/input/hg19.chrom.sizes.csv", sep=",")
chrom_centromere = pd.read_csv("/home/mateo/pytorch_docker/CNVSig/data/input/chrom_centromere_info.csv", sep=",")
rep_time = pd.read_csv("/home/mateo/pytorch_docker/CNVSig/data/input/Encode_replication_timing.tsv", sep="\t")


def readAscatSavePickle(input, output):
    df = pd.read_csv(input, delimiter=",")
    print("Before filtering: ", np.shape(df))
    df = df[~df.Chr.isin([23, 24])]
    df = df.query("ACF < 0.99 and ACF > 0.1")
    # Do we want to remove 'normal' segments?
    # df = df.query("cn != 2 and Ploidy != 2.0")

    # deal with outliers
    df.loc[df["nAraw"] < 0, "nAraw"] = 0.0
    df.loc[df["nBraw"] < 0, "nBraw"] = 0.0
    df.loc[df["cn"] < 0, "cn"] = 0.0
    df.loc[df["cn"] > 10, "cn"] = 10.0
    print("After filtering: ", np.shape(df))
    df['middleOfSegment'] = (df['End'] + df['Start'])/2
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
                    distance = np.log10(np.abs(df.iloc[j]['Start'] - df.iloc[i]['End'] + 1))
                    distances.append(distance)
                    break
    # print(distances)
    df['log10_distanceToNearestCNV'] = distances
    #print(df)
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
        df.loc[df["logR"] <= -8, "logR"] = -8.0
        df.loc[df["logR"] >= 8, "logR"] = 8.0
        df['changepoint'] = changePoint(df)
        df['log10_segmentSize'] = np.log10(df['End'] - df['Start'] + 1)
        df['loh'] = df.apply(lambda row: label_loh(row), axis=1)
        df['allelicImbalance'] = df.apply(lambda row: allelicImbalance(row), axis=1)
    return df


def generate_short_arm_bins(bin_row):
    bins = pd.DataFrame()
    previous = 0
    for short_bins in range(0, int(bin_row['number_of_bins_on_short_arm'].values[0])):
        bins = pd.concat([bins, pd.DataFrame({'from':previous, 'to':previous + bin_row['short_arm_bin_size'], 'arm':'short'})])
        previous = previous + bin_row['short_arm_bin_size']
    return bins

def generate_long_arm_bins(bin_row):
    bins = pd.DataFrame()
    previous = bin_row['cent_end'].values[0]
    for short_bins in range(0, int(bin_row['number_of_bins_on_long_arm'].values[0])):
        bins = pd.concat([bins, pd.DataFrame({'from': previous, 'to': previous + bin_row['long_arm_bin_size'], 'arm':'long'})])
        previous = previous + bin_row['long_arm_bin_size']
    return bins

def scaleMinMax(x,xmin,xmax,f):
    if f in ['loh', 'allelicImbalance']:
        return x
    else:
        return (x-xmin)/(xmax-xmin)
pd.options.mode.chained_assignment = None
def checkIfEmpty(tmp, feature, minMax_of_feature):
    if not tmp.empty:
        m = tmp[feature].mean()
        s = tmp[feature].std()
        if tmp[feature].isin([-np.inf]).values.sum() != 0:
            tmp[feature].replace([-np.inf], 0, inplace=True)
            print(tmp[feature].mean())
        return scaleMinMax(tmp[feature].mean(),minMax_of_feature[0],minMax_of_feature[1],feature)
    else:
        return 0

def computeMeanValueBasedOnBinAndFeature(chr_specific_features, bin_from, bin_to,arm, feature, minMax_of_feature):
    if feature in ['loh', 'allelicImbalance']:
        tmp = chr_specific_features[chr_specific_features['middleOfSegment'].between(bin_from, bin_to)]
        return checkIfEmpty(tmp, feature, minMax_of_feature)
    else:
        tmp = chr_specific_features[chr_specific_features['middleOfSegment'].between(bin_from, bin_to)]
        return checkIfEmpty(tmp, feature, minMax_of_feature)

def makeSquareImage(chr_data, chr_specific_bins, feature, minMax_of_feature):
    bin_values = []
    for index, row in chr_specific_bins.iterrows():
        val = computeMeanValueBasedOnBinAndFeature(chr_data, row['from'], row['to'],row['arm'], feature, minMax_of_feature)
        bin_values.append(val)

    return np.array(bin_values)

def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])


# df = readPickle("data/output/merged_features.pickle")
# print(df['log10_distanceToNearestCNV'].isin([-np.inf]).values.sum())
# df.replace([np.inf, -np.inf], 0.0, inplace=True)
# print(min(df['log10_distanceToNearestCNV']))

# minMax_of_feature = pd.read_csv("data/output/min_max_features.csv")
# features = ['cn', 'log10_distanceToNearestCNV', 'logR', 'changepoint', 'log10_segmentSize',
#             'loh', 'allelicImbalance', 'log10_distToCentromere', 'replication_timing']
# order_of_chromosomes = [4,7,2,5,6,13,3,8,9,18,12,1,10,11,14,22,19,17,20,16,15,21]
# all_features = []
# files_in_sample = glob.glob("data/output/compute_features/CPCT02010037T" + "/*.pickle")
# if len(files_in_sample) == 22:
#     path_to_chr_files = files_in_sample[0]
#     path_to_chr_files = "/".join(path_to_chr_files.split("/")[:-1])
#     for f in features:
#         bins_for_specific_chr = [] # make empty list
#         for specific_chr in order_of_chromosomes: # for every chr with specific order
#             chr_data = readPickle("{}/{}.pickle".format(path_to_chr_files,specific_chr)) # read file data/output/compute_fatures/ID/4.pickle
#             chr_specific_bins = pd.read_csv("{}/{}.csv".format("data/output/chromosome_bins_def/", specific_chr)) # read bin file for that chr data/output/chrom_bin_def/4.csv
#             specific_feature_bin_row = makeSquareImage(chr_data, chr_specific_bins, f,minMax_of_feature[f]) # returns 22 values representing all bins for specific features and chr
#             bins_for_specific_chr.append(specific_feature_bin_row) # I append each row (chr) here
#         assert np.shape(bins_for_specific_chr) == (22, 22) # after all chrom run this should be 22,22
#         all_features.append(bins_for_specific_chr) # append feature profile to a list
#     total = np.dstack(all_features) # stack all feature profiles as depth
#     assert np.shape(total) == (22, 22, 9)
#
# else:
#     savePickle(-1, "data/output/make_square_images/{}.pickle".format("data/output/compute_features/CPCT02010003T").split("/")[3])