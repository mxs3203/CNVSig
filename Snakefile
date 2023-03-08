import glob
import os

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm

from feature_util import readPickle, savePickle, readAscatSavePickle, computeFeatures, makeFilesForEachSampleAndChr, \
    makeSquareImage, generate_short_arm_bins, generate_long_arm_bins, minMax

ruleorder: csv_to_pickle_filter > pickle_to_files > compute_features > compute_bin_sizes > make_square_images
NUM_BINS = 22
IDs = pd.read_csv("data/input/ID_list.csv").ID.to_list()
features = ['cn', 'log10_distanceToNearestCNV', 'logR', 'changepoint', 'log10_segmentSize',
            'loh', 'allelicImbalance', 'log10_distToCentromere', 'replication_timing']
#IDs = IDs[0:10]

features_that_need_quantiles = ['log10_distanceToNearestCNV', 'logR', 'changepoint', 'log10_segmentSize',
                                'log10_distToCentromere',
                                'replication_timing']

order_of_chromosomes = [4,7,2,5,6,13,3,8,9,18,12,1,10,11,14,22,19,17,20,16,15,21]


rule all:
    input:
        expand("data/output/compute_features/{sample}/", sample=IDs),
        expand("data/output/make_square_images_/{sample}.pickle", sample=IDs)

rule csv_to_pickle_filter:
    input: "data/input/hmf_ascat.csv"
    output: "data/output/csv_to_pickle_filter/hmf_ascat.obj"
    run: readAscatSavePickle(input[0], output[0])

rule pickle_to_files:
    input: rules.csv_to_pickle_filter.output
    output: directory(expand("data/output/pickle_to_files/{sample}/", sample=IDs))
    params: out_folder="data/output/pickle_to_files"
    run:
        df = readPickle(input[0])
        makeFilesForEachSampleAndChr(df, IDs, params.out_folder)

rule compute_features:
    input: "data/output/pickle_to_files/{sample}/"
    output: directory("data/output/compute_features/{sample}/")
    run:
        if not os.path.exists(str(output)):
            os.mkdir(str(output))
        for chr_df in tqdm(glob.glob(str(input) + "/*.pickle")):
            df = readPickle(chr_df)
            if not df.empty:
                df = computeFeatures(df)
                file_name = chr_df.split("/")[-1]
                savePickle(df, "{}/{}".format(str(output), file_name))

rule compute_bin_sizes:
    input: "data/input/chrom_centromere_info.csv", "data/input/hg19.chrom.sizes.csv"
    output: "data/output/chromosome_bins_info.csv", directory("data/output/chromosome_bins_def/"), "data/output/min_max_features.csv"
    run:
        # Find minMax of each feature and merge all features
        print("Merging all sample features into a single file...")
        total = []
        for folder in tqdm(glob.glob("data/output/compute_features" + "/*")):
            for chr in glob.glob(str(folder) + "/*.pickle"):
                df = readPickle(chr)
                total.append(df)
        all_data = pd.concat(total)
        all_data.replace([-np.inf],0.0,inplace=True)
        # summarized_loh = all_data.groupby(['ID', 'Chr'])['loh'].apply(lambda x: (x == 1).sum()).reset_index(name='count')
        # summarized_ai = all_data.groupby(['ID', 'Chr'])['allelicImbalance'].apply(lambda x: (x == 1).sum()).reset_index(name='count')
        # print(summarized_loh['count'].quantile(q=[0.90,0.95,0.99,0.99]))
        # print(summarized_ai['count'].quantile(q=[0.90,0.95,0.99,0.99]))
        savePickle(all_data, "data/output/merged_features.pickle")
        all_data.to_csv("data/output/merged_features.csv")

        print("Computing minMax of each feature...")
        feature_min_max = all_data[features].apply(minMax)
        # from empircal data :) 
        feature_min_max['loh'] = feature_min_max['loh'].replace([1], 3.6e-06)
        feature_min_max['allelicImbalance'] = feature_min_max['allelicImbalance'].replace([1], 6.13e-05)
        print(feature_min_max)
        feature_min_max.to_csv("data/output/min_max_features.csv")

        print("Computing Bin sizes for each chr...")
        # this is actual compute bin sizes
        centromeres = pd.read_csv(input[0])
        chrom_size_df = pd.read_csv(input[1],sep=",",names=None)
        total = []
        for chr in tqdm(range(1,23)):# for chr 1 to 22
            cent_start = centromeres.loc[centromeres['chrom'] == chr]['chromStart'].values[0]
            cent_end = centromeres.loc[centromeres['chrom'] == chr]['chromEnd'].values[0]
            chrom_size = chrom_size_df[chrom_size_df['Chr'] == chr]['location'].values[0]
            total.append([chr, cent_end, cent_start, chrom_size, cent_start-1, chrom_size-cent_end,  (chrom_size-cent_end) - (cent_start-1)])

        total = pd.DataFrame(total, columns=['chr','cent_start','cent_end','chrom_size','short_arm_l', 'long_arm_l', 'diff_long_short'])
        total['ratio'] = total['chrom_size']/total['long_arm_l']
        total['number_of_bins_on_long_arm'] = round(NUM_BINS/total['ratio'])
        total['number_of_bins_on_short_arm'] = NUM_BINS - total['number_of_bins_on_long_arm']
        total['short_arm_bin_size'] = total['cent_start']/total['number_of_bins_on_short_arm']
        total['long_arm_bin_size'] = (total['chrom_size'] - total['cent_end'])/total['number_of_bins_on_long_arm']
        total.to_csv("data/output/chromosome_bins_info.csv")
        print("Exporting chr specific bin definition to a folder...")
        os.mkdir("data/output/chromosome_bins_def")
        for chr in tqdm(range(1, 23)):
            bin_df = total[total['chr'] == chr]
            bin_values = pd.concat([generate_short_arm_bins(bin_df), generate_long_arm_bins(bin_df)])
            bin_values.to_csv("data/output/chromosome_bins_def/{}.csv".format(chr))

# Make an image of where rows bins, columns are features and depth is chr
rule make_square_images:
    input: rules.compute_features.output, rules.compute_bin_sizes.output[1], rules.compute_bin_sizes.output[2]
    output: "data/output/make_square_images_/{sample}.pickle"
    run:
        minMax_of_feature = pd.read_csv(input[2])
        all_features = []
        files_in_sample = glob.glob(str(input[0]) + "/*.pickle")
        if len(files_in_sample) == 22:
            path_to_chr_files = files_in_sample[0]
            path_to_chr_files = "/".join(path_to_chr_files.split("/")[:-1])
            for f in features:
                bins_for_specific_chr = [] # make empty list
                for specific_chr in order_of_chromosomes: # for every chr with specific order
                    chr_data = readPickle("{}/{}.pickle".format(path_to_chr_files,specific_chr)) # read file data/output/compute_fatures/ID/4.pickle
                    chr_specific_bins = pd.read_csv("{}/{}.csv".format(input[1], specific_chr)) # read bin file for that chr data/output/chrom_bin_def/4.csv
                    specific_feature_bin_row = makeSquareImage(chr_data, chr_specific_bins, f,minMax_of_feature[f]) # returns 22 values representing all bins for specific features and chr
                    bins_for_specific_chr.append(specific_feature_bin_row) # I append each row (chr) here
                assert np.shape(bins_for_specific_chr) == (22,22) # after all chrom run this should be 22,22
                all_features.append(bins_for_specific_chr) # append feature profile to a list
            total = np.dstack(all_features) # stack all feature profiles as depth
            assert np.shape(total) == (22, 22, 9)
            savePickle(total, "data/output/make_square_images_/{}.pickle".format(str(input[0]).split("/")[3]))
        else:
            savePickle(-1, "data/output/make_square_images_/{}.pickle".format(str(input[0]).split("/")[3]))
        # why is this not -1
        # apperantly last string here is empty string



# valid_IDs = []
# for file in glob.glob("data/output/make_square_images_/*.pickle"):
#     df = readPickle(file)
#     if np.shape(df) == (9,9,22):
#         # split path by / and take the last which is ID.pickle and then split by . to get ID only
#         valid_IDs.append(file.split("/")[-1].split(".")[0])