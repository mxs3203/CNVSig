import glob
import os

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm

from feature_util import readPickle, savePickle, readAscatSavePickle, computeFeatures, makeFilesForEachSampleAndChr, \
    makeLongImage, makeSquareImage

ruleorder: csv_to_pickle_filter > pickle_to_files > compute_features > merge_all_features > feature_statistic >
         make_long_images > make_square_images

IDs = pd.read_csv("data/input/ID_list.csv").ID.to_list()
features = ['cn', 'log10_distanceToNearestCNV', 'logR', 'changepoint', 'log10_segmentSize',
            'loh', 'allelicImbalance', 'log10_distToCentromere', 'replication_timing']
#IDs = IDs[0:25]

bins_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

rule all:
    input: expand("data/output/compute_features/{sample}/", sample=IDs), "data/output/merged_features.pickle", "data/output/feature_quantiles.pickle", expand("data/output/make_square_images/{sample}.pickle", sample=IDs),expand("data/output/make_long_images/{sample}.pickle", sample=IDs)

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

rule merge_all_features:
    input: "data/output/compute_features/"
    output: "data/output/merged_features.pickle", "data/output/merged_features.csv",
    run:
        total = []
        for folder in tqdm(glob.glob(str(input) + "/*")):
            for chr in glob.glob(str(folder) + "/*.pickle"):
                df = readPickle(chr)
                total.append(df)
        all_data = pd.concat(total)
        savePickle(all_data, "data/output/merged_features.pickle")
        all_data.to_csv("data/output/merged_features.csv")

rule feature_statistic:
    input: "data/output/merged_features.pickle"
    output: "data/output/feature_quantiles.pickle"
    run:
        df = readPickle(str(input))
        quantiles = df[features].quantile(q=bins_quantiles)
        print(quantiles)
        savePickle(quantiles, str(output))

# Make an image of where rows are=features*bin and columns are chromosomes
rule make_long_images:
    input: rules.compute_features.output, rules.feature_statistic.output
    output: "data/output/make_long_images/{sample}.pickle"
    run:
        quantiles = readPickle(str(input[1]))
        all_chr = []
        if len(glob.glob(str(input[0])+ "/*.pickle")) >= 2:
            for specific_chr in glob.glob(str(input[0]) + "/*.pickle"):
                chr_data = readPickle(specific_chr)
                chr_data = chr_data[features]
                chr_data_binned = makeLongImage(chr_data, quantiles, features, len(bins_quantiles))
                all_chr.append(chr_data_binned)
            all_chr = pd.concat(all_chr, axis=1)
            all_chr.columns = range(1, 23)
            savePickle(all_chr, "data/output/make_long_images/{}.pickle".format(str(input[0]).split("/")[3]))
        else:
            savePickle(-1, "data/output/make_long_images/{}.pickle".format(str(input[0]).split("/")[3]))
# Make an image of where rows bins, columns are features and depth is chr
rule make_square_images:
    input: rules.compute_features.output, rules.feature_statistic.output
    output: "data/output/make_square_images/{sample}.pickle"
    run:
        quantiles = readPickle(str(input[1]))
        all_chr = []
        if len(glob.glob(str(input[0])+ "/*.pickle")) >= 2:
            for specific_chr in glob.glob(str(input[0]) + "/*.pickle"):
                chr_data = readPickle(specific_chr)
                chr_data = chr_data[features]
                chr_data_binned = makeSquareImage(chr_data, quantiles, features, len(bins_quantiles))
                all_chr.append(chr_data_binned)

            total = np.dstack(all_chr)
            assert np.shape(total) == (len(bins_quantiles), len(features), 22)
            savePickle(total, "data/output/make_square_images/{}.pickle".format(str(input[0]).split("/")[3]))
        else:
            savePickle(-1, "data/output/make_square_images/{}.pickle".format(str(input[0]).split("/")[3]))
        # why is this not -1
        # apperantly last string here is empty string