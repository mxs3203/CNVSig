import glob
import os

import pandas as pd
from tqdm.asyncio import tqdm

from feature_util import readPickle, savePickle, readAscatSavePickle, computeFeatures, makeFilesForEachSampleAndChr

ruleorder: csv_to_pickle_filter > pickle_to_files > compute_features

rule all:
    input:
        "data/output/compute_features/"

rule csv_to_pickle_filter:
    input:
        "data/input/hmf_ascat.csv"
    output:
        "data/output/csv_to_pickle_filter/hmf_ascat.obj"
    run:
        readAscatSavePickle(input[0], output[0])

rule pickle_to_files:
    input:
        rules.csv_to_pickle_filter.output
    output:
        "data/output/pickle_to_files/"
    run:
        df = readPickle(input[0])
        makeFilesForEachSampleAndChr(df, output[0])



rule compute_features:
    input:
        rules.pickle_to_files.output
    output:
        directory("data/output/compute_features/"), "data/output/merged_features.pickle"
    benchmark:
        "benchmarks/ascatfeatures.csv"
    run:
        os.mkdir(output[0])
        for id in tqdm(glob.glob(input[0] + "/*")): # all sample folder
            id_folder = "{}/{}/".format(output[0], id.split("/")[3])
            os.mkdir(id_folder)
            for chr_df in glob.glob(id + "/*.pickle"):
                df = readPickle(chr_df)
                if not df.empty:
                    df = computeFeatures(df)
                    file_name = chr_df.split("/")[4]
                    savePickle(df, "{}/{}".format(id_folder, file_name))
                else:
                    print("DF empty for {} {} ".format(id, chr_df))
