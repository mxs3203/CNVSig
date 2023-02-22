import glob
import os

import pandas as pd
from tqdm.asyncio import tqdm

from feature_util import readPickle, savePickle, readAscatSavePickle, computeFeatures, makeFilesForEachSampleAndChr

ruleorder: csv_to_pickle_filter > pickle_to_files > compute_features
IDs = pd.read_csv("data/input/ID_list.csv").ID.to_list()
#IDs = IDs[0:5]


rule all:
    input:
        expand("data/output/compute_features/{sample}", sample=IDs)

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
        directory(expand("data/output/pickle_to_files/{sample}/", sample=IDs))
    params:
        out_folder = "data/output/pickle_to_files"
    run:
        df = readPickle(input[0])
        makeFilesForEachSampleAndChr(df,IDs, params.out_folder)


rule compute_features:
    input:
        directory("data/output/pickle_to_files/{sample}/")
    output:
        directory("data/output/compute_features/{sample}/")
    run:
        os.mkdir(str(output))
        for chr_df in glob.glob(str(input)+ "/*.pickle"):
            df = readPickle(chr_df)
            if not df.empty:
                df = computeFeatures(df)
                file_name = chr_df.split("/")[-1]
                savePickle(df, "{}/{}".format(output,file_name))
            else:
                print("DF empty for {} {} ".format(id, chr_df))
