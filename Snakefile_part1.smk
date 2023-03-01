import pandas as pd

from feature_util import readPickle, readAscatSavePickle, makeFilesForEachSampleAndChr
import pandas as pd

from feature_util import readPickle, readAscatSavePickle, makeFilesForEachSampleAndChr

ruleorder: csv_to_pickle_filter > pickle_to_files

IDs = pd.read_csv("data/input/ID_list.csv").ID.to_list()
features = ['cn', 'log10_distanceToNearestCNV', 'logR', 'changepoint', 'log10_segmentSize',
            'loh', 'allelicImbalance', 'log10_distToCentromere', 'replication_timing']
#IDs = IDs[0:25]

rule all:
    input: expand("data/output/pickle_to_files/{sample}/", sample=IDs)

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