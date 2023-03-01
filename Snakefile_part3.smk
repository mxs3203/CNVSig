import glob
import numpy as np

from feature_util import readPickle

features = ['cn', 'log10_distanceToNearestCNV', 'logR', 'changepoint', 'log10_segmentSize',
            'loh', 'allelicImbalance', 'log10_distToCentromere', 'replication_timing']
#IDs = IDs[0:25]

valid_IDs = []
for file in glob.glob("data/output/make_square_images/*.pickle"):
    df = readPickle(file)
    if np.shape(df) == (9,9,22):
        # split path by / and take the last which is ID.pickle and then split by . to get ID only
        valid_IDs.append(file.split("/")[-1].split(".")[0])
print(len(valid_IDs))

rule all:
    input: "data/output/test.test"

rule test:
    input: expand("data/output/make_sqaure_images/{sample}.pickle", sample=valid_IDs)
    output: "data/output/test.test"
    run:
        print(input)
