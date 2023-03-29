

import numpy as np
import pandas as pd

from feature_util import readPickle
#
tcga_ascat = pd.read_csv("../data/input/ascat_tcga.csv")
hmf_ascat = pd.read_csv("../data/hmf/input/hmf_ascat.csv")
cancer_types = hmf_ascat['TCGA_type'].unique()
tcga_ascat_filtered = tcga_ascat[tcga_ascat['cancer_type'].isin(cancer_types)]
IDs = pd.DataFrame(tcga_ascat_filtered['ID'].unique(), columns=["ID"])
IDs.to_csv("../data/input/ID_list.csv")

a = tcga_ascat_filtered.groupby(["ID", "Chr"]).size()
b = a[a > 30].index.values
c = [item[0] for item in b]
tcga_ascat_filtered_filtered = tcga_ascat_filtered[tcga_ascat_filtered['ID'].isin(c)]

IDs = pd.DataFrame(tcga_ascat_filtered_filtered['ID'].unique(), columns=["ID"])
IDs.to_csv("../data/input/ID_list.csv")