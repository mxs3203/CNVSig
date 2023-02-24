import glob
import pandas as pd

IDs = []
for ids in glob.glob("data/output/compute_features/*"):
    if len(glob.glob(ids+ "/*.pickle")) == 22:
        IDs.append(ids.split("/")[-1])

pd.DataFrame(IDs).to_csv("data/output/IDs_with_all_chr.csv")