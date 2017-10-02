"""
Create a data set containing provider-level procedure summaries and
some additional provider-level variables.  The dataset is saved as a
compressed CSV format file.
"""

import pandas as pd
import numpy as np
import gzip

# The CMS utilization file name
util_file = "Medicare_Provider_Util_Payment_PUF_CY2015.txt.gz"

# The CMS medicare part D file name
med_file = "PartD_Prescriber_PUF_NPI_Drug_15.txt.gz"

# The file name to use for the output file.
out_file = "Medicare_Provider_Util_Payment_PUF_CY2015.txt.gz"

# Get all the NPI numbers from the utilization file
npi_util = []
with gzip.open(util_file, "rb") as fid:
    next(fid) # skip header
    next(fid) # skip header
    for line in fid:
        line = line.decode("latin1")
        f = line.split("\t")
        npi_util.append(f[0])

# Get all the NPI numbers from the prescription file
npi_drugs = []
with gzip.open(med_file, "rb") as fid:
    next(fid) # skip header
    for line in fid:
        line = line.decode("latin1")
        f = line.split("\t")
        npi_drugs.append(f[0])

isect = np.asarray(list(set(npi_util) & set(npi_drugs)), dtype=np.int64)
isect = np.sort(isect)

df = pd.read_csv(out_file, skiprows=[1], delimiter="\t")

df = df.loc[df.npi.isin(isect), :]


dx = df.loc[:, ["npi", "nppes_entity_code", "provider_type", "hcpcs_code", "line_srvc_cnt"]]
dx = dx.groupby(["npi", "hcpcs_code"]).agg({"nppes_entity_code": "first", "provider_type": "first",
                                            "line_srvc_cnt": np.sum})
dx = dx.reset_index()
dx.to_csv("2015_utilization_reduced.csv.gz", compression="gzip")
