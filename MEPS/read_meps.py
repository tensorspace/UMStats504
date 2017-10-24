import pandas as pd
import gzip

fid = open("h171su_vars.txt")
colnames, colspecs = [], []
for line in fid:
    line = line.replace("$", "")
    toks = line.split()
    colnames.append(toks[1])
    s = int(toks[0]) - 1
    w = int(float(toks[2]))
    colspecs.append((s, s+w))

with gzip.open("h171.dat.gz", "rt") as fid:
    meps = pd.read_fwf(fid, colspecs=colspecs, names=colnames)

meps = meps.loc[meps.PERWT14F > 0]
