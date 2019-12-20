
import glob,os

import pandas as pd
import numpy as np

files = glob.glob("study_results[*.csv")

results = []

for file in files:
    df = pd.read_csv(file,header=0,index_col=False)
    results.extend(df.values.tolist())

print(results)

loss = ["no_da","ddc_mmd","jan","dan","coral","sl"]

datasets = ["amazon","webcam","dslr"]
summary = []
df = pd.DataFrame(results).iloc[:,1:]

for n1 in datasets:
    for n2 in datasets:
        dataset = []
        for l in loss:
            results_l = df[df[1].str.contains(l)]
            index = results_l[2].str.contains(n1) & results_l[3].str.contains(n2)
            new = results_l[index]
            if new.size >0:
                mean = new.mean().values[0]
                std = new .std().values[0]
                dataset.extend([mean, std])
        summary.append([n1]+[n2]+dataset)


df = pd.DataFrame(summary)
df.loc['mean'] = df.mean()
df.to_csv("merged_results.csv")