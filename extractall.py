import pandas as pd

brca = pd.read_csv('brca.csv')
gtruth = pd.read_csv('samples_fileid_tumor.tsv', sep = "\t", index_col = 0)

genenamerow = brca.iloc[:, 0].rename("Gene stable ID")

cols = list(brca)

a = 1
for i in cols:
    if i != "Unnamed: 0":
        if gtruth.loc[i][0] == True:
            label = "m"
        else:
            label = "b"
        fn = "examples/example-" + str(a) + "-" + label + ".csv"
        (pd.concat([genenamerow, brca[i]], axis = 1)).to_csv(fn, index = False)
        a += 1