# coding: utf-8
import pandas as pd
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# Fill in missing embarked with S
train["C"] = train["C"].fillna("None")
train["D"] = train["D"].fillna("None")
train["E"] = train["E"].fillna("None")
train["F"] = train["F"].fillna("None")
train["I"] = train["I"].fillna("None")
train["J"] = train["J"].fillna("None")

test["C"] = test["C"].fillna("None")
test["D"] = test["D"].fillna("None")
test["E"] = test["E"].fillna("None")
test["F"] = test["F"].fillna("None")
test["I"] = test["I"].fillna("None")
test["J"] = test["J"].fillna("None")

# Write our changed dataframes to csv.
test.to_csv("./test.csv", index=False)
train.to_csv('./train.csv', index=False)
