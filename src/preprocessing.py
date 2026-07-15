import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/train.csv")
# Transform
#Plan 1 to change Name into the Title

Name = df["Name"]
print(Name)
df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")
print(df["Title"].unique())
df["Title"] = df["Title"].replace({
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs"
})
print(df["Title"].unique())
rare_titles = [
    "Lady",
    "the Countess",
    "Capt",
    "Col",
    "Don",
    "Dr",
    "Major",
    "Rev",
    "Sir",
    "Jonkheer"
]

df["Title"] = df["Title"].replace(rare_titles, "Rare")
print(df["Title"].value_counts())
print(df.columns.to_list())
# making anothere column Family size
df["Family_Size"] = df["SibSp"]+df["Parch"]+1
print(df["Family_Size"].value_counts())

# Missing_value handelling
from ml_utils import Missing_Summary
print(Missing_Summary(df))
