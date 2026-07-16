import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
def checking_missing():
    Age = df["Age"].copy()
    #For median 
    median_age = Age.median()
    Median_Age = Age.fillna(round(median_age,0))
    #Mean
    mean_age = Age.mean()
    Mean_Age = Age.fillna(round(mean_age))
    #Zero
    print(f"Skew for mean is added  :{Mean_Age.skew():.2f}")
    print(f"Skew for median is added  :{Median_Age.skew():.2f}")
    print(f"Actual Skew : {Age.skew():.2f}")
    #histoogram for each
    datasets = [
    ("Original Age", Age),
    ("Mean Filled", Mean_Age),
    ("Median Filled", Median_Age)
        ]

    for title, data in datasets:


        plt.figure(figsize=(8,4))
        sns.histplot(data, kde=True)
        plt.title(title)
        plt.xlabel("Age")
        plt.savefig(f"{title}_histogram.png")
        plt.close()

        plt.figure(figsize=(8,2))
        sns.boxplot(x=data)
        plt.title(title)
        plt.savefig(f"{title}_boxplot.png")
        plt.close()
    print(Age.describe())

    print(Mean_Age.describe())

    print(Median_Age.describe())    
checking_missing()