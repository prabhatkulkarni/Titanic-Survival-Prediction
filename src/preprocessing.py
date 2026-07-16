import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/raw/train.csv")
# Transform
#col here is which column tile extraction is done
def extract_title(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()

    df["Title"] = df[col].str.extract(
        r",\s*([^.]*)\.",
        expand=False
    )

    df["Title"] = df["Title"].replace({
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    })

    rare_titles = [
        "Lady", "the Countess", "Capt", "Col",
        "Don", "Dr", "Major", "Rev",
        "Sir", "Jonkheer"
    ]

    df["Title"] = df["Title"].replace(rare_titles, "Rare")

    df.drop(columns=[col], inplace=True)

    return df
# making anothere column Family size
def create_family_size(df : pd.DataFrame ) -> pd.DataFrame:
    df = df.copy()
    df["Family_Size"] = df["SibSp"]+df["Parch"]+1
    df.drop(columns=["SibSp","Parch"],inplace=True)
    return df
def impute_age(df :pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    return df
def drop_columns(df: pd.DataFrame, col:list[ str]) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=col)
    return df
def preprocess(df : pd.DataFrame)->pd.DataFrame :
    df = extract_title(df,'Name')
    df = create_family_size(df)
    df = drop_columns(df, ["PassengerId", "Embarked"])
    return df
mdf = preprocess(df)
print(mdf.columns.to_list())