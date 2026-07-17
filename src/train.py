from encoding import create_encoder_and_imputer
from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

Path  = "data/raw/train.csv"

df = pd.read_csv(Path)
df = preprocess(df)

X = df.drop(columns="Survived")
y = df["Survived"]

X_train,y_train,X_test,y_test = train_test_split(X,y , random_state=42 , stratify= y , test_size= 0.2) 

transformer = create_encoder_and_imputer()
transformer.fit_transform(X_train,y)
