from encoding import non_scaler_transformer,scaler_transformer
from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np

Path  = "data/raw/train.csv"

df = pd.read_csv(Path)
df = preprocess(df)

X = df.drop(columns="Survived")
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y , random_state=42 , stratify= y , test_size= 0.2) 

#Descision Tree , Random forest , AdaBoost ,  XGBoost don't require Scalar 
Non_scaler_models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(
    ),
    "ExtraTrees":ExtraTreesClassifier()
}
Scaler_models = {
    "Logistic Classification":LogisticRegression(),
    "KNN Classification":KNeighborsClassifier(),
    "SVM Classification":SVC()
}
for name,model in Non_scaler_models.items():
    models = Pipeline(
        steps=
        [
           ( "preprocess",non_scaler_transformer()),
            ("name" , model)
        ]
    
    )
    models.fit(X_train,y_train)
    y_predict = models.predict(X_test)
    
for name , model in Scaler_models.items():
    models = Pipeline(
        steps=
        [
            ("preprocess",scaler_transformer()),
            ("name" , model )
        ]
    )
    models.fit(X_train,y_train)
    y_predict=models.predict(X_test)