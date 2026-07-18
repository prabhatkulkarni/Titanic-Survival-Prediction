from .encoding import non_scaler_transformer,scaler_transformer
from .preprocessing import preprocess
from .evaluate import evaluate
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
import joblib
def train()->pd.DataFrame:
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
        "Logistic Classification":LogisticRegression(max_iter=1000, random_state=42),
        "KNN Classification":KNeighborsClassifier(),
        "SVM Classification":SVC()
    }
    best_score = 0
    best_pipeline = None
    best_name = None

    results = []
    for name,model in Non_scaler_models.items():
        models = Pipeline(
            steps=
        [
           ( "preprocess",non_scaler_transformer()),
            ("model", model)
        ]
    
        )
        models.fit(X_train,y_train)
        y_pred = models.predict(X_test)

        report = evaluate(y_test, y_pred)
        report["model"] = name 
        results.append(report)
        if report["accuracy"] > best_score:
            best_score = report["accuracy"]
            best_pipeline = models
            best_name = name

    for name , model in Scaler_models.items():
        models = Pipeline(
            steps=
            [
            ("preprocess",scaler_transformer()),
            ("model", model)
            ]
        )
        models.fit(X_train,y_train)
        y_pred = models.predict(X_test)

        report = evaluate(y_test, y_pred)
        report["model"] = name 
        results.append(report)
        if report["accuracy"] > best_score:
            best_score = report["accuracy"]
            best_pipeline = models
            best_name = name
    joblib.dump(best_pipeline, "models/model.joblib")
    print(f"Saved best model: {best_name}")
    results_df = pd.DataFrame(results)
    return results_df,y_test,y_pred