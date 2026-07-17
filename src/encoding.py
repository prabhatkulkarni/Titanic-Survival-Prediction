from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_encoder_and_imputer():
    categorical_features = ["Sex", "Embarked", "Title"]
    numerical_feature = ["Age"]
    categorical_pipeline = Pipeline(
        steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
    )

    numerical_pipeline = Pipeline(
        steps=[
        ("imputer", SimpleImputer(strategy="mean"))
    ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
        ("cat", categorical_pipeline,categorical_features ),
        ("num", numerical_pipeline, numerical_feature )
        ],
        remainder="passthrough"
    )

    return column_transformer
