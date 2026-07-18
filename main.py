from src.evaluate import evaluate
from src.train import train
from src.visualize import complete_visualize
from src.preprocessing import preprocess
import pandas as pd
import joblib
results_df,y_test,y_prob = train()
evaluate(y_test,y_prob)
complete_visualize()
best_model_name = results_df.sort_values(
    "accuracy",
    ascending=False
).iloc[0]["model"]
print(f"Best model is {best_model_name}")
model = joblib.load("models/model.joblib")

sample = pd.read_csv("data/raw/test.csv").iloc[[0]]
sample = preprocess(sample)
prediction = model.predict(sample)
if prediction ==[ 0 ]:
    print(f"Model predicts not Survived")
else:
    print("Model predicts Survived")
