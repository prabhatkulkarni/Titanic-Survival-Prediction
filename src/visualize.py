from train import train
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.metrics import roc_curve, auc

result_df,y_test,y_prob = train()
def plot_metrics_bar():
    print("Metrics_Bar")
    metrics = ["accuracy", "precision", "recall", "f1"]

    for col in metrics:
        plt.figure(figsize=(10,6))
        sns.barplot(data=result_df, x="model", y=col)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()
def plot_confusion_matrix() :
    print("Confusion Matrix")
    for _, row in result_df.iterrows():

        model = row["model"]
        cm = row["confusion"]

        plt.figure(figsize=(6,5))

        sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"]
        )
        plt.xlabel("Predicted Survived")
        plt.ylabel("Actual Survived")
        plt.title(model)

        plt.show()
def roc_curve_visual(y_test ,y_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,6))

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")

    plt.plot([0,1], [0,1], "--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.show()
def complete_visualize():
    plot_metrics_bar()
    plot_confusion_matrix()
    roc_curve_visual(y_test,y_prob)