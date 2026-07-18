from evaluate import evaluate
from train import train
from visualize import complete_visualize
result_df,y_test,y_prob = train()
evaluate(y_test,y_prob)
complete_visualize()