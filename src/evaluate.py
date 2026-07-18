from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score
def evaluate(y_test, y_pred)    :
    result ={
    "accuracy": accuracy_score(y_pred= y_pred,y_true=y_test),
    "precision": precision_score(y_pred= y_pred,y_true=y_test),
    "recall": recall_score(y_pred= y_pred,y_true=y_test),
    "f1": f1_score(y_pred= y_pred,y_true=y_test),
    "classification": classification_report(y_pred= y_pred,y_true=y_test),
    "confusion": confusion_matrix(y_pred= y_pred,y_true=y_test)
    }
    return result