#Various metrics for evaluation of the model, most important one being f1
from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score, classification_report

def eval(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='samples')
    pr = precision_recall_curve(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return {"f1": f1, "pr": pr, "acc": acc}
