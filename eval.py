#Various metrics for evaluation of the model, most important one being f1
from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score, classification_report
#import data

# labels = data.load_labels("/Users/jochem/Documents/school/Uni KI jaar 4/Scriptie/Train Data/training_data_11_September_release/EN/subtask-2-annotations.txt")
# data.encode_labels(labels)

def eval(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='samples')
    pr = precision_recall_curve(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return {"f1": f1, "pr": pr, "acc": acc}

# scores = eval(labels.iloc[0,4],labels.iloc[0,4])
# print(scores["f1"], scores["pr"], scores["acc"])



# f1 = f1_score(labels.iloc[0,4], labels.iloc[4,4], pos_label = 1, average = 'binary')
# print(f1)

