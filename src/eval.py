from sklearn.metrics import classification_report, hamming_loss
import numpy as np
import pandas as pd

# Evaluator class, to receive all relevant metrics from a set of true and predictor labels
# https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics
class Evaluator:
    def __init__(self, y_pred, y_true):
        assert(len(y_true) == len(y_pred))
        self.y_pred = y_pred
        self.y_true = y_true
    
    # classification report 
    def prec_recall_f1(self, target_names = None):
        return classification_report(y_true=self.y_true, 
                                     y_pred=self.y_pred, 
                                     target_names=target_names,
                                     output_dict=True,
                                     zero_division=0.0)
    
    # hamming loss
    def hammingloss(self):
        return hamming_loss(y_pred=self.y_pred, y_true=self.y_true)
    
    # exact match ratio
    def exact_match_ratio(self):
        return np.all(self.y_pred == self.y_true, axis = 1).mean()
    
    # function to run and have everything
    # TODO: maybe eval function can save to a file? 
    def eval(self) -> dict:
        prf1 = self.prec_recall_f1()
        hamming = self.hammingloss()
        exactmatchratio = self.exact_match_ratio()
        averages = {
            "micro avg": prf1["micro avg"],
            "macro avg": prf1["macro avg"],
            "weighted avg": prf1["weighted avg"],
            "samples avg": prf1["samples avg"]
        }
        avg_df = pd.DataFrame(averages).T
        print(avg_df.head())
        return {
            "classification_report": averages,
            "hamming_loss": hamming,
            "exact_match_ratio":exactmatchratio
        }
