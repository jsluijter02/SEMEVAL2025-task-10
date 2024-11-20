from sklearn.metrics import classification_report, hamming_loss
import numpy as np

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
    def eval(self, target_names) -> dict:
        prf1 = self.precision_recall_f1(target_names)
        hamming = self.hammingloss()
        exactmatchratio = self.exact_match_ratio()

        return {
            "classification_report": prf1,
            "hamming_loss": hamming,
            "exact_match_ratio":exactmatchratio
        }
