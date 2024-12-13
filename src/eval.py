from sklearn.metrics import classification_report, hamming_loss
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

import utils

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

class ErrorAnalyzer:
    def __init__(self, y_pred, y_true, ids):
        self.y_pred = y_pred
        self.y_true = y_true
        self.ids = ids
        self.__sub_mlb = utils.load_sub_mlb()

    # Detects empty predictions and returns a dictionary of their ids (key) and what their true predictions should have been (value)
    def all_zero_detect(self):
        empty_preds = {}
        for i, pred in enumerate(self.y_pred):
            if not np.any(pred):
                empty_preds[self.ids[i]] = self.__sub_mlb.inverse_transform(self.y_true[i])
        return empty_preds
            

    # returns the invalid predictions, so Other and another class
    def invalid_other(self):
        other_index = 45 # TODO: DONT MAKE THIS HARDCODED
        wrong_indices = [i for i,c in enumerate(self.y_pred) if c[other_index] == 1 and np.sum(c)>1]
        df = pd.DataFrame({"y_pred": self.__sub_mlb.inverse_transform(self.y_pred[wrong_indices]), "y_true": self.__sub_mlb.inverse_transform(self.y_true[wrong_indices])}, index=self.ids.to_numpy()[wrong_indices])
        df.to_excel("../wrong_predictions.xlsx")
        # TODO: find out how to find the Other classes and the classes its adding to these predictions
        return df
    
    #
    def invalid_CC_URW(self):
        # df = pd.DataFrame(self.y_pred,columns=self.__sub_mlb.classes_, index=self.ids)
        # sub_URW_classes = [cls for cls in self.__sub_mlb.classes_ if cls[:3] == "URW"]
        # sub_CC_classes = [cls for cls in self.__sub_mlb.classes_ if cls[:2] == "CC"]
        
        # wrong = df.loc[df[sub_URW_classes].sum(axis=1) >=1 & df[sub_CC_classes].sum(axis=1) >= 1]
        # wrong.to_clipboard()
        # print(wrong)
        ...

    # gives an overview of the classes that give the most fp and fn
    def most_wrongly_predicted(self):
        ... # TODO: find all fp's fn's in data and print which classes and the support of them
    
    # Runs all algorithms
    def analyze(self):
        #self.invalid_CC_URW()
        ...