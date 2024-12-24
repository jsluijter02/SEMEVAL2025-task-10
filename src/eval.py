import os
from sklearn.metrics import multilabel_confusion_matrix, classification_report, hamming_loss, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
        confusion = multilabel_confusion_matrix(self.y_true, self.y_pred)
        print(confusion)
        
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
    # TODO: write this one out to an excel file as well.
    def all_zero_detect(self):
        print(self.y_pred.shape)
        print(self.y_true.shape)
        print(self.ids.shape)
        empty_preds = {}
        for i, pred in enumerate(self.y_pred):
            if not np.any(pred):
                empty_preds[self.ids.to_numpy()[i]] = self.__sub_mlb.inverse_transform(np.array([self.y_true[i]]))

        print(empty_preds)
        return empty_preds
            

    # returns the invalid predictions, so Other and another class
    def invalid_other(self):
        os.makedirs("../erroranalysis", exist_ok=True)
        other_index = 45 # TODO: DONT MAKE THIS HARDCODED
        wrong_indices = [i for i,c in enumerate(self.y_pred) if c[other_index] == 1 and np.sum(c)>1]
        wrong = pd.DataFrame({"y_pred": self.__sub_mlb.inverse_transform(self.y_pred[wrong_indices]), "y_true": self.__sub_mlb.inverse_transform(self.y_true[wrong_indices])}, index=self.ids.to_numpy()[wrong_indices])
        wrong.to_excel("../erroranalysis/invalidOther.xlsx")
        # TODO: find out how to find the Other classes and the classes its adding to these predictions
        return wrong
    
    # returns the predictions where both CC and URW classes were selected
    # also writes these to an excel file
    def invalid_CC_URW(self):
        os.makedirs("../erroranalysis", exist_ok=True)
        df = pd.DataFrame(self.y_pred,columns=self.__sub_mlb.classes_, index=self.ids)
        print(df)
        sub_URW_classes = [cls for cls in self.__sub_mlb.classes_ if cls[:3] == "URW"]
        sub_CC_classes = [cls for cls in self.__sub_mlb.classes_ if cls[:2] == "CC"]
        
        wrong = df.loc[(df[sub_URW_classes].sum(axis=1) >=1) & (df[sub_CC_classes].sum(axis=1) >= 1)]
        wrong.to_excel("../erroranalysis/invalidCCURW.xlsx")
        
        return wrong

    # gives an overview of the classes that give the most fp and fn
    # def most_wrongly_predicted(self):
    #     # TODO: find all fp's fn's in data and print which classes and the support of them
    #     # I also want to see what classes were predicted the most instead of these
    #     cfm = multilabel_confusion_matrix(self.y_true, self.y_pred)
    #     # fig, ax = plt.subplots(19,5,figsize=(5*2,19*2))
    #     # ax = ax.flatten()
    #     # for i,cm in enumerate(cfm):
    #     #     disp = ConfusionMatrixDisplay(cm)
    #     #     disp.plot(ax=ax[i])
    #     #     ax[i].title = str(self.__sub_mlb.classes_)
    #     # print(cfm)
    #     fps = ...
    #     fns = ...

    
    # how many classes does the model on average predict per label?
    def avg_num_predictions(self):
        ...
    
    # Runs all algorithms
    def analyze(self):
        self.all_zero_detect()
        self.invalid_other()
        self.invalid_CC_URW()
        # self.most_wrongly_predicted()
        ...

if __name__ == "__main__":
    y_p = np.load("./predictions.npz")["y_pred"]
    y_t = np.load("./predictions.npz")["y_true"]
    err = ErrorAnalyzer(y_true=y_t,y_pred=y_p, ids = None)
