from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
import os
import utils

class Postprocessor:
    def __init__(self, y_pred, ids):
        self.y_pred = y_pred
        self.ids = ids
        self.sub_mlb = utils.load_sub_mlb()
    
    def save_predictions(self, model_name:str = "all"):
        # opens a new predictions file with the current date and time
        # TODO: change to f1 score later, so we can see which predictions are the best!!! -> this does not work on the test and dev set egg head
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        narr_dict = self.narrative_dictionary()
        y_pred_dom, y_pred_sub = self.narr_predictions(narr_dict)
        
        prediction_directory = os.path.join("./predictions",f"{model_name}")
        file_path = os.path.join(prediction_directory,f"{date}.txt")
        os.makedirs(prediction_directory, exist_ok=True)

        with open(file_path, 'w') as f:
            for i in range(len(self.ids)):
                dom_narrs = ";".join(y_pred_dom[i])
                sub_narrs = ";".join(y_pred_sub[i])
                id = self.ids.iloc[i]

                f.write(f"{id}\t{dom_narrs}\t{sub_narrs}\n")
    
        
    def narrative_dictionary(self):
        # first create the mapping from sub- to dominant narratives
        narr_dict = {}

        for sub in self.sub_mlb.classes_:
            if sub == "Other":
                narr_dict[sub] = "Other" # Other Other
                continue
            sub_narr_split = sub.split(":")
            dom_from_sub = sub_narr_split[0]+":"+sub_narr_split[1]
            narr_dict[sub] = dom_from_sub

        return narr_dict

    # narr_prediction takes the predicted y and transforms they 
    def narr_predictions(self, narr_dict):
        y_pred_sub = self.sub_mlb.inverse_transform(self.y_pred)
        y_pred_dom = []
        for pred in y_pred_sub:
            dom_list = []
            for sub in pred:
                dom = narr_dict[sub]
                dom_list.append(dom)
            y_pred_dom.append(dom_list)
        print(y_pred_dom)
        return y_pred_dom, y_pred_sub