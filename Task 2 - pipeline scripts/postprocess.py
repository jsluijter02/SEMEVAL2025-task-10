from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
import os

class Postprocessor:
    def __init__(self, y_pred):
        self.y_pred = y_pred
    
    def save_predictions(self, sub_mlb, ids, model_name:str = "all"):
        # opens a new predictions file with the current date and time
        # TODO: change to f1 score later, so we can see which predictions are the best!!! -> this does not work on the test and dev set egg head
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        narr_dict = self.narrative_dictionary(sub_mlb)
        y_pred_dom, y_pred_sub = self.narr_predictions(narr_dict, sub_mlb)
        
        file_path = os.path.join("./predictions",f"{model_name}/{date}.txt")
        with open(file_path, 'w') as f:
            for i in range(len(ids)):
                dom_narrs = ";".join(y_pred_dom[i])
                sub_narrs = ";".join(y_pred_sub[i])
                id = ids.iloc[i]

                f.write(f"{id}\t{dom_narrs}\t{sub_narrs}\n")
    
        
    def narrative_dictionary(self, sub_mlb = MultiLabelBinarizer):
        # first create the mapping from sub- to dominant narratives
        narr_dict = {}

        for sub in sub_mlb.classes_:
            if sub == "Other":
                narr_dict[sub] = "Other" # Other Other
                continue
            sub_narr_split = sub.split(":")
            dom_from_sub = sub_narr_split[0]+":"+sub_narr_split[1]
            narr_dict[sub] = dom_from_sub

        return narr_dict

    # narr_prediction takes the predicted y and transforms they 
    def narr_predictions(self, narr_dict, sub_mlb = MultiLabelBinarizer):
        y_pred_sub = sub_mlb.inverse_transform(self.y_pred)
        y_pred_dom = []
        for pred in y_pred_sub:
            dom_list = []
            for sub in pred:
                dom = narr_dict[sub]
                dom_list.append(dom)
            y_pred_dom.append(dom_list)
        print(y_pred_dom)
        return y_pred_dom, y_pred_sub