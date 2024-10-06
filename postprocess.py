# puts the predicted y's back into the text file with the ids
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
import os

def save_predictions(y_pred, sub_mlb, ids):
    # opens a new predictions file with the current date and time
    # change to f1 score later, so we can see which predictions are the best!!!
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    narr_dict = narrative_dictionary(sub_mlb)
    y_pred_dom, y_pred_sub = narr_predictions(y_pred, narr_dict, sub_mlb)
    
    file_path = os.path.join("./predictions",f"{date}.txt")
    with open(file_path, 'w') as f:
        for i in range(len(ids)):
            dom_narrs = ";".join(y_pred_dom[i])
            sub_narrs = ";".join(y_pred_sub[i])
            id = ids.iloc[i]

            f.write(f"{id}\t{dom_narrs}\t{sub_narrs}\n")
    
        

def narrative_dictionary(sub_mlb = MultiLabelBinarizer):
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

def narr_predictions(y_pred, narr_dict, sub_mlb = MultiLabelBinarizer):
    y_pred_sub = sub_mlb.inverse_transform(y_pred)
    y_pred_dom = []
    for pred in y_pred_sub:
        dom_list = []
        for sub in pred:
            dom = narr_dict[sub]
            dom_list.append(dom)
        y_pred_dom.append(dom_list)
    print(y_pred_dom)
    return y_pred_dom, y_pred_sub

