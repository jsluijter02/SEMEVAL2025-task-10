# puts the predicted y's back into the text file with the ids
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer

# def save_predictions(df, y_pred, sub_mlb):
#     # opens a new predictions file with the current date and time
#     date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
#     narr_dict = narrative_dictionary(sub_mlb)
#     y_pred_dom, y_pred_sub = narr_predictions()
#     ##hoe nu id?
#     with open(f"predictions/{date}", 'w') as f:
        

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

