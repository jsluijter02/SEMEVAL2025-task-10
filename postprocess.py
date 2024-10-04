# puts the predicted y's back into the text file with the ids
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer

def save_predictions(df, y_pred, sub_mlb):
    # opens a new predictions file with the current date and time
    date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    #with open(f"predictions/{date}", 'w') as f:
        

##hoe omzetten van ypred naar labels naar dom narrs?

