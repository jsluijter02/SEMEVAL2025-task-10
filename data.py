import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# loads all text files from the directory into a single dataframe
# id = file name (string)
# text = files contents (string)
def load_text_data(directory):
    data = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory,file)
        with open(file_path, 'r') as f:
            raw_text = f.read()
            data.append({"id":file, "text": raw_text})
    print("Data length: ", len(data))
    df = pd.DataFrame(data)
    return df

# saves the narrative labels to dataframe. 
# id = filename (string), dom_narr = list of narratives (list<string>), sub-narr = the list of subnarratives(list<string>) 
def load_labels(file_path):
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            tags = line.strip().split("\t")
            
            text_id = tags[0].strip()
            # print(text_id)
            dom_narrs = [narr.strip() for narr in tags[1].split(";")]
            sub_narrs = [narr.strip() for narr in tags[2].split(";")]

            labels.append({"id":text_id,"dom_narr": dom_narrs, "sub_narr": sub_narrs})
    print("Labels length:", len(labels))
    df = pd.DataFrame(labels)
    #print(labels)
    return df

# one-hot encodes the dominant- and sub-narrative labels              
def encode_labels(df):
    dom_mlb = MultiLabelBinarizer()
    sub_mlb = MultiLabelBinarizer()

    dom_narr_enc = dom_mlb.fit_transform(df["dom_narr"])
    #print("dominant narratives: ", (dom_mlb.classes_))
    sub_narr_enc = sub_mlb.fit_transform(df["sub_narr"])
    #print("sub-narratives: ", (sub_mlb.classes_))
    
    return dom_mlb, sub_mlb, dom_narr_enc, sub_narr_enc


# function for the whole thing
def load_data():
    df_text = load_text_data("/Users/jochem/Documents/school/Uni KI jaar 4/Scriptie/Train Data/training_data_16_October_release/EN/raw-documents")
    df_labels = load_labels("/Users/jochem/Documents/school/Uni KI jaar 4/Scriptie/Train Data/training_data_16_October_release/EN/subtask-2-annotations.txt")
    df_merged = pd.merge(df_text, df_labels, on="id")
    #df_merged.to_clipboard()
    return df_merged

# missing_files = set(df_labels["id"]) - set(df["id"])
# print(missing_files)
# value = df.iloc[0,4]
# print(value)
# print(type(value))
# print(len(value))

# df = load_data()
# dom_mlb, sub_mlb, dom_narr_enc, sub_narr_enc = encode_labels(df)

# print(dom_mlb.classes_) 
# sub narr classes zijn gewoon de URW: dom: sub
# split ":"? 
# misschien map maken en dan in op kunnen zoeken?
##hoe omzetten van ypred naar labels naar dom narrs?

