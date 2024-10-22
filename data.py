import os
import pandas as pd

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
# id = filename (string), 
# dom_narr = list of narratives (list<string>), 
# sub-narr = the list of subnarratives(list<string>) 
def load_labels(file_path):
    labels = []
    with open(file_path, "r") as f:
        for line in f:
            tags = line.strip().split("\t")
            
            text_id = tags[0].strip()

            dom_narrs = [narr.strip() for narr in tags[1].split(";")]
            sub_narrs = [narr.strip() for narr in tags[2].split(";")]

            labels.append({"id":text_id,"dom_narr": dom_narrs, "sub_narr": sub_narrs})
    print("Labels length:", len(labels))
    df = pd.DataFrame(labels)
    return df



