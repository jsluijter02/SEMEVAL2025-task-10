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


# function for the whole thing
def load_data():
    df_text_en = load_text_data("/Users/jochem/Documents/school/Uni KI jaar 4/Scriptie/Train Data/training_data_16_October_release/EN/raw-documents")
    df_text_hi = load_text_data("/Users/jochem/Documents/school/Uni KI jaar 4/Scriptie/Train Data/training_data_16_October_release/HI/raw-documents")
    df_text_pt = load_text_data("/Users/jochem/Documents/school/Uni KI jaar 4/Scriptie/Train Data/training_data_16_October_release/PT/raw-documents")
    df_text = pd.concat([df_text_en, df_text_hi, df_text_pt])

    df_labels_en = load_labels("/Users/jochem/Documents/school/Uni KI jaar 4/Scriptie/Train Data/training_data_16_October_release/EN/subtask-2-annotations.txt")
    df_labels_hi = load_labels("/Users/jochem/Documents/school/Uni KI jaar 4/Scriptie/Train Data/training_data_16_October_release/HI/subtask-2-annotations.txt")
    df_labels_pt = load_labels("/Users/jochem/Documents/school/Uni KI jaar 4/Scriptie/Train Data/training_data_16_October_release/PT/subtask-2-annotations.txt")
    df_labels = pd.concat([df_labels_en, df_labels_hi, df_labels_pt])

    df_merged = pd.merge(df_text, df_labels, on="id")
    #df_merged.to_clipboard()
    return df_merged


