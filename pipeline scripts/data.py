import os
import pandas as pd
from deep_translator import GoogleTranslator

# class DataLoader():
#     def __init__(self):
#         self.directories = 
#     def translate_file(source_lang):
#      def load_data():
#     def load_labels():

# loads all text files from the directory into a single dataframe
# id = file name (string)
# text = files contents (string)
def load_text_data(directory:str, translate: bool = False, source_lang:str = "auto"):
    data = []

    # initialize the translator for if we need to translate
    target_lang = "en"
    translator = GoogleTranslator(source=source_lang, target=target_lang)

    for file in os.listdir(directory):
        file_path = os.path.join(directory,file)

        # translator translates the file and appends that to the dataframe
        if translate:
            try:
                text = translator.translate_file(file_path)
                data.append({"id":file, "text": text})
            except Exception as e:
                print(f"Error while reading:{file}")
            continue
        
        # when we dont need to translate, the file is saved as is
        try:
            with open(file_path, 'r') as f:
                text = f.read()
                data.append({"id":file, "text": text})
        except Exception as e:
            print(f"Error while reading:{file}")

    print("Data length: ", len(data))
    df = pd.DataFrame(data)
    return df

# this is dumb
def translate_longer_file(file_path, source_lang: str = "auto"):
    translated = []

    # initialize the translator for if we need to translate
    target_lang = "en"
    translator = GoogleTranslator(source=source_lang, target=target_lang)

    with open(file_path, "r") as f:
        text = f.read()
        piece = ""

        for line in text:
            if len(line)+len(piece) > 4500:
                print(piece)
                translated.append(translator.translate(piece))
                piece = line
            else:
                piece += line
        
        translated.append(translator.translate(piece))
    
    return "".join(translated)

# saves the narrative labels to dataframe. 
# id = filename (string), 
# dom_narr = list of narratives (list<string>), 
# sub-narr = the list of subnarratives(list<string>) 
def load_labels(file_path:str):
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

def load_explanation_data(file_path:str):
    explanations = []

    with open(file_path, "r") as f:
        for line in f:
            tags = line.strip().split("\t")

            text_id = tags[0].strip()

            dom_narr = tags[1].strip()
            sub_narr = tags[2].strip()
            explanation = tags[3]

            explanations.append({"id": text_id, "task_3_dom_narr": dom_narr, "task_3_sub_narr": sub_narr, "task_3_explanation": explanation})
            #print(explanation)
    
    df = pd.DataFrame(explanations)
    return df

