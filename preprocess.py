from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

#raw text preprocessing steps
def tf_idf_vectorize(X):
    vec = TfidfVectorizer()
    Xtf = vec.fit_transform(X)
    return Xtf

def translate():
    target_lang = "en"

    ...

def embed(X):
    ...


# label preprocessing steps
# one-hot encodes the dominant- and sub-narrative labels              
def encode_labels(df):
    dom_mlb = MultiLabelBinarizer()
    sub_mlb = MultiLabelBinarizer()

    dom_narr_enc = dom_mlb.fit_transform(df["dom_narr"])
    #print("dominant narratives: ", (dom_mlb.classes_))
    sub_narr_enc = sub_mlb.fit_transform(df["sub_narr"])
    #print("sub-narratives: ", (sub_mlb.classes_))
    
    return dom_mlb, sub_mlb, dom_narr_enc, sub_narr_enc