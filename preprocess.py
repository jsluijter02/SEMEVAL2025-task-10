from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from deep_translator import GoogleTranslator
from transformers import BertModel, BertTokenizer


#raw text preprocessing steps
def tf_idf_vectorize(X):
    vec = TfidfVectorizer()
    Xtf = vec.fit_transform(X)
    return Xtf

# https://www.geeksforgeeks.org/how-to-generate-word-embedding-using-bert/
def embed(X):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    encoding = tokenizer.batch_encode_plus(X,padding=True, truncation=False, return_tensors="pt", add_special_tokens=True) # truncation??

def tokenizer(X):
    ...
    
# label preprocessing steps
# one-hot encodes the dominant- and sub-narrative labels              
def encode_labels(df):
    dom_mlb = MultiLabelBinarizer()
    sub_mlb = MultiLabelBinarizer()

    dom_narr_enc = dom_mlb.fit_transform(df["dom_narr"])
    df = pd.concat([df, pd.DataFrame(dom_narr_enc, columns=dom_mlb.classes_)], axis=1)

    sub_narr_enc = sub_mlb.fit_transform(df["sub_narr"])
    df = pd.concat([df, pd.DataFrame(sub_narr_enc, columns=sub_mlb.classes_)], axis=1)
    
    return dom_mlb, sub_mlb, dom_narr_enc, sub_narr_enc, df