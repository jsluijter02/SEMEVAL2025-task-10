from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import numpy as np
import torch.nn as nn


#raw text preprocessing steps
def tf_idf_vectorize(X):
    vec = TfidfVectorizer()
    Xtf = vec.fit_transform(X)
    return Xtf

# thanks ruben voor idee convolution
def embed_x(X, SBERTmodel: SentenceTransformer):
    sentences = sent_tokenize(X)
    embeddings = SBERTmodel.encode(sentences)

    mean_pool_embedding = np.mean(embeddings, axis = 0)
    #print(mean_pool_embedding)
    return mean_pool_embedding

    
# label preprocessing steps
# one-hot encodes the dominant- and sub-narrative labels              
def encode_labels(df):
    dom_mlb = MultiLabelBinarizer()
    sub_mlb = MultiLabelBinarizer()

    dom_narr_enc = dom_mlb.fit_transform(df["dom_narr"])
    df = pd.concat([df, pd.DataFrame(dom_narr_enc, columns=dom_mlb.classes_)], axis=1)

    sub_narr_enc = sub_mlb.fit_transform(df["sub_narr"])
    df = pd.concat([df, pd.DataFrame(sub_narr_enc, columns=sub_mlb.classes_)], axis=1)
    
    return dom_mlb, sub_mlb, df