from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from deep_translator import GoogleTranslator

#raw text preprocessing steps
def tf_idf_vectorize(X):
    vec = TfidfVectorizer()
    Xtf = vec.fit_transform(X)
    return Xtf

# function using google translate to translate the non-english texts into english
# if the text is too long, we chunk it,
# translate the chunk,
# and recombine the chunk into a single string
# it takes long probabl, but at least this is free
# https://github.com/nidhaloff/deep-translator
def translate(df, source_lang):
    target_lang = "en"
    translator = GoogleTranslator(source=source_lang, target=target_lang)

    # get the texts to translate
    df["text"] = df["text"].apply(...)

    #return df

def chunk(text):
    ...
def translate_chunk(text):
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