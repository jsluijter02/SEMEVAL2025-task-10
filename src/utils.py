# utility functions
import pandas as pd
import pickle
import numpy as np

def load_data(directory:str = "../data/newdata.csv"):
    return pd.read_csv(directory)

def load_sub_mlb(directory:str="../pkl_files/sub_mlb.pkl"):
    with open(directory, "rb") as f:
        sub_mlb = pickle.load(f)
    return sub_mlb

def load_dom_mlb(directory:str="../pkl_files/dom_mlb.pkl"):
    with open(directory, "rb") as f:
        dom_mlb = pickle.load(f)
    return dom_mlb

def load_embeddings(directory:str="../pkl_files/embeddings.pkl"):
    embeddings = None
    with open(directory, "rb") as f:
        embeddings = pickle.load(f)
    return np.vstack(embeddings)

def load_ids(directory:str = "../data/newdata.csv"):
    df = load_data()
    return df["id"]