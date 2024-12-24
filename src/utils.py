# utility functions
import pandas as pd
import pickle
import numpy as np

def load_data(train_directory:str = "../data/train_data.pkl", test_directory:str="../data/test_data.pkl"):
    train_df = pd.read_pickle(train_directory)
    test_df = pd.read_pickle(test_directory)
    return {"train": train_df, "test": test_df}


def load_sub_mlb(directory:str="../pkl_files/sub_mlb.pkl"):
    with open(directory, "rb") as f:
        sub_mlb = pickle.load(f)
    return sub_mlb

def load_dom_mlb(directory:str="../pkl_files/dom_mlb.pkl"):
    with open(directory, "rb") as f:
        dom_mlb = pickle.load(f)
    return dom_mlb

def load_embeddings(directory:str="../pkl_files/embeddings.pkl", normalized:bool = False):
    embeddings = None
    with open(directory, "rb") as f:
        embeddings = pickle.load(f)
    return np.vstack(embeddings) # TODO: moet dit wel gevstacked worden? # TODO: normalized toevoegen

def load_ids():
    data = load_data()
    train_ids = data["train"]["id"]
    test_ids = data["test"]["id"]
    return {"train": train_ids, "test": test_ids}

# if __name__ == "__main__":
#     load_ids()