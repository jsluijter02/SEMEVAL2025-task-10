# utility functions
import pickle
def load_sub_mlb(directory:str="../pkl_files/sub_mlb.pkl"):
    with open(directory, "rb") as f:
        sub_mlb = pickle.load(f)
    return sub_mlb

def load_dom_mlb(directory:str="../pkl_files/dom_mlb.pkl"):
    with open(directory, "rb") as f:
        dom_mlb = pickle.load(f)
    return dom_mlb