# https://github.com/traviscoan/cards/blob/master/preprocess.py
import pandas as pd
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor:
    def load_data(self, directory:str = "../data/newdata.csv", split:bool = False, embed:bool=False, tfidf:bool = False):
        df = pd.read_csv(directory)
        
        sub_mlb = None
        with open("../pkl_files/sub_mlb.pkl", "rb") as f:
            sub_mlb = pickle.load(f)

        X = df["text"]
        y = df[sub_mlb.classes_].values

        if embed:
            X = self.load_embeddings()
        elif tfidf:
            X = self.tfidf_vectorize()
        
        if split:
            from sklearn.model_selection import train_test_split

    
    def load_embeddings(self, directory:str = "../pkl_files/embeddings.pkl"):
        embeddings = None
        with open(directory, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings
    
    def tfidf_vectorize(self, X, min_df:float = 0.01, max_df:float = 0.35):
        vec = TfidfVectorizer(min_df=min_df,max_df=max_df)
        return vec.fit_transform(X)
        

    def clean():
        ...
    
    # which state task?
    # also say they can come to room 7.27
    