# https://github.com/traviscoan/cards/blob/master/preprocess.py
import pandas as pd
import pickle 
import numpy as np 

import utils

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class Preprocessor:
    """
    The class preprocessor is responsible for making sure the pipeline receives the correct form of data for the model.
    The class:
        1. loading in the data from the csv file
        2. formatting X and y
        3. loading the embeddings for X
        4. loading the tf-idf scores for X
        5. splitting the data into train and test sets

    Attributes:
    - tf_idf: a dictionary of the form: {"min_df":..., "max_df":...}, containing a minimum document frequency and maximum document frequency. Adjust as needed with set_tfidf function.
    - embedding_directory: a string to the embedding directory. Adjust this as needed with set_embedding_directory function.
    - split_test_size: a float, indicating the proportion of the test set. If set, the preprocess function will split the data into train and test set.
    - data_directory: directory to the dataset
    """
    def __init__(self, data_directory:str = "../data/newdata.csv"):
        self.tf_idf = None
        self.embed = None
        self.__split = None 
        self.data_directory = data_directory

    # setting functions, these are recommended to use rather that setting the attributes manually...
    def set_tfidf(self, min_df, max_df, max_features):
        self.tf_idf = {"min_df": min_df, "max_df": max_df, "max_features": max_features}
    
    def set_embedding_directory(self, directory:str = "../pkl_files/embeddings.pkl"):
        self.embed = directory
    
    def split(self, split:float):
        if(split > 0 and split < 1):
            self.__split = split

    # loading functions: these are called by the preprocess function. 
    def load_data(self):
        return pd.read_csv(self.data_directory)

    def load_classes(self):
        # TODO: should i split this up with a sub or dom label mode??
        df = self.load_data()
        sub_mlb = utils.load_sub_mlb()
        return df[sub_mlb.classes_].values

    def load_embeddings(self):
        embeddings = None
        with open(self.embed, "rb") as f:
            embeddings = pickle.load(f)
        return np.vstack(embeddings)
    
    def tfidf_vectorize(self):
        df = self.load_data()
        vec = TfidfVectorizer(min_df=self.tf_idf["min_df"], max_df=self.tf_idf["max_df"], max_features=self.tf_idf["max_features"])
        return vec.fit_transform(df["text"])
    
    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.__split, random_state=1)
        return {"train": X_train, "test": X_test}, {"train": y_train, "test": y_test} 

    # once all parameters are set, call the preprocess function to obtain the X and y needed for the model.
    def preprocess(self):
        if self.tf_idf:
            X = self.tfidf_vectorize()
        elif self.embed:
            X = self.load_embeddings()
        else:
            X = self.load_data()["text"]
        
        y = self.load_classes()

        if self.__split:
            X, y = self.split_data(X, y)

        return X, y
    