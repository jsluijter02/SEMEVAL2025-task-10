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
    - __embedding_directory: a string to the embedding directory. Adjust this as needed with set___embedding_directory function.
    - split_test_size: a float, indicating the proportion of the test set. If set, the preprocess function will split the data into train and test set.
    - __data_directory: directory to the dataset
    """
    def __init__(self):
        self.tf_idf = None
        self.__embedding_directory = None
        self.__split = None 
        self.vectorizer = None

    # setting functions, these are recommended to use rather that setting the attributes manually...
    def tfidf(self, min_df = 1, max_df = 1.0, max_features = None):
        self.tf_idf = {"min_df": min_df, "max_df": max_df, "max_features": max_features}
    
    def embed(self, directory:str = "../pkl_files/embeddings.pkl", normalized:bool = False):
        self.__embedding_directory = (directory, normalized)
    
    # Do we want to use the split?
    def split(self, split:bool = True):
        self.__split = split

    def load_classes(self, df:pd.DataFrame):
        # TODO: should i split this up with a sub or dom label mode??
        sub_mlb = utils.load_sub_mlb()
        return df[sub_mlb.classes_].values
    

    # preprocess function: general function that will load the data, preprocess the data (tf-idf vectorize/embed, 
    # depending on which of these parameters were set with the set functions) it loads this into X. 
    # Also loads the classes into y.
    # If split, it lastly splits the X and y data into the desired train/test set ratio.
    # returns: X and y 
    # NOTE: if split, X = {X["train"], X["test"]}, y = {y["train"], y["test"]}, see split_data function.

    def preprocess(self):
        # load in the data set
        data = utils.load_data()
        train_data = data["train"]
        test_data = data["test"]
        
        # if we have set tf_idf parameters, then we can define X as a set of tf idf vectors.
        if self.tf_idf:
            vec = TfidfVectorizer(min_df=self.tf_idf["min_df"], max_df=self.tf_idf["max_df"], max_features=self.tf_idf["max_features"])
            self.vectorizer = vec
            X = {"train": vec.fit_transform(train_data["translated_text"]), "test": vec.transform(test_data["translated_text"])}
        
        # if we set an embedding directory, we can use the sentence transformer embeddings
        # TODO: once cleaning data is implemented, then maybe we can move this to the top, and only clean data for tf idf or raw text.
        elif self.__embedding_directory:
            X = utils.load_embeddings(self.__embedding_directory[0], self.__embedding_directory[1])
        
        # if we have not set any other parameters for X, then we pass X as the raw text
        else:
            X = {"train": train_data["translated_text"], "test": test_data["translated_text"]}
        
        # classes are always the sub classes, so load those for y
        y = {"train": self.load_classes(train_data), "test": self.load_classes(test_data)}

        # load the text file identifiers
        ids = utils.load_ids()
        
        return X, y, ids
    