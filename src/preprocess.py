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
        self.embedding_directory = None
        self.__split = None 
        self.data_directory = data_directory
        self.vectorizer = None

    # setting functions, these are recommended to use rather that setting the attributes manually...
    def tfidf(self, min_df = 1, max_df = 1.0, max_features = None):
        self.tf_idf = {"min_df": min_df, "max_df": max_df, "max_features": max_features}
    
    def embed(self, directory:str = "../pkl_files/embeddings.pkl", normalized:bool = False):
        self.embedding_directory = [directory, normalized]
    
    # set the test size proportion split.
    def split(self, split:float = 0.2):
        if(split > 0 and split < 1):
            self.__split = split

    def load_classes(self, df:pd.DataFrame):
        # TODO: should i split this up with a sub or dom label mode??
        sub_mlb = utils.load_sub_mlb()
        return df[sub_mlb.classes_].values
    
    def tfidf_vectorize(self, df:pd.DataFrame):
        vec = TfidfVectorizer(min_df=self.tf_idf["min_df"], max_df=self.tf_idf["max_df"], max_features=self.tf_idf["max_features"])
        self.vectorizer = vec
        return vec.fit_transform(df["text"])
    
    def split_data(self, X, y, ids):
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(X, y, ids, test_size=self.__split, random_state=1)
        return {"train": X_train, "test": X_test}, {"train": y_train, "test": y_test}, {"train": ids_train, "test": ids_test}

    # preprocess function: general function that will load the data, preprocess the data (tf-idf vectorize/embed, 
    # depending on which of these parameters were set with the set functions) it loads this into X. 
    # Also loads the classes into y.
    # If split, it lastly splits the X and y data into the desired train/test set ratio.
    # returns: X and y 
    # NOTE: if split, X = {X["train"], X["test"]}, y = {y["train"], y["test"]}, see split_data function.

    def preprocess(self):
        # load in the data set
        df = utils.load_data(self.data_directory)
        
        # if we have set tf_idf parameters, then we can define X as a set of tf idf vectors.
        if self.tf_idf:
            X = self.tfidf_vectorize(df)
        
        # if we set an embedding directory, we can use the sentence transformer embeddings
        # TODO: once cleaning data is implemented, then maybe we can move this to the top, and only clean data for tf idf or raw text.
        elif self.embedding_directory:
            X = utils.load_embeddings(self.embedding_directory[0], self.embedding_directory[1])
        
        # if we have not set any other parameters for X, then we pass X as the raw text
        else:
            X = df["text"]
        
        # classes are always the sub classes, so load those for y
        y = self.load_classes(df)

        # load the text file identifiers
        ids = utils.load_ids(self.data_directory)

        # finally, if we have set a split parameter for test size, split X and y. 
        # X = {X["train"], X["test"]}, y = {y["train"], y["test"]}, see split_data function.
        if self.__split:
            X, y, ids = self.split_data(X, y, ids)
        
        # If we don't split, all instances will be treated as "test" data. 
        # This format ensures that the pipeline class knows how to handle this.
        else: 
            X = {"test": X}
            y = {"test": y}
            ids = {"test", ids}

        return X, y, ids
    