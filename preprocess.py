from sklearn.feature_extraction.text import TfidfVectorizer

#raw text preprocessing steps
def tf_idf_vectorize(X):
    vec = TfidfVectorizer()
    Xtf = vec.fit_transform(X)
    return Xtf


#label preprocessing steps