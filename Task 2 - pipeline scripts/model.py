class LogisticRegression:
    """
    A model class for logistic regression. To accomodate the multilabeled data, 
    sklearns LogisticRegression class is wrapped in an sklearn MultiOutputClassifier.

    functions:
    - fit: fit takes the train variables x_train and y_train and fits the logistic regression to the training data.
    - predict: makes predictions on the x_test set.

    attributes:
    - model: model stores the sklearn model object.
    """
    def __init__(self, solver= "liblinear", class_weight="balanced", max_iter=1000):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multioutput import MultiOutputClassifier
        self.model = MultiOutputClassifier(LogisticRegression(solver=solver,class_weight=class_weight,max_iter=max_iter))
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    # TODO: maybe return y_pred and y_true?
    def predict(self, X_test):
        return self.model.predict(X_test)

class SVM:
    ...

class GPT4o:
    ...

class LOOCV_LogisticRegression:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multioutput import MultiOutputClassifier 
    
    #def fit(self, X_train)


if __name__ == "__main__":
    print("hi im running when the scripts running")
    import pandas as pd
    import pickle
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("./data/data.csv")
    print(df)
    sub_mlb = ...
    with open("./pkl Files/sub_mlb.pkl", "rb") as f:
        sub_mlb = pickle.load(f)

    print(sub_mlb.classes_)
    X = df["text"] # convert to numeric values
    y = df[sub_mlb.classes_].values

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.1, random_state = 1)

    cl = LogisticRegression()
    cl.fit(X_train=X_train, y_train=y_train)
    y_pred = cl.predict(X_test=X_test)

    print(y_pred)