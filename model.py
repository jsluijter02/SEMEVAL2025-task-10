from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import LeaveOneOut
from collections import Counter
import numpy as np

def logistic_regression_classifier(X_train, y_train, X_test):
    lr = MultiOutputClassifier(LogisticRegression(class_weight="balanced", solver= "liblinear", max_iter=10000))
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    return y_pred

#y_train = np array, find the most frequent label ("other")
def majority_classifier(y_train, X_test): 
    # finds most common label, converts numpy array to tuple, counts which tuple most frequent 
    max_y = find_majority(y_train)

    # now prediction: all X's are the most frequently occurring y
    y_pred = np.tile(max_y, (X_test.shape[0],1))
    print(y_pred)
    return y_pred

def find_majority(y_train):
    y_tuples = [tuple(y) for y in y_train]
    max_y = Counter(y_tuples).most_common(1)
    max_y = np.array(max_y[0][0])
    return max_y

# Does the logistic regression n times, as the data set is quite small
def logistic_regression_loocv(X, y, majority_ensemble = False):
    # initialize the cross validation
    cv = LeaveOneOut()
    cv.get_n_splits(X)
    
    # output arrays of the predicted label for that instance of y and the true label, so we can evaluate it later
    labels_pred = []
    labels_true = []

    # Most common label = Other
    max_y = find_majority(y)

    for i, (train_index, test_index) in enumerate(cv.split(X)):
        # Get the train and test instances for this fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # LR doesn't work if one of the classes has no positive instances, skip if this happens
        valid_y = np.any(y_train != 0, axis=0)
        if not np.all(valid_y):
            print(f"Skipped fold {i}, labels invalid")
            continue
        
        # If not invalid, train the classifier
        lr = MultiOutputClassifier(LogisticRegression(class_weight= "balanced", solver="liblinear", max_iter=100, penalty="l2"))
        lr.fit(X_train,y_train)

        # Make prediction for the test instance
        y_pred = lr.predict(X_test)

        if(majority_ensemble):
            if np.all(y_pred == 0):
                y_pred = max_y

        labels_pred.append(y_pred)
        labels_true.append(y_test)

    # good format for classification report
    labels_pred = np.vstack(labels_pred)
    labels_true = np.vstack(labels_true)
    print("Predicted Labels: ", labels_pred)
    print("True Labels: ", labels_true)

    return labels_pred, labels_true
    

