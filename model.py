from sklearn.linear_model import LogisticRegression
from collections import Counter
import numpy as np

# does not yet work, because of multiclass multilabel data
# def log_reg(X_train, y_train, X_test):
#     lr = LogisticRegression()
#     lr.fit(X_train, y_train)

#     y_pred = lr.predict(X_test)
#     return y_pred

#y_train = np array, find the most frequent label ("other")
def most_common(X_train, y_train, X_test): 
    # finds most common label, converts numpy array to tuple, counts which tuple most frequent 
    y_tuples = [tuple(y) for y in y_train]
    max_y = Counter(y_tuples).most_common(1)
    max_y = np.array(max_y[0][0])
    print(X_test.shape[0])
    #print(X_test)
    #print(max_y)
    #print(type(max_y))
    # now prediction: all X's are the most frequently occurring y
    y_pred = np.tile(max_y, (X_test.shape[0],1))
    print(y_pred)
    return y_pred
