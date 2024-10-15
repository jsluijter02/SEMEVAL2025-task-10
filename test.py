import data
import model
import eval
import postprocess
import numpy as np
#misschien kan ik dit beter in een notebook doen?
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

# loads the data and one hot encode the appropriate labels
dat = data.load_data()
dom_mlb, sub_mlb, dom_narr_enc, sub_narr_enc = data.encode_labels(dat)

# print(sub_narr_enc)
# print(type(sub_narr_enc))
# # setup X and y for the models
X = dat.text
y = sub_narr_enc

# vectorize the X data for the logreg model - ignore cus were not using it rn
#hoort ook eig beetje in preprocessing 
vectorizer = TfidfVectorizer()
Xtf = vectorizer.fit_transform(X)
# print(X)

# #divide the data into train and test splits, small test size, cause little train data
#X_train, X_test, y_train, y_test, y_natural_train, y_natural_test = train_test_split(Xtf, y, dat.sub_narr, test_size = 0.01)
# print(id_test)
# print(X_train)
# print(y_train.shape)
# print(y_test.shape)
# print(type(y_natural_train))
# classes = y_natural_train.tolist()
# print("length classes:", len(classes))
# print("length y_train:", y_train.shape[0])

# # # have the model predict the test set
# y_pred = model.majority_classifier(y_train, X_test)
# print(y_train)
y_pred, y_true = model.logistic_regression_loocv(Xtf,y)
# print("Pred Y:", y_pred)
# print("True Y:", y_test)

# # # #evaluate the predictions
# metrics = eval.eval(y_test, y_pred)
# for metric in metrics:
#     print(metric)

print(classification_report(y_true, y_pred, target_names=sub_mlb.classes_))

# # #hierna nog dus de post processing dat het in een txt file gezet wordt hoe semeval hem wil hebben
# # postprocess.save_predictions(y_pred,sub_mlb,id_test)