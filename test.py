import data
import model
import eval
import postprocess
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
X = dat["text"]
y = sub_narr_enc

# vectorize the X data for the logreg model - ignore cus were not using it rn
#hoort ook eig beetje in preprocessing 
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(X)

#divide the data into train and test splits, small test size, cause little train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
# print(y_train.shape)
# print(y_test.shape)

# have the model predict the test set
y_pred = model.most_common(X_train, y_train, X_test)
print("TEST Y:", y_test)

# #evaluate the predictions
# metrics = eval.eval(y_test, y_pred)
# for metric in metrics:
#     print(metric)

print(classification_report(y_test, y_pred, target_names=sub_mlb.classes_))

#hierna nog dus de post processing dat het in een txt file gezet wordt hoe semeval hem wil hebben