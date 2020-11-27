from sklearn import tree as sk
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score

# loading datasets from the csv files
data = pd.read_csv("./x_train_gr_smpl.csv")
labels = pd.read_csv("./y_train_smpl.csv")
y_bin = pd.read_csv("./y_train_smpl_0.csv")
X_test = pd.read_csv("./testing_data/x_test_gr_smpl.csv")
Y_test = pd.read_csv("./testing_data/y_test_smpl.csv")

def randomForestClassifier():
    #initializaing Random forest classifier
    rf_model = RandomForestClassifier(random_state=1)
    #fitting the training X and training Y on random forest classifier
    rf_model.fit(data,labels["0"])
    #getting the predictions on Test data in order to achieve predicted classes
    y_pred = rf_model.predict(X_test)
    print("Printing Different accuracy measures for Analyzing Random Forest")

    print("Different accuracy measures for Analyzing Random Forest")
    print("Precision : ", metrics.precision_score(Y_test, y_pred, average='micro'))
    print("Recall : ", metrics.recall_score(Y_test, y_pred, average='micro'))
    print("F1 Measure : ", metrics.f1_score(Y_test, y_pred, average='micro'))
    accuracy = accuracy_score(Y_test, y_pred)
    print("Accuracy: " + str(accuracy))
randomForestClassifier()
