from sklearn import tree as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# loading datasets from the csv files
data = pd.read_csv("./x_train_gr_smpl.csv")
labels = pd.read_csv("./y_train_smpl.csv")
y_bin = pd.read_csv("./y_train_smpl_0.csv")

def randomForestClassifier():
    #initializaing Random forest classifier
    rf_model = RandomForestClassifier(random_state=1)
    print("Getting cross_validation score with kfold 10")
    rf_cross_score = cross_val_score(rf_model, data, labels["0"], scoring="neg_mean_squared_error", cv=10)

    # Building confusion matrix for rf using cross validation predition
    print("Getting predictions with cross validation prediction method")
    cross_pred = cross_val_predict(rf_model, data, labels["0"], cv=10)
    print("Confusion Matrix:\n", metrics.confusion_matrix(labels["0"], cross_pred))

    print("Printing Different accuracy measures for Analyzing Random Forest")
    print("Mean accuracy score - Random Forest: ", rf_cross_score.mean())
    print("ROC_AREA : ", metrics.roc_auc_score(y_bin, cross_pred))
    tpr, fpr, thresholds = metrics.roc_curve(y_bin, cross_pred)
    print("TP Rate : ", tpr)
    print("FP Rate : ", fpr)
    print("Thresholds : ", thresholds)
    print("Precision : ", metrics.precision_score(labels["0"], cross_pred, average='micro'))
    print("Recall : ", metrics.recall_score(labels["0"], cross_pred, average='micro'))
    print("F1 Measure : ", metrics.f1_score(labels["0"], cross_pred, average='micro'))

    accuracy = accuracy_score(labels["0"], cross_pred)
    print("Cross validation Accuracy: " + str(accuracy))
randomForestClassifier()
