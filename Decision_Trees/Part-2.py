from sklearn import tree as sk
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
# from dtreeviz.trees import dtreeviz
from sklearn import tree


# loading datasets from the csv files
data = pd.read_csv("./x_train_gr_smpl.csv")
labels = pd.read_csv("./y_train_smpl.csv")
y_bin = pd.read_csv("./y_train_smpl_0.csv")
X_test = pd.read_csv("./testing_data/x_test_gr_smpl.csv")
Y_test = pd.read_csv("./testing_data/y_test_smpl.csv")

def visualizeRandomForestClassifier():
    #initializaing the  Random forest classifier
    rf_model = RandomForestClassifier(criterion = 'entropy',random_state=42)
    #fitting the training X and training Y on random forest classifier
    rf_model.fit(data,labels["0"])
    # checking the total number of estimators which are total decision trees
    print("Total estimators : ",len(rf_model.estimators_))
    # Plotting decision trees for the 1st estimator (estimators[0]) with 16 depth
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(rf_model.estimators_[0], feature_names=data.columns, filled=True)
    # Saving image to local folder
    fig.savefig('First_estimator.png')
    # initializing random forest classifier with max_depth 2
    rf_model = RandomForestClassifier(criterion = 'entropy',random_state=42,max_depth=2)
    rf_model.fit(data,labels["0"])
    # plotting 1st estimator with max depth 3
    fig = plt.figure(figsize=(20,20))
    _ = tree.plot_tree(rf_model.estimators_[0], feature_names=data.columns, filled=True)
    fig.savefig('Max_depth_3_tree.png')
visualizeRandomForestClassifier()
