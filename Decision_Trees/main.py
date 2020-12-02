# Decision Trees
# This is the main file that will be ran for the decision tree half of the coursework
# F20DL CW2
from sklearn import tree as sk
from sklearn.ensemble import RandomForestClassifier
from Decision_Trees import J48_DT, RF_DT
import pandas as pd
import numpy as np

# total training data
data = pd.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pd.read_csv("../training_data/y_train_smpl.csv")
testingData = pd.read_csv("../testing_data/x_test_gr_smpl.csv")
testingLabels = pd.read_csv("../testing_data/y_test_smpl.csv")

# ==================================== Decision Tree ====================================
DT = sk.DecisionTreeClassifier(max_depth=None, max_features=4, min_samples_split=4, random_state=42)
# run J48 decision tree
print("\nRunning J48 ....\n")
J48_DT.runJ48(DT, vis_tree=True, visualise=False, mean_std=False, data=data, labels=labels,
              testingData=testingData, testingLabels=testingLabels, save=False)

# =================================== Random Forrest ====================================
rf_model = RandomForestClassifier(max_depth=None, max_features=4, min_samples_split=4, random_state=42)
# run RF tree
print("\nRunning Random Forrest Classifier ....\n")
labels = np.ravel(labels)
RF_DT.runRF(rf_model, vis_tree=True, visualise=False, mean_std=False, data=data, labels=labels,
            testingData=testingData, testingLabels=testingLabels, save=False)

