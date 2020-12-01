from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import Decision_Trees.TFunctions as tf
import Metrics as mt

# loading datasets from the csv files
data = pd.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pd.read_csv("../training_data/y_train_smpl.csv")
testingData = pd.read_csv("../testing_data/x_test_gr_smpl.csv")
testingLabels = pd.read_csv("../testing_data/y_test_smpl.csv")
labels = np.ravel(labels)

# initializing Random forest classifier
rf_model = RandomForestClassifier(max_depth=None, max_features=2, random_state=1)

# Question 1
tf.crossValidation(rf_model, data, labels, visualise=False, mean_std=False)

# Question 2
rf_model.fit(data, labels)
rf_tree = rf_model.estimators_[0]
tf.visualiseTree(rf_tree, save=False)

# Question 3
print("\nTesting using a dataset testing data ....\n")
mt.classifier_tester(rf_model, "test", data, labels, testingData, testingLabels, visualise=False)

# Question 4
print("\nTesting using 4000 moved testing data ....\n")
train_4000 = pd.read_csv("../4000_data/x_train_gr_smpl.csv4000.csv")
train_labels_4000 = pd.read_csv("../4000_data/y_train_smpl.csv4000.csv")
test_4000 = pd.read_csv("../4000_data/x_test_gr_smpl.csv_4000.csv")
test_labels_4000 = pd.read_csv("../4000_data/y_test_smpl.csv_4000.csv")
train_labels_4000 = np.ravel(train_labels_4000)
mt.classifier_tester(rf_model, "4000", train_4000, train_labels_4000, test_4000, test_labels_4000, visualise=False)

# Question 5
print("\nTesting using 9000 moved testing data ....\n")
train_9000 = pd.read_csv("../9000_data/x_train_gr_smpl.csv9000.csv")
train_labels_9000 = pd.read_csv("../9000_data/y_train_smpl.csv9000.csv")
test_9000 = pd.read_csv("../9000_data/x_test_gr_smpl.csv_9000.csv")
test_labels_9000 = pd.read_csv("../9000_data/y_test_smpl.csv_9000.csv")
train_labels_9000 = np.ravel(train_labels_9000)
mt.classifier_tester(rf_model, "9000", train_9000, train_labels_9000, test_9000, test_labels_9000, visualise=False)
