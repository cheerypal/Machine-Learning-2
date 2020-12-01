from sklearn import tree as sk
import pandas as pd
import Metrics as mt
import Decision_Trees.TFunctions as tf

# total training data
data = pd.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pd.read_csv("../training_data/y_train_smpl.csv")
testingData = pd.read_csv("../testing_data/x_test_gr_smpl.csv")
testingLabels = pd.read_csv("../testing_data/y_test_smpl.csv")


# initialise decision tree classifier
DT = sk.DecisionTreeClassifier(max_depth=None, max_features=2, random_state=42)

# Question 1
tf.crossValidation(tree=DT, data=data, labels=labels, visualise=False, mean_std=False)

# Question 2
tf.visualiseTree(DT.fit(data, labels), save=False)

# Question 3
print("\nTesting using dataset testing data ....\n")
mt.classifier_tester(DT, "test", data, labels, testingData, testingLabels, visualise=False)

# Question 4
print("\nTesting using 4000 moved testing data ....\n")
train_4000 = pd.read_csv("../4000_data/x_train_gr_smpl.csv4000.csv")
train_labels_4000 = pd.read_csv("../4000_data/y_train_smpl.csv4000.csv")
test_4000 = pd.read_csv("../4000_data/x_test_gr_smpl.csv_4000.csv")
test_labels_4000 = pd.read_csv("../4000_data/y_test_smpl.csv_4000.csv")
mt.classifier_tester(DT, "4000", train_4000, train_labels_4000, test_4000, test_labels_4000, visualise=False)

# Question 5
print("\nTesting using 9000 moved testing data ....\n")
train_9000 = pd.read_csv("../9000_data/x_train_gr_smpl.csv9000.csv")
train_labels_9000 = pd.read_csv("../9000_data/y_train_smpl.csv9000.csv")
test_9000 = pd.read_csv("../9000_data/x_test_gr_smpl.csv_9000.csv")
test_labels_9000 = pd.read_csv("../9000_data/y_test_smpl.csv_9000.csv")
mt.classifier_tester(DT, "9000", train_9000, train_labels_9000, test_9000, test_labels_9000, visualise=False)
