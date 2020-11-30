from sklearn import tree as sk
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import Metrics as mt

# total training data
data = pd.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pd.read_csv("../training_data/y_train_smpl.csv")
testingData = pd.read_csv("../testing_data/x_test_gr_smpl.csv")
testingLabels = pd.read_csv("../testing_data/y_test_smpl.csv")


# generate j48 decision tree for various depths and present the accuracy with a confusion matrix
def decisionTree(tree, visualise, mean_std):
    # Run the J48 tree classifier
    print("\nJ48 Starting ....")

    # visualise tree
    if visualise:
        visualiseTree(DT.fit(data, labels), save=False)

    # Run cross validation to find the mean score and use neg_mean_squared_error
    if mean_std:
        print("Starting CV ....")
        cross = cross_val_score(tree, data, labels, scoring="neg_mean_squared_error", cv=10)
        tree_scores = np.sqrt(-cross)
        print("Finished CV ")

        # Present means score and standard deviation for the cross validation
        print("\n### Mean Score : " + str(tree_scores.mean()) + " ###")
        print("### STD Score " + str(tree_scores.std()) + " ###\n")

    # Run cross validation to find the prediction on the data and labels then print the confusion matrix
    print("Starting confusion matrix ....")
    cross_pred = cross_val_predict(tree, data, labels, cv=10)
    print("Confusion Matrix:\n", metrics.confusion_matrix(labels, cross_pred))

    # get precision, recall and f1 measure
    print("\n", metrics.classification_report(labels, cross_pred))

    # get tpr, fpr and ROC area
    print("\n", mt.get_TPR_FPR(cross_pred, "train", visualise))
    print("\n", mt.get_ROC_AREA(cross_pred, "train"))

    # Get accuracy of the cross validation
    accuracy = accuracy_score(labels, cross_pred)
    print("\nAccuracy: " + str(accuracy))


# visualise tree using matplotlib
def visualiseTree(tree, save):
    plt.figure(figsize=(50, 50))
    print("Plotting Tree ....")
    sk.plot_tree(tree, fontsize=16, filled=True)
    if save:
        plt.savefig("plots/Decision_Tree_Q1.png")
    plt.show()


# initialise decision tree classifier
DT = sk.DecisionTreeClassifier(max_depth=None, max_features=2, random_state=42)

# Question 1
#decisionTree(tree=DT, visualise=True, mean_std=False)

# Question 2
#visualiseTree(DT.fit(data, labels), save=False)

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
