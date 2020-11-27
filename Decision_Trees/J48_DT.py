from sklearn import tree as sk
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

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
    print("\n", get_TPR_FPR(cross_pred, "train", visualise))
    print("\n", get_ROC_AREA(cross_pred, "train"))

    # Get accuracy of the cross validation
    accuracy = accuracy_score(labels, cross_pred)
    print("\nAccuracy: " + str(accuracy))


# get ROC using the binary files
# fileType can either be "train" or "test"
# returns data frame containing the ROC area for each binary file
def get_ROC_AREA(prediction, fileType):
    auc_arr = list()
    for i in range(0, 10):
        if fileType != "4000" and fileType != "9000":
            auc_arr.append(metrics.roc_auc_score(pd.read_csv("../" + fileType + "ing_data/y_" + fileType + "_smpl_" +
                                                             str(i) + ".csv"), prediction))
        else:
            auc_arr.append(metrics.roc_auc_score(pd.read_csv("../" + fileType + "_data/y_test_smpl_" +
                                                             str(i) + ".csv_" + fileType + ".csv"), prediction))
    return pd.DataFrame(data=auc_arr, columns=["ROC Area"])


# get tpr and fpr rate for certain binary files.
# fileType can either be "train" or "test"
# returns data frame containing the True positive rate and false positive rate for each binary file
def get_TPR_FPR(prediction, fileType, visualise):
    rates = list()

    for i in range(0, 10):
        if fileType != "4000" and fileType != "9000":
            fpr, tpr, thresholds = metrics.roc_curve(
                pd.read_csv("../" + fileType + "ing_data/y_" + fileType + "_smpl_" +
                            str(i) + ".csv"), prediction)

            rates.append([np.mean(tpr), np.mean(fpr)])
            if visualise:
                plot_roc_curve(i, fpr, tpr)

        else:
            fpr, tpr, thresholds = metrics.roc_curve(pd.read_csv("../" + fileType + "_data/y_test_smpl_" +
                                                                 str(i) + ".csv_" + fileType + ".csv"), prediction)

            rates.append([np.mean(tpr), np.mean(fpr)])
            if visualise:
                plot_roc_curve(i, fpr, tpr)

    return pd.DataFrame(data=rates, columns=["Average tpr", "Average fpr"])


# visualise tree using matplotlib
def visualiseTree(tree, save):
    plt.figure(figsize=(50, 50))
    print("Plotting Tree ....")
    sk.plot_tree(tree, fontsize=16, filled=True)
    if save:
        plt.savefig("plots/Decision_Tree_Q1.png")
    plt.show()


# plots ROC curve given FPR and TPR values
def plot_roc_curve(label, fpr, tpr):
    title = "Rate for class " + str(label)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.suptitle(title)
    plt.grid(True)
    plt.show()


# Question 3
# test method for when test data needs to be used.
def decision_trees_test_data(tree, testType, train_data, train_labels, testData, testLabels, visualise):
    print("Starting .... \n")
    decTree = tree.fit(train_data, train_labels)
    if visualise:
        visualiseTree(decTree, save=False)

    # predicts the labels from the test data given
    pred = tree.predict(testData)
    print("Starting Confusion Matrix ....")
    # plots confusion matrix between testingLabels and the predicted labels
    print("\nConfusion Matrix\n", metrics.confusion_matrix(testLabels, pred))
    print("\n")
    print("\n", metrics.classification_report(testLabels, pred))

    # get tpr, fpr and ROC area
    print("\n", get_TPR_FPR(pred, fileType=testType, visualise=visualise))
    print("\n", get_ROC_AREA(pred, fileType=testType))

    accuracy = accuracy_score(testLabels, pred)
    print("\nAccuracy: " + str(accuracy))


# initialise decision tree classifier
DT = sk.DecisionTreeClassifier(max_depth=None, max_features=5, random_state=42)

# Question 1
#decisionTree(tree=DT, visualise=False, mean_std=False)

# Question 3
print("\nTesting using dataset testing data ....\n")
#decision_trees_test_data(DT, "test", data, labels, testingData, testingLabels, visualise=False)

# Question 4
print("\nTesting using 4000 moved testing data ....\n")
train_4000 = pd.read_csv("../4000_data/x_train_gr_smpl.csv4000.csv")
train_labels_4000 = pd.read_csv("../4000_data/y_train_smpl.csv4000.csv")
test_4000 = pd.read_csv("../4000_data/x_test_gr_smpl.csv_4000.csv")
test_labels_4000 = pd.read_csv("../4000_data/y_test_smpl.csv_4000.csv")
decision_trees_test_data(DT, "4000", data, labels, test_4000, test_labels_4000, visualise=False)

# Question 5
print("\nTesting using 9000 moved testing data ....\n")
train_9000 = pd.read_csv("../9000_data/x_train_gr_smpl.csv9000.csv")
train_labels_9000 = pd.read_csv("../9000_data/y_train_smpl.csv9000.csv")
test_9000 = pd.read_csv("../9000_data/x_test_gr_smpl.csv_9000.csv")
test_labels_9000 = pd.read_csv("../9000_data/y_test_smpl.csv_9000.csv")
decision_trees_test_data(DT, "9000", data, labels, test_9000, test_labels_9000, visualise=False)
