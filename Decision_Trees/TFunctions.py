import matplotlib.pyplot as plt
from sklearn import tree as sk
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import numpy as np
import Metrics as mt
import pandas as pd


# visualise tree using tree.plot_tree
# Inputs : tree - tree to be visualised - RandomForrest and DecisionTree
# Inputs : save - boolean for saving the file in the file structure
# Outputs : Plot of the tree
def visualiseTree(tree, save):
    plt.figure(figsize=(50, 50))
    print("\nPlotting Tree ....")
    sk.plot_tree(tree, fontsize=16, filled=True)
    if save:
        plt.savefig("plots/Decision_Tree_Q1.png")
    plt.show()


# runs the cross validation algorithms on the training data using the tree
# Inputs : tree - this is the tree classifier used for cross validation
# Inputs : data, labels - corresponding data and classes of the dataset
# Inputs : Visualise - this is a boolean param when True it will plot the ROC Curves
# Inputs : mean_std - this is a boolean param when True will print the mean and std of cross validation
# Outputs : Confusion matrix of cross validation, Precision, Recall, F1-Measure, ROC Curves, ROC Area and Accuracy
# Outputs : ROC Curves if visualise is True
def crossValidation(classifier, data, labels, visualise, mean_std):
    print("Cross Validation started ....")
    auc_arr = []
    if mean_std:
        print("Starting CV ....")
        cross = cross_val_score(classifier, data, labels, scoring="neg_mean_squared_error", cv=10)
        tree_scores = np.sqrt(-cross)
        print("Finished CV ")

        # Present means score and standard deviation for the cross validation
        print("\n### Mean Score : " + str(tree_scores.mean()) + " ###")
        print("### STD Score " + str(tree_scores.std()) + " ###\n")

    # Predict classes for confusion matrix
    print("Start confusion matrix ....")
    prediction = cross_val_predict(classifier, data, labels, cv=10)
    print("Confusion Matrix:\n", metrics.confusion_matrix(labels, prediction))

    # get precision, recall and f1 measure
    print("\n", metrics.classification_report(labels, prediction))
    # loops through each of the classes
    for i in range(0,10):
        # gets cross val probability of the class
        file = np.ravel(pd.read_csv("../training_data/y_train_smpl_"+str(i)+".csv"))
        print("ROC class: ", i)
        # get ROC values and TPR, FPR for ROC curve
        probs = cross_val_predict(classifier, data, file, method="predict_proba", cv=10)
        fpr, tpr, thresholds = metrics.roc_curve(file, probs[:,1])
        auc_arr.append(metrics.roc_auc_score(file, probs[:,1]))

        if visualise:
            plt.plot(fpr, tpr, label="Class " + str(i))

    plt.title("ROC Visualisation")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # get tpr, fpr and ROC area using the predicted labels
    print(pd.DataFrame(data=auc_arr, columns=["ROC Area"]))

    # Get accuracy of the cross validation
    accuracy = metrics.accuracy_score(labels, prediction)
    print("\nAccuracy: " + str(accuracy))
