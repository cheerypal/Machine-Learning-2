import matplotlib.pyplot as plt
from sklearn import tree as sk
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import numpy as np
import Metrics as mt


# visualise tree using matplotlib
def visualiseTree(tree, save):
    plt.figure(figsize=(50, 50))
    print("\nPlotting Tree ....")
    sk.plot_tree(tree, fontsize=16, filled=True)
    if save:
        plt.savefig("plots/Decision_Tree_Q1.png")
    plt.show()


def crossValidation(tree, data, labels, visualise, mean_std):
    print("Cross Validation started ....")

    if mean_std:
        print("Starting CV ....")
        cross = cross_val_score(tree, data, labels, scoring="neg_mean_squared_error", cv=10)
        tree_scores = np.sqrt(-cross)
        print("Finished CV ")

        # Present means score and standard deviation for the cross validation
        print("\n### Mean Score : " + str(tree_scores.mean()) + " ###")
        print("### STD Score " + str(tree_scores.std()) + " ###\n")

    print("Start confusion matrix ....")
    prediction = cross_val_predict(tree, data, labels, cv=10)
    print("Confusion Matrix:\n", metrics.confusion_matrix(labels, prediction))

    # get precision, recall and f1 measure
    print("\n", metrics.classification_report(labels, prediction))

    # get tpr, fpr and ROC area
    print("\n", mt.get_TPR_FPR(prediction, "train", visualise))
    print("\n", mt.get_ROC_AREA(prediction, "train"))

    # Get accuracy of the cross validation
    accuracy = metrics.accuracy_score(labels, prediction)
    print("\nAccuracy: " + str(accuracy))
