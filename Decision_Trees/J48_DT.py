from sklearn import tree as sk
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

plt.rcParams["figure.figsize"] = (50, 50)

data = pd.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pd.read_csv("../training_data/y_train_smpl.csv")
X_test = pd.read_csv("../testing_data/x_test_gr_smpl.csv")
Y_test = pd.read_csv("../testing_data/y_test_smpl.csv")
y_bin = pd.read_csv("../training_data/y_train_smpl_0.csv")


# generate j48 decision tree for various depths and present the accuracy with a confusion matrix
def decisionTree(tree):
    # Run the J48 tree classifier
    print("\nJ48 Starting ...")
    visualiseTree(DT.fit(data, labels))
    # Run cross validation to find the mean score and use neg_mean_squared_error
    print("Starting CV .......")
    cross = cross_val_score(tree, data, labels, scoring="neg_mean_squared_error", cv=10)
    tree_scores = np.sqrt(-cross)
    print("Finished CV ")

    # Present means score and standard deviation for the cross validation
    print("\n### Mean Score : " + str(tree_scores.mean()) + " ###")
    print("### STD Score " + str(tree_scores.std()) + " ###\n")

    # Run cross validation to find the prediction on the data and labels then print the confusion matrix
    print("Starting confusion matrix .....")
    cross_pred = cross_val_predict(tree, data, labels, cv=10)
    print("Confusion Matrix:\n", metrics.confusion_matrix(labels, cross_pred))
    # tpr, fpr, auc = metrics.roc_curve(labels, cross_pred)
    """ y_pred = label_binarize(cross_pred, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_test = label_binarize(labels, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print(roc_auc)"""
    print(metrics.classification_report(labels, cross_pred))



    """
    print("Thresholds : ", auc)
    print("Precision : ", metrics.precision_score(labels, cross_pred, average='micro'))
    print("Recall : ", metrics.recall_score(labels, cross_pred, average='micro'))
    print("F1 Measure : ", metrics.f1_score(labels, cross_pred, average='micro'))
    print("ALL : ", metrics.precision_recall_fscore_support(labels, cross_pred, average='macro'))
    """

    # Get accuracy of the cross validation
    accuracy = accuracy_score(labels, cross_pred)
    print("Accuracy: " + str(accuracy))


def plot_roc(model, X_test, y_test):
    # calculate the fpr and tpr for all thresholds of the classification
    probabilities = model.predict_proba(np.array(X_test))
    predictions = probabilities[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# visualise tree using matplotlib
def visualiseTree(tree):
    print("Plotting Tree ....")
    sk.plot_tree(tree, fontsize=16, filled=True)
    plt.savefig("plots/Decision_Tree_Q1.png")
    plt.show()


def decision_trees_test_data(tree):
    tree.fit(data, labels)
    prob = tree.predict(X_test)
    print(prob)
    print(len(prob))
    metrics.confusion_matrix()


DT = sk.DecisionTreeClassifier(max_depth=6, random_state=42)

decisionTree(DT)
