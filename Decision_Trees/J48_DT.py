from sklearn import tree as sk
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.figsize"] = (50, 50)

data = pd.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pd.read_csv("../training_data/y_train_smpl.csv")
X_test = pd.read_csv("../testing_data/x_test_gr_smpl.csv")
Y_test = pd.read_csv("../testing_data/y_test_smpl.csv")


# generate j48 decision tree for various depths and collect average scores
def decisionTree():
    print("\nJ48 Starting ...")
    tree = sk.DecisionTreeClassifier(max_depth=None, random_state=42)
    visualiseTree(tree.fit(data, labels))
    print("Starting CV .......")
    cross = cross_val_score(tree, data, labels, scoring="neg_mean_squared_error", cv=10)
    tree_scores = np.sqrt(-cross)
    print("Finished CV ")
    print("\n### Mean Score : " + str(tree_scores.mean()) + " ###")
    print("### STD Score " + str(tree_scores.std()) + " ###\n")
    cross_pred = cross_val_predict(tree, data, labels, cv=10)
    print("Confusion Matrix:\n", metrics.confusion_matrix(labels, cross_pred))


# visualise tree using matplotlib
def visualiseTree(tree):
    print("Plotting Tree ....")
    sk.plot_tree(tree, fontsize=16, filled=True)
    plt.savefig("plots/Decision_Tree_Q1.png")
    plt.show()


decisionTree()
