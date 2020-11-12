from sklearn import tree as sk
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.figsize"] = (30, 20)

data = pd.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pd.read_csv("../training_data/y_train_smpl.csv")


# generate j48 decision tree for various depths and collect average scores
def decisionTree():
    z = []
    for i in range(2, 6):
        print("Tree Depth : " + str(i))
        tree = sk.DecisionTreeClassifier(max_depth=i, random_state=42).fit(data, labels)
        visualiseTree(tree.fit(data, labels), depth=i)
        print("Starting CV with tree depth : " + str(i) + " .......")
        cross = cross_val_score(tree, data, labels, scoring="neg_mean_squared_error", cv=10)
        tree_scores = np.sqrt(-cross)
        print("Finished CV with tree depth : " + str(i))
        print("\n### Mean Score for Tree Depth " + str(i) + " : " + str(tree_scores.mean()) + " ###")
        print("### STD Score for Tree Depth " + str(i) + " : " + str(tree_scores.std()) + " ###\n")
        z.append(tree_scores)
    print(z)


# visualise tree using matplotlib
def visualiseTree(tree, depth):
    print("Plotting Tree ....")
    sk.plot_tree(tree, fontsize=16, filled=True)
    plt.savefig("plots/Decision_Tree_Depth_" + str(depth) + "_Q1.png")
    plt.show()


decisionTree()
