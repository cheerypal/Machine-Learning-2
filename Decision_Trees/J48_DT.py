from sklearn import tree as sk
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import pandas as pd
plt.rcParams["figure.figsize"] = (30, 20)

data = pd.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pd.read_csv("../training_data/y_train_smpl.csv")


# generate j48 decision tree for various depths and collect average scores
def decisionTree():
    z = []
    for i in range(2, 5):
        print("Tree Depth : " + str(i))
        tree = vis_tree = sk.DecisionTreeClassifier(max_depth=i, random_state=42)
        visualiseTree(vis_tree.fit(data, labels), depth=i)
        print("Starting CV with tree depth : " + str(i) + " .......")
        cross = cross_validate(tree, data, labels, cv=10)
        print("Finished CV with tree depth : " + str(i))
        print("\n### Mean Score for Tree Depth " + str(i) + " : " + str(cross) + " ###\n")
        z.append(cross)
    print(z)


# visualise tree using matplotlib
def visualiseTree(tree, depth):
    print("Plotting Tree ....")
    sk.plot_tree(tree, fontsize=16, filled=True)
    plt.savefig("plots/Decision_Tree_Depth_" + str(depth) + "_Q1.png")
    plt.show()


decisionTree()
