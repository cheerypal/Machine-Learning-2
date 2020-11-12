from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas
X = pandas.read_csv("../training_data/x_train_gr_smpl.csv")
y = pandas.read_csv("../training_data/y_train_smpl.csv")

clf = MLPClassifier(random_state=1, max_iter=300).fit(X, y)




