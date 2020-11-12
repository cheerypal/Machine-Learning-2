from sklearn.linear_model import LinearRegression
import pandas
import numpy
from sklearn.model_selection import cross_val_score


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


linear_classifier = LinearRegression()

training_data = pandas.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pandas.read_csv("../training_data/y_train_smpl.csv")

linear_classifier.fit(training_data, labels)

linear_scores = cross_val_score(linear_classifier, training_data, labels, scoring="neg_mean_squared_error", cv=10)
display_scores(numpy.sqrt(-linear_scores))
