from sklearn.linear_model import LinearRegression
import pandas
import numpy
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

lin_reg = LinearRegression()

X = pandas.read_csv("../training_data/x_train_gr_smpl.csv")
y = pandas.read_csv("../training_data/y_train_smpl.csv")

lin_reg.fit(X, y)

lin_scores = cross_val_score(lin_reg, X, y, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = numpy.sqrt(-lin_scores)
display_scores(lin_rmse_scores)