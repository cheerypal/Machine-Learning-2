from sklearn import linear_model
from sklearn.decomposition import PCA
import numpy
import pandas
from sklearn import svm
from sklearn.model_selection import KFold

X = pandas.read_csv("../training_data/x_train_gr_smpl.csv")
y = pandas.read_csv("../training_data/y_train_smpl_0.csv")

X_test = pandas.read_csv("../testing_data/x_test_gr_smpl.csv")
y_test = pandas.read_csv("../testing_data/y_test_smpl_0.csv")


y = numpy.ravel(y)
clf = svm.SVC()
clf.fit(X, y)
print(clf)
print(clf.intercept_)

scores = []
best_svr = clf # i dont know about this one
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_svr.fit(X_train, y_train)
    scores.append(best_svr.score(X_test, y_test))

best_svr.fit(X, y)
scores.append(best_svr.score(X_test, y_test))

print(numpy.mean(scores))

"""
x = pandas.read_csv("training_data/x_train_gr_smpl.csv")

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pandas.DataFrame(data=principalComponents, columns=['principal component 1', "principal component 2"])

finalDf = pandas.concat([principalDf, y], axis=1)
"""