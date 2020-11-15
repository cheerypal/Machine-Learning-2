from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import numpy

import pandas
training_data = pandas.read_csv("../training_data/x_train_gr_smpl.csv")
X = pandas.read_csv("../training_data/x_train_gr_smpl.csv")
y = pandas.read_csv("../training_data/y_train_smpl.csv")
y_bin = pandas.read_csv("../training_data/y_train_smpl_0.csv")
labels = numpy.ravel(y)


print("MLP classifier starting")
clf = MLPClassifier(random_state=1, max_iter=300).fit(X, labels)
print("MLP classifier finished")

print("Cross val starting")
label_predictions = cross_val_predict(clf, training_data, labels, cv=10)
print("Cross val finished")

print("Building confusion matrix")
confusion_matrix = confusion_matrix(labels, label_predictions)
print("confusion matrix complete")


print(confusion_matrix)


tpr, fpr, thresholds = roc_curve(y_bin, label_predictions)
print("TP Rate : ", tpr)
print("FP Rate : ", fpr)
print("Thresholds : ", thresholds)
print("Precision : ", precision_score(labels, label_predictions, average='micro'))
print("Recall : ", recall_score(labels, label_predictions, average='micro'))
print("F1 Measure : ", f1_score(labels, label_predictions, average='micro'))

accuracy = accuracy_score(labels, label_predictions)

print("Accuracy: " + str(accuracy))




