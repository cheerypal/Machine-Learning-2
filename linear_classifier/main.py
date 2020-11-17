import pandas
import numpy
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, precision_score, recall_score, f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

training_data = pandas.read_csv("../training_data/x_train_gr_smpl.csv")
labels = pandas.read_csv("../training_data/y_train_smpl.csv")
y_bin = pandas.read_csv("../training_data/y_train_smpl_0.csv")
labels = numpy.ravel(labels)

# Always scale the input. The most convenient way is to use a pipeline.
linear_classifier = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
label_predictions = cross_val_predict(linear_classifier, training_data, labels, cv=10)
confusion_matrix = confusion_matrix(labels, label_predictions)

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